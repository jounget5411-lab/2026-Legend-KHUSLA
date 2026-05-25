#!/usr/bin/env python3
"""
통합 경로계획 — 상태머신 WAIT → CONE → LANE.

상태:
  WAIT: 정지. YOLO 초록불(GREEN=5) 감지 시 CONE으로 전환.
  CONE: 라바콘 주행. 왼쪽 콘 줄 2차 피팅 + 2.2m 오프셋 = 중앙선.
        콘 miss 30회(3초) 연속 시 LANE으로 전환.
  LANE: 차선 주행. 노란 중앙선(cls_id=8) 기반 2차 피팅.
        lane_planner.py의 plan() 함수 — 친구(sm) 원본 그대로.

구독:
  /fused/obstacles  — 라이다 장애물 (CONE 모드용)
  /fused/lane       — 차선 점 (LANE 모드용, z=cls_id: 8=노란, 6=흰)
  /detect/events_raw — YOLO 이벤트 (WAIT에서 초록불 감지)

발행:
  /center_path  — 중앙 경로 (PoseArray, motion이 추종)
  /target       — 목표점 (PointStamped)
  /lane_left    — 좌측 경계 (시각화용)
  /lane_right   — 우측 경계 (시각화용)
  /lane_fits    — 피팅 결과 (LANE 모드 디버그용)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# LANE 모드: 친구 plan() + 헬퍼 (lane_planner.py = 친구 path_planner_node.py 원본)
from .lane_planner import plan as lane_plan
from .lane_planner import (
    YELLOW_CLS_IDS, WHITE_CLS_IDS, PLAN_HZ, TARGET_X, TARGET_Y_LIMIT,
    _polyval_clipped, _sample_xs, _yellow_inlier_xs, _yellow_inlier_ys,
)

# ======================== CONE 상수 ========================
# 라바콘 구간: 라이다 /fused/obstacles에서 왼쪽 콘 줄을 피팅해
# 트랙 절반(2.2m) 오프셋으로 중앙선을 만든다.

CONE_X_MIN = 0.3           # 전방 ROI 최소 (m)
CONE_X_MAX = 10.0          # 전방 ROI 최대 (m)
CONE_Y_MIN = -6.0          # 좌우 ROI (m)
CONE_Y_MAX = 6.0
TRACK_WIDTH = 4.4           # 실측 트랙 폭 (m)
TRACK_HALF_WIDTH = TRACK_WIDTH / 2.0

CONE_FIT_MIN_POINTS = 2
CONE_FIT_MIN_X_SPAN = 0.8
CONE_FIT_CURVE_MAX = 1.5
CONE_FIT_SLOPE_MAX = 3.0
CONE_LEFT_ACCEPT_DIST = 1.5
CONE_FIT_SMOOTH_ALPHA = 0.20
CONE_FIT_MAX_MISS = 8

CONE_SAMPLE_X_START = 0.5
CONE_SAMPLE_X_END = 6.0
CONE_SAMPLE_N = 25

CONE_TO_LANE_MISS = 30
CONE_GRACE_TICKS = 30

# ======================== YOLO 이벤트 cls_id ========================

GREEN_CLS_ID = 5
RED_CLS_ID = 9
YELLOW_LIGHT_CLS_ID = 11
LEFT_SIGN_CLS_ID = 7
CHILD_START_CLS_ID = 3
CHILD_END_CLS_ID = 1

# ======================== 헬퍼 ========================

def _poses_from_xy(stamp, xs, ys):
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = "lidar_frame"
    for x, y in zip(xs, ys):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        msg.poses.append(p)
    return msg


def _make_pose_array(stamp, xs, ys):
    out = PoseArray()
    out.header.stamp = stamp
    out.header.frame_id = "lidar_frame"
    for x, y in zip(xs, ys):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        out.poses.append(p)
    return out


# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self.phase = "WAIT"

        # CONE 데이터
        self._obstacles = []
        self._cone_prev_fit = None
        self._cone_miss = 0
        self._cone_grace = 0

        # LANE 데이터 (친구 _on_lane과 동일 구조)
        self._yellow_xs = np.array([], dtype=np.float64)
        self._yellow_ys = np.array([], dtype=np.float64)
        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._has_lane = False

        # YOLO 이벤트
        self._events = []

        # 구독
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/detect/events_raw", self._on_events, 10)

        # 발행
        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)
        self._pub_fits = self.create_publisher(PoseArray, "/lane_fits", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self._log_counter = 0
        self.get_logger().info("path_planner_node started (WAIT→CONE→LANE)")

    # ---- 콜백 ----

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y, p.position.z)
                           for p in msg.poses]

    def _on_lane(self, msg: PoseArray):
        """친구 path_planner_node._on_lane 그대로 — cls_id로 노란/흰 분리."""
        self._has_lane = True
        if not msg.poses:
            self._yellow_xs = np.array([], dtype=np.float64)
            self._yellow_ys = np.array([], dtype=np.float64)
            self._white_xs = np.array([], dtype=np.float64)
            self._white_ys = np.array([], dtype=np.float64)
            return
        xs = np.fromiter((p.position.x for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        ys = np.fromiter((p.position.y for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        cls = np.fromiter((int(p.position.z) for p in msg.poses),
                          dtype=np.int32, count=len(msg.poses))
        ym = np.isin(cls, list(YELLOW_CLS_IDS))
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        self._yellow_xs = xs[ym]
        self._yellow_ys = ys[ym]
        self._white_xs = xs[wm]
        self._white_ys = ys[wm]

    def _on_events(self, msg: PoseArray):
        self._events = [int(p.position.z) for p in msg.poses]

    # ---- 메인 틱 ----

    def _tick(self):
        stamp = self.get_clock().now().to_msg()

        if self.phase == "WAIT":
            self._tick_wait(stamp)
        elif self.phase == "CONE":
            self._tick_cone(stamp)
        elif self.phase == "LANE":
            self._check_lane_events()
            self._tick_lane(stamp)
        elif self.phase == "STOP_SIGNAL":
            self._tick_stop_signal(stamp)
        elif self.phase == "TURN_LEFT":
            self._tick_turn_left(stamp)
        elif self.phase == "SCHOOL_ZONE":
            self._tick_school_zone(stamp)

        # 로그
        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            self.get_logger().info(f"phase={self.phase}")

    # ---- WAIT ----

    def _tick_wait(self, stamp):
        if GREEN_CLS_ID in self._events:
            self.get_logger().info("GREEN detected → CONE")
            self.phase = "CONE"
            self._cone_grace = CONE_GRACE_TICKS
            self._cone_miss = 0

    # ---- CONE ----

    def _tick_cone(self, stamp):
        sample_xs = np.linspace(CONE_SAMPLE_X_START, CONE_SAMPLE_X_END, CONE_SAMPLE_N)

        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size > 0:
            roi = ((all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
                   & (all_y >= CONE_Y_MIN) & (all_y <= CONE_Y_MAX))
            all_x, all_y = all_x[roi], all_y[roi]

        left_fit = None
        if all_x.size >= CONE_FIT_MIN_POINTS:
            if self._cone_prev_fit is not None:
                expected_y = np.polyval(self._cone_prev_fit, all_x)
                left_mask = np.abs(all_y - expected_y) < CONE_LEFT_ACCEPT_DIST
            else:
                left_mask = all_y > 0

            lx, ly = all_x[left_mask], all_y[left_mask]
            if lx.size >= CONE_FIT_MIN_POINTS:
                x_span = float(np.max(lx) - np.min(lx)) if lx.size >= 2 else 0.0
                if x_span >= CONE_FIT_MIN_X_SPAN:
                    try:
                        deg = 2 if lx.size >= 3 and x_span >= 2.0 else 1
                        w = 1.0 / (1.0 + lx * lx)
                        coef = np.polyfit(lx, ly, deg, w=w)
                        if deg == 1:
                            coef = np.array([0.0, coef[0], coef[1]])
                        a, b, _ = coef
                        if abs(a) <= CONE_FIT_CURVE_MAX and abs(b) <= CONE_FIT_SLOPE_MAX:
                            left_fit = coef
                    except (np.linalg.LinAlgError, ValueError):
                        pass

        if self._cone_grace > 0:
            self._cone_grace -= 1

        if left_fit is None:
            if self._cone_grace <= 0:
                self._cone_miss += 1
            if self._cone_prev_fit is not None and self._cone_miss <= CONE_FIT_MAX_MISS:
                prev = self._cone_prev_fit.copy()
                prev[0] *= max(0.0, 1.0 - self._cone_miss * 0.2)
                left_fit = prev
            else:
                self._cone_prev_fit = None
                if self._cone_grace <= 0 and self._cone_miss >= CONE_TO_LANE_MISS:
                    self.get_logger().info(
                        f"cones gone (miss={self._cone_miss}) → LANE")
                    self.phase = "LANE"
                return
        else:
            if self._cone_prev_fit is None:
                self._cone_prev_fit = np.asarray(left_fit, dtype=np.float64)
            else:
                self._cone_prev_fit = (
                    (1.0 - CONE_FIT_SMOOTH_ALPHA) * self._cone_prev_fit
                    + CONE_FIT_SMOOTH_ALPHA * np.asarray(left_fit, dtype=np.float64))
            left_fit = self._cone_prev_fit
            self._cone_miss = 0

        center_coef = left_fit.copy()
        center_coef[2] -= TRACK_HALF_WIDTH

        left_ys = np.polyval(left_fit, sample_xs)
        center_ys = np.clip(np.polyval(center_coef, sample_xs), -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        right_ys = center_ys - TRACK_HALF_WIDTH

        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, left_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, right_ys))
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        self._publish_target(stamp, TARGET_X, target_y)

    # ---- LANE 이벤트 체크 (YOLO) ----

    def _check_lane_events(self):
        """LANE 주행 중 YOLO 이벤트로 예외 상태 전환."""
        events = self._events

        if RED_CLS_ID in events or YELLOW_LIGHT_CLS_ID in events:
            self.get_logger().info("RED/YELLOW light → STOP_SIGNAL")
            self.phase = "STOP_SIGNAL"
            return

        # TODO: 아래 전환 조건은 나중에 구현
        # if LEFT_SIGN_CLS_ID in events:
        #     self.get_logger().info("LEFT sign → TURN_LEFT")
        #     self.phase = "TURN_LEFT"
        #     return
        #
        # if CHILD_START_CLS_ID in events:
        #     self.get_logger().info("CHILD_START → SCHOOL_ZONE")
        #     self.phase = "SCHOOL_ZONE"
        #     return

    # ---- STOP_SIGNAL: 빨간불 정지, 초록불 복귀 ----

    def _tick_stop_signal(self, stamp):
        # 정지 — center_path 발행 안 함 → motion 정지
        if GREEN_CLS_ID in self._events:
            self.get_logger().info("GREEN detected → LANE")
            self.phase = "LANE"

    # ---- TURN_LEFT: 좌회전 (TODO) ----

    def _tick_turn_left(self, stamp):
        # TODO: 좌회전 로직 구현. 지금은 차선 주행 유지.
        self._tick_lane(stamp)

    # ---- SCHOOL_ZONE: 어린이구역 (TODO) ----

    def _tick_school_zone(self, stamp):
        # TODO: 감속 등. 지금은 차선 주행 유지.
        self._tick_lane(stamp)
        if CHILD_END_CLS_ID in self._events:
            self.get_logger().info("CHILD_END → LANE")
            self.phase = "LANE"

    # ---- LANE (친구 plan() 그대로) ----

    def _tick_lane(self, stamp):
        if not self._has_lane:
            return

        result = lane_plan(self._yellow_xs, self._yellow_ys,
                           self._white_xs, self._white_ys,
                           self._obstacles)

        if result is None:
            return

        target_x, target_y = result["target"]
        sample_xs = result["sample_xs"]

        self._publish_target(stamp, target_x, target_y)
        self._pub_center.publish(_make_pose_array(stamp, sample_xs, result["sample_ys"]))
        self._pub_left.publish(_make_pose_array(stamp, sample_xs, result["left_ys"]))
        self._pub_right.publish(_make_pose_array(stamp, sample_xs, result["right_ys"]))

        all_fits = result.get("all_fits", [])
        if all_fits:
            flat = PoseArray()
            flat.header.stamp = stamp
            flat.header.frame_id = "lidar_frame"
            for coef in all_fits:
                ys = _polyval_clipped(coef, sample_xs)
                for x, y in zip(sample_xs, ys):
                    p = Pose()
                    p.position.x = float(x)
                    p.position.y = float(y)
                    flat.poses.append(p)
            self._pub_fits.publish(flat)

    # ---- 공통 ----

    def _publish_target(self, stamp, x, y):
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = "lidar_frame"
        t.point.x = float(x)
        t.point.y = float(y)
        self._pub_target.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
