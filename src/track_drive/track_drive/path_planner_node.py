#!/usr/bin/env python3
"""
통합 경로계획 — 상태머신 WAIT → CONE → LANE.

WAIT: 정지. YOLO 초록불(GREEN=5) 감지 → CONE.
CONE: 라바콘 주행 (왼쪽 콘 피팅 + 2.2m 오프셋). 콘 사라지면 → LANE.
LANE: 차선 주행 (sm plan — 노란 중앙선 피팅). 나중에 예외 상태 추가.

구독: /fused/obstacles, /fused/lane, /detect/events_raw, /auto_mode
발행: /center_path, /target, /lane_left, /lane_right
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 공통 상수 ========================

PLAN_HZ = 10
TARGET_X = 3.0
TARGET_Y_LIMIT = 4.0

# ======================== CONE 상수 ========================

CONE_X_MIN = 0.3
CONE_X_MAX = 10.0
CONE_Y_MIN = -6.0
CONE_Y_MAX = 6.0
TRACK_WIDTH = 4.4
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

# CONE→LANE 전환: 콘 miss가 이 이상 연속
CONE_TO_LANE_MISS = 30          # 3초 연속 못 찾으면 전환
CONE_GRACE_TICKS = 30           # CONE 진입 후 3초간 miss 카운트 안 함
LANE_MIN_POINTS_FOR_SWITCH = 10

# ======================== LANE 상수 (sm plan 간소화) ========================

LANE_FIT_X_MIN = 0.5
LANE_FIT_X_MAX = 12.0
LANE_FIT_MIN_POINTS = 12
LANE_FIT_MIN_X_SPAN = 0.75
LANE_FIT_CURVE_MAX = 1.30
LANE_FIT_SLOPE_MAX = 2.80
LANE_HALF_WIDTH = 1.50
LANE_SMOOTH_ALPHA = 0.28
LANE_MAX_MISS = 6
LANE_SAMPLE_X_MIN = 1.0
LANE_SAMPLE_X_MAX = 11.0
LANE_SAMPLE_COUNT = 13
LANE_TARGET_SMOOTH_ALPHA = 0.22

# ======================== WAIT 상수 ========================

GREEN_CLS_ID = 5

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


# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self.phase = "WAIT"

        # 데이터
        self._obstacles = []
        self._lane_xs = np.array([])
        self._lane_ys = np.array([])
        self._events = []  # YOLO event cls_id 리스트

        # CONE 상태
        self._cone_prev_fit = None
        self._cone_miss = 0
        self._cone_grace = 0

        # LANE 상태
        self._lane_prev_fit = None
        self._lane_miss = 0
        self._lane_prev_target_y = 0.0

        # 구독
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/detect/events_raw", self._on_events, 10)

        # 발행
        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self._log_counter = 0
        self.get_logger().info("path_planner_node started (WAIT→CONE→LANE)")

    # ---- 콜백 ----

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y, p.position.z)
                           for p in msg.poses]

    def _on_lane(self, msg: PoseArray):
        if msg.poses:
            self._lane_xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
            self._lane_ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)
        else:
            self._lane_xs = np.array([])
            self._lane_ys = np.array([])

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
            self._tick_lane(stamp)

        # 로그
        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            self.get_logger().info(f"phase={self.phase}")

    # ---- WAIT ----

    def _tick_wait(self, stamp):
        # 정지 — /center_path 발행 안 함 → motion이 정지
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

        # 왼쪽 줄 추출
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

        # grace period 카운트다운
        if self._cone_grace > 0:
            self._cone_grace -= 1

        # EMA + miss
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

        # 중앙선
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

    # ---- LANE ----

    def _tick_lane(self, stamp):
        xs = self._lane_xs
        ys = self._lane_ys

        # ROI
        if xs.size > 0:
            roi = ((xs > LANE_FIT_X_MIN) & (xs < LANE_FIT_X_MAX)
                   & (np.abs(ys) < TARGET_Y_LIMIT))
            xs, ys = xs[roi], ys[roi]

        lane_fit = None
        if (xs.size >= LANE_FIT_MIN_POINTS
                and float(np.max(xs) - np.min(xs)) >= LANE_FIT_MIN_X_SPAN):
            try:
                w = 1.0 / (1.0 + xs * xs)
                coef = np.polyfit(xs, ys, 2, w=w)
                a, b, _ = coef
                if abs(a) <= LANE_FIT_CURVE_MAX and abs(b) <= LANE_FIT_SLOPE_MAX:
                    lane_fit = coef
            except (np.linalg.LinAlgError, ValueError):
                pass

        # EMA + miss
        if lane_fit is None:
            self._lane_miss += 1
            if self._lane_prev_fit is not None and self._lane_miss <= LANE_MAX_MISS:
                lane_fit = self._lane_prev_fit
            else:
                self._lane_prev_fit = None
                return
        else:
            if self._lane_prev_fit is None:
                self._lane_prev_fit = np.asarray(lane_fit, dtype=np.float64)
            else:
                self._lane_prev_fit = (
                    (1.0 - LANE_SMOOTH_ALPHA) * self._lane_prev_fit
                    + LANE_SMOOTH_ALPHA * np.asarray(lane_fit, dtype=np.float64))
            lane_fit = self._lane_prev_fit
            self._lane_miss = 0

        sample_xs = np.linspace(LANE_SAMPLE_X_MIN, LANE_SAMPLE_X_MAX, LANE_SAMPLE_COUNT)
        center_ys = np.clip(np.polyval(lane_fit, sample_xs), -TARGET_Y_LIMIT, TARGET_Y_LIMIT)

        left_coef = lane_fit.copy(); left_coef[2] += LANE_HALF_WIDTH
        right_coef = lane_fit.copy(); right_coef[2] -= LANE_HALF_WIDTH

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs,
                               np.polyval(left_coef, sample_xs)))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs,
                                np.polyval(right_coef, sample_xs)))

        raw_y = float(np.clip(np.polyval(lane_fit, TARGET_X),
                              -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        delta = raw_y - self._lane_prev_target_y
        target_y = self._lane_prev_target_y + LANE_TARGET_SMOOTH_ALPHA * delta
        self._lane_prev_target_y = target_y
        self._publish_target(stamp, TARGET_X, target_y)

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
