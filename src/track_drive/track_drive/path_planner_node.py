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
# OVERTAKE 모드: common.py의 오른쪽 실선 기반 경로
from .common import LaneMemory, plan_drive_target

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
CONE_FIT_MAX_MISS = 20

CONE_SAMPLE_X_START = 0.5
CONE_SAMPLE_X_END = 6.0
CONE_SAMPLE_N = 25

CONE_TO_LANE_MISS = 30
CONE_GRACE_TICKS = 20          # 2초 (20Hz)

# ======================== YOLO 이벤트 cls_id ========================

GREEN_CLS_ID = 5
RED_CLS_ID = 9
YELLOW_LIGHT_CLS_ID = 11
LEFT_SIGN_CLS_ID = 7
CHILD_START_CLS_ID = 3
CHILD_END_CLS_ID = 1

# ======================== 사람 감지 (PEDESTRIAN) ========================

PED_X_MAX = 8.0             # 전방 이 거리 이내
PED_ROAD_HALF_WIDTH = 1.75  # center_path 기준 ± 이 폭
PED_MIN_STOP_TICKS = 60     # 최소 정지 시간 (3초, 20Hz)
PED_COOLDOWN_TICKS = 400    # 한번 감지 후 20초간 재감지 안 함 (20Hz)
PED_CAR_CLUSTER_COUNT = 3   # 도로 안 클러스터 이 이상이면 차 (사람 아님)
PED_CAR_SPREAD = 1.5        # 클러스터 간 거리 이 이내면 밀집 (차)

# ======================== 추월 (OVERTAKE) ========================

OVT_X_MAX = 10.0            # 전방 이 거리 이내에서 차 감지
OVT_ROAD_HALF_WIDTH = 1.75  # 도로 안 판정
OVT_MIN_R = 0.4             # 이 이상이면 차 (사람은 r<0.4)
OVT_MIN_CLUSTERS = 2        # 도로 안 큰 클러스터 이 이상이면 차 감지
OVT_LANE1_CHECK_X = 6.0     # 1차선 차 추월 완료 판단: 이 거리 안에 1차선 장애물 없으면 추월됨
OVT_LANE1_Y_MIN = 0.5       # 1차선 = center_y 기준 이만큼 왼쪽(y+) 이상
OVT_RETURN_DELAY = 100      # 추월 완료 후 5초(20Hz) 뒤 1차선 복귀
OVT_FRONT_CAR_DIST = 5.0    # 또는: 앞에 2차선 차가 이 거리 안에 오면 복귀

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

        self.phase = "IDLE"  # 시뮬 연결 전

        # CONE 데이터
        self._obstacles = []
        self._cone_prev_fit = None
        self._cone_miss = 0
        self._cone_grace = 0

        # PEDESTRIAN
        self._ped_timer = 0
        self._ped_cooldown = 0

        # OVERTAKE
        self._ovt_memory = LaneMemory()
        self._ovt_return_timer = 0
        self._ovt_lane1_gone_count = 0

        # LANE 데이터 (친구 _on_lane과 동일 구조)
        self._yellow_xs = np.array([], dtype=np.float64)
        self._yellow_ys = np.array([], dtype=np.float64)
        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._has_lane = False

        # YOLO 이벤트
        self._events = []

        # LANE 모드 중앙선 (사람 감지 기준선)
        self._lane_center_coef = None

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
        from std_msgs.msg import Bool
        self._pub_estop = self.create_publisher(Bool, "/emergency_stop", 10)

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

        if self.phase == "IDLE":
            if len(self._obstacles) > 0 or self._has_lane:
                self.get_logger().info("Sim connected → WAIT")
                self.phase = "WAIT"
            return
        elif self.phase == "WAIT":
            self._tick_wait(stamp)
        elif self.phase == "CONE":
            self._tick_cone(stamp)
        elif self.phase == "LANE":
            self._check_pedestrian()
            self._check_overtake()
            self._tick_lane(stamp)
        elif self.phase == "PEDESTRIAN":
            self._tick_pedestrian(stamp)
        elif self.phase == "OVERTAKE":
            self._tick_overtake(stamp)
        elif self.phase == "OVERTAKE_RETURN":
            self._tick_overtake_return(stamp)

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
                method = "track"
            else:
                left_mask = all_y > 0
                method = "y>0"

            lx, ly = all_x[left_mask], all_y[left_mask]
            # 디버그: 모든 콘 + 왼쪽 판별 결과
            self.get_logger().info(
                f"[CONE_DBG] method={method} all={list(zip([f'{x:.1f}' for x in all_x],[f'{y:.1f}' for y in all_y]))} "
                f"left={list(zip([f'{x:.1f}' for x in lx],[f'{y:.1f}' for y in ly]))}")
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
                    self._tick_lane(stamp)  # 즉시 차선 경로 발행 (경로 끊김 방지)
                    return
                if self._cone_grace > 0:
                    # grace 동안 피팅 실패 → 살짝 직진 (콘에 다가가기)
                    sample_xs = np.linspace(CONE_SAMPLE_X_START, CONE_SAMPLE_X_END, CONE_SAMPLE_N)
                    self._pub_center.publish(_poses_from_xy(stamp, sample_xs, np.zeros(CONE_SAMPLE_N)))
                    self._publish_target(stamp, TARGET_X, 0.0)
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

    # ---- PEDESTRIAN: 사람 감지 → 정지 → 도로 밖으로 나가면 출발 ----

    def _check_pedestrian(self):
        """LANE 주행 중 도로 안에 사람(작은 장애물) 있으면 정지. 밀집 클러스터면 차→무시."""
        if self._ped_cooldown > 0:
            self._ped_cooldown -= 1
            return
        if self._lane_center_coef is None:
            return

        # 도로 안 작은 클러스터 모으기 (r < 0.4 = 사람 크기)
        road_obs = []
        for ox, oy, _r in self._obstacles:
            if ox > PED_X_MAX or ox < 0.3:
                continue
            if _r < 0.1 or _r >= 0.4:
                continue  # 너무 작으면(점1개=노이즈/꼬깔) 무시, 너무 크면(차/나무) 무시
            center_y = float(np.polyval(self._lane_center_coef, ox))
            dist = abs(oy - center_y)
            if dist <= PED_ROAD_HALF_WIDTH:
                road_obs.append((ox, oy, _r))
                self.get_logger().info(
                    f"[PED_DBG] obs({ox:.1f},{oy:.1f}) r={_r:.2f} "
                    f"center_y={center_y:.2f} dist={dist:.2f}")

        if not road_obs:
            return

        # 밀집 판단: 도로 안 클러스터끼리 거리 PED_CAR_SPREAD 이내로 모여있으면 차
        # 서로 가까운 클러스터 개수 세기
        dense_count = 0
        for i, (x1, y1, _) in enumerate(road_obs):
            for x2, y2, _ in road_obs[i+1:]:
                if abs(x1 - x2) + abs(y1 - y2) < PED_CAR_SPREAD:
                    dense_count += 1

        if len(road_obs) >= PED_CAR_CLUSTER_COUNT and dense_count >= 2:
            return  # 밀집 클러스터 여러 개 = 차 → 무시 (친구 추월 담당)

        # 사람: 도로 안 클러스터 1~2개, 밀집 아님
        ox, oy, _r = road_obs[0]
        self.get_logger().info(
            f"PEDESTRIAN detected at ({ox:.1f},{oy:.1f}) "
            f"road_obs={len(road_obs)} dense={dense_count} → STOP")
        self._ped_timer = PED_MIN_STOP_TICKS
        self.phase = "PEDESTRIAN"
        from std_msgs.msg import Bool
        self._pub_estop.publish(Bool(data=True))

    def _tick_pedestrian(self, stamp):
        """정지. _tick_lane 호출해서 피팅 유지, estop으로 motion 정지."""
        from std_msgs.msg import Bool
        # 차선 피팅/발행 유지 (재출발 시 끊김 방지)
        self._tick_lane(stamp)
        # motion 정지
        self._pub_estop.publish(Bool(data=True))

        self._ped_timer -= 1

        if self._ped_timer > 0:
            return  # 최소 정지 시간 안 지남

        # 사람이 중앙선 왼쪽(y > center_y)으로 넘어갔나?
        if self._lane_center_coef is not None:
            for ox, oy, _r in self._obstacles:
                if ox > PED_X_MAX or ox < 0.3:
                    continue
                if _r < 0.1 or _r >= 0.4:
                    continue
                center_y = float(np.polyval(self._lane_center_coef, ox))
                if abs(oy - center_y) < PED_ROAD_HALF_WIDTH:
                    # 도로 안에 있음 — 중앙선 넘었나?
                    if oy > center_y:
                        self.get_logger().info(
                            f"Pedestrian crossed center ({ox:.1f},{oy:.1f}) cy={center_y:.2f} → LANE")
                        self._pub_estop.publish(Bool(data=False))
                        self._ped_cooldown = PED_COOLDOWN_TICKS
                        self.phase = "LANE"
                        return
                    else:
                        return  # 아직 오른쪽 → 계속 정지

        # 장애물 사라짐 → 출발
        self.get_logger().info("Pedestrian gone → LANE (cooldown 20s)")
        self._pub_estop.publish(Bool(data=False))
        self._ped_cooldown = PED_COOLDOWN_TICKS
        self.phase = "LANE"

    # ---- OVERTAKE: 차 감지 → 2차선 주행 → 추월 후 1차선 복귀 ----

    def _check_overtake(self):
        """LANE 주행 중 도로 위 큰 장애물(차) 감지 → OVERTAKE."""
        if self._ped_cooldown > 0:
            return  # 보행자 쿨다운 중엔 추월 감지 안 함
        if self._lane_center_coef is None:
            return

        big_on_road = 0
        for ox, oy, _r in self._obstacles:
            if ox > OVT_X_MAX or ox < 0.3:
                continue
            if _r < OVT_MIN_R:
                continue  # 작은 거 = 사람/콘
            center_y = float(np.polyval(self._lane_center_coef, ox))
            if abs(oy - center_y) <= OVT_ROAD_HALF_WIDTH:
                big_on_road += 1

        if big_on_road >= OVT_MIN_CLUSTERS:
            self.get_logger().info(
                f"OVERTAKE: {big_on_road} big obstacles on road → 2차선")
            self._ovt_memory = LaneMemory()
            self._ovt_lane1_gone_count = 0
            self._ovt_return_timer = 0
            self.phase = "OVERTAKE"

    def _tick_overtake(self, stamp):
        """2차선 주행 (오른쪽 흰선 기반). 1차선 차 사라지면 복귀 준비."""
        # 오른쪽 흰선 기반 경로 (plan_drive_target)
        # lane_xs/ys에서 전체 차선 점 사용
        all_xs = np.concatenate([self._yellow_xs, self._white_xs]) if self._has_lane else np.array([])
        all_ys = np.concatenate([self._yellow_ys, self._white_ys]) if self._has_lane else np.array([])

        result = plan_drive_target(all_xs, all_ys, self._ovt_memory)

        if result.ok:
            self._publish_target(stamp, result.target_x, result.target_y)
            if result.path_xs is not None and len(result.path_xs) >= 2:
                self._pub_center.publish(
                    _make_pose_array(stamp, result.path_xs, result.path_ys))

        # 1차선 차 추월 완료 판단: center_y 왼쪽(y+)에 큰 장애물 없으면
        lane1_car = False
        if self._lane_center_coef is not None:
            for ox, oy, _r in self._obstacles:
                if ox > OVT_LANE1_CHECK_X or ox < 0.3:
                    continue
                if _r < OVT_MIN_R:
                    continue
                center_y = float(np.polyval(self._lane_center_coef, ox))
                if oy > center_y + OVT_LANE1_Y_MIN:  # 1차선 = 중앙선 왼쪽
                    lane1_car = True
                    break

        if not lane1_car:
            self._ovt_lane1_gone_count += 1
        else:
            self._ovt_lane1_gone_count = 0

        # 1차선 차가 10틱(0.5초) 연속 안 보이면 → 복귀 타이머 시작
        if self._ovt_lane1_gone_count >= 10:
            self.get_logger().info("Lane1 car gone → OVERTAKE_RETURN")
            self._ovt_return_timer = OVT_RETURN_DELAY
            self.phase = "OVERTAKE_RETURN"

    def _tick_overtake_return(self, stamp):
        """1차선 복귀 대기. 타이머 후 또는 앞차 접근 시 LANE 복귀."""
        # 2차선 주행 유지하면서 타이머 카운트
        self._tick_overtake(stamp)
        # phase가 다시 OVERTAKE로 바뀌지 않게
        self.phase = "OVERTAKE_RETURN"

        self._ovt_return_timer -= 1

        # 앞에 2차선 차 접근 체크 (빠른 복귀)
        front_car = False
        for ox, oy, _r in self._obstacles:
            if ox > OVT_FRONT_CAR_DIST or ox < 0.3:
                continue
            if _r >= OVT_MIN_R:
                front_car = True
                break

        if self._ovt_return_timer <= 0 or front_car:
            reason = "timer" if self._ovt_return_timer <= 0 else "front car"
            self.get_logger().info(f"OVERTAKE done ({reason}) → LANE")
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

        # center_path의 coef 저장 (사람 감지 기준선으로 사용)
        all_fits = result.get("all_fits", [])
        if all_fits:
            self._lane_center_coef = all_fits[0]  # yellow_fit = center

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
