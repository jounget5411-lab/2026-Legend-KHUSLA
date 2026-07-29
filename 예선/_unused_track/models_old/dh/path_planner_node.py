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
from std_msgs.msg import Bool

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
CONE_FIT_MAX_MISS = 20

CONE_SAMPLE_X_START = 0.5
CONE_SAMPLE_X_END = 6.0
CONE_SAMPLE_N = 25

CONE_TO_LANE_MISS = 30
CONE_GRACE_TICKS = 20          # 2초 (20Hz)

# ======================== YOLO 클래스 ID (best.pt 학습 시점 기준) ========================
# 0=BLACK_CAR  1=CHILD_END  2=CHILD_LANE  3=CHILD_START
# 4=CROSSROAD_OUT  5=GOAL  6=GREEN  7=GREEN_CAR  8=HUMAN
# 9=LANE  10=LEFT  11=MID  12=POLICE  13=RED  14=STOP  15=YELLOW

# /detect/events_raw 에서 보는 플래그류
GREEN_CLS_ID = 6
RED_CLS_ID = 13
YELLOW_LIGHT_CLS_ID = 15
LEFT_SIGN_CLS_ID = 10
CHILD_START_CLS_ID = 3
CHILD_END_CLS_ID = 1
POLICE_CLS_ID = 12

# /detect/road_pixels 에서 보는 위치 의미 있는 객체 (bbox bottom v = 거리 proxy)
STOP_CLS_ID = 14
CROSSROAD_CLS_ID = 4
GOAL_CLS_ID = 5
HUMAN_CLS_ID = 8

# ======================== SHORTCUT (지름길 좌회전) 상수 ========================
# 진입: 정지선 충분히 가까움 + POLICE 없음 → WAITING (정지, LEFT 대기)
# WAITING → LEFT 신호 검출 → TURNING_1 (좌측 호 강제, 2초)
# TURNING_1 → FOLLOW (지름길 차선 추종, _tick_lane 재사용)
# FOLLOW → CROSSROAD 가까움 OR 15s timeout → TURNING_2 (좌측 호, 2초) → LANE 복귀

STOP_V_THRESHOLD = 400.0          # 480p 기준, 정지선 bbox bottom v가 이 이상이면 가까움
CROSS_V_THRESHOLD = 380.0         # 480p 기준, CROSSROAD_OUT bbox bottom v 임계

SC_TURN_R = 2.5                   # 강제 좌회전 호 반경 (m, 작을수록 강한 회전)
SC_TURN_TICKS = 40                # 강제 좌회전 지속 (40 ticks = 2초 @ 20Hz)
SC_ARC_SAMPLE_X_START = 0.5       # 호 샘플 시작 x (m)
SC_ARC_SAMPLE_X_END = 4.0         # 호 샘플 끝 x (m)
SC_ARC_SAMPLE_N = 12              # 호 샘플 점 개수

SC_FOLLOW_HARD_TIMEOUT_TICKS = 300  # FOLLOW에서 CROSSROAD 못 잡으면 15초 후 강제
SC_BYPASS_COOLDOWN_TICKS = 200      # POLICE 봤을 때 같은 정지선 재트리거 차단 (10초)
SC_DONE_COOLDOWN_TICKS = 200        # SHORTCUT 한 번 끝난 후 재진입 차단 (10초)

LEFT_SIGNAL_DEBOUNCE_TICKS = 2      # LEFT 신호 최소 검출 회수 (디바운싱)

# ======================== 사람 감지 (PEDESTRIAN) ========================

PED_X_MAX = 8.0             # 전방 이 거리 이내
PED_ROAD_HALF_WIDTH = 1.75  # center_path 기준 ± 이 폭
PED_MIN_STOP_TICKS = 60     # 최소 정지 시간 (3초, 20Hz)
PED_COOLDOWN_TICKS = 400    # 한번 감지 후 20초간 재감지 안 함 (20Hz)
PED_CAR_CLUSTER_COUNT = 3   # 도로 안 클러스터 이 이상이면 차 (사람 아님)
PED_CAR_SPREAD = 1.5        # 클러스터 간 거리 이 이내면 밀집 (차)

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

        # SHORTCUT (지름길 좌회전)
        self._stop_max_v = None          # /detect/road_pixels의 STOP bbox bottom v 최댓값
        self._cross_max_v = None         # /detect/road_pixels의 CROSSROAD_OUT bbox bottom v 최댓값
        self._sc_sub = None              # None | "WAITING" | "TURNING_1" | "FOLLOW" | "TURNING_2"
        self._sc_tick = 0                # 현재 sub-state에서 머무른 틱 수
        self._sc_cooldown = 0            # 재트리거 차단 카운터
        self._left_signal_hits = 0       # LEFT 디바운싱용 누적 카운트

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
        # SHORTCUT용 YOLO 객체 픽셀 (STOP, CROSSROAD_OUT 거리 트리거)
        self.create_subscription(PoseArray, "/detect/road_pixels", self._on_road, 10)

        # 발행
        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)
        self._pub_fits = self.create_publisher(PoseArray, "/lane_fits", 10)
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

    def _on_road(self, msg: PoseArray):
        """YOLO 객체 픽셀 — STOP/CROSSROAD_OUT bbox bottom v 추출 (거리 proxy)."""
        stop_v = None
        cross_v = None
        for p in msg.poses:
            cls = int(p.position.z)
            v_bot = float(p.position.y)
            if cls == STOP_CLS_ID:
                if stop_v is None or v_bot > stop_v:
                    stop_v = v_bot
            elif cls == CROSSROAD_CLS_ID:
                if cross_v is None or v_bot > cross_v:
                    cross_v = v_bot
        self._stop_max_v = stop_v
        self._cross_max_v = cross_v

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
            self._check_shortcut()
            self._tick_lane(stamp)
        elif self.phase == "PEDESTRIAN":
            self._tick_pedestrian(stamp)
        elif self.phase == "SHORTCUT":
            self._tick_shortcut(stamp)

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

    # ====================================================================
    # SHORTCUT: 정지선 좌회전 지름길 — PEDESTRIAN 패턴 그대로 따라감
    # 진입: LANE 안 _check_shortcut()가 STOP_v ≥ 임계 + POLICE 없음 확인
    # sub-state: WAITING → TURNING_1 → FOLLOW → TURNING_2 → LANE 복귀
    # ====================================================================

    def _check_shortcut(self):
        """LANE 주행 중 정지선이 충분히 가까우면 SHORTCUT 진입 판단."""
        if self._sc_cooldown > 0:
            self._sc_cooldown -= 1
            return
        if self._stop_max_v is None:
            return
        if self._stop_max_v < STOP_V_THRESHOLD:
            return

        # POLICE 보이면 지름길 포기 — 같은 정지선 다시 안 트리거하게 쿨다운
        if POLICE_CLS_ID in self._events:
            self._sc_cooldown = SC_BYPASS_COOLDOWN_TICKS
            self.get_logger().info(
                f"POLICE detected at stopline (v={self._stop_max_v:.0f}) "
                f"→ bypass shortcut ({SC_BYPASS_COOLDOWN_TICKS} tick cooldown)")
            return

        # 진입
        self.phase = "SHORTCUT"
        self._sc_sub = "WAITING"
        self._sc_tick = 0
        self._left_signal_hits = 0
        self._pub_estop.publish(Bool(data=True))
        self.get_logger().info(
            f"STOP near (v={self._stop_max_v:.0f}) + no POLICE → SHORTCUT.WAITING")

    def _tick_shortcut(self, stamp):
        """SHORTCUT sub-state dispatch."""
        if self._sc_sub == "WAITING":
            self._tick_sc_waiting(stamp)
        elif self._sc_sub == "TURNING_1":
            self._tick_sc_turning(stamp, after="FOLLOW")
        elif self._sc_sub == "FOLLOW":
            self._tick_sc_follow(stamp)
        elif self._sc_sub == "TURNING_2":
            self._tick_sc_turning(stamp, after="LANE")
        else:
            # 알 수 없는 sub — 안전망
            self.get_logger().warn(f"Unknown sc_sub={self._sc_sub} → LANE 복귀")
            self._exit_shortcut_to_lane()

    def _tick_sc_waiting(self, stamp):
        """정지하고 LEFT 신호 대기. PEDESTRIAN 정지 패턴과 동일."""
        self._pub_estop.publish(Bool(data=True))
        # 차선 피팅/발행 유지 — FOLLOW 진입 시 끊김 방지 (PEDESTRIAN과 동일)
        self._tick_lane(stamp)

        # LEFT 신호 디바운싱
        if LEFT_SIGN_CLS_ID in self._events:
            self._left_signal_hits += 1
        else:
            self._left_signal_hits = max(0, self._left_signal_hits - 1)

        if self._left_signal_hits >= LEFT_SIGNAL_DEBOUNCE_TICKS:
            self._pub_estop.publish(Bool(data=False))
            self._sc_sub = "TURNING_1"
            self._sc_tick = 0
            self.get_logger().info(
                f"LEFT signal confirmed ({self._left_signal_hits} hits) → TURNING_1")

    def _tick_sc_turning(self, stamp, after):
        """좌측 호 강제 발행 (TURNING_1, TURNING_2 공용). after = 다음 sub-state."""
        self._publish_left_arc(stamp)
        self._sc_tick += 1

        if self._sc_tick >= SC_TURN_TICKS:
            if after == "FOLLOW":
                self._sc_sub = "FOLLOW"
                self._sc_tick = 0
                self.get_logger().info("TURNING_1 done → FOLLOW (지름길 차선 추종)")
            elif after == "LANE":
                self.get_logger().info("TURNING_2 done → LANE 복귀")
                self._exit_shortcut_to_lane()

    def _tick_sc_follow(self, stamp):
        """지름길 위 차선 추종. CROSSROAD 가까워지거나 timeout이면 TURNING_2."""
        # 평소 LANE 추종 그대로 재사용
        self._tick_lane(stamp)
        self._sc_tick += 1

        # 종료 조건 1: CROSSROAD_OUT 가까이 봤음
        if self._cross_max_v is not None and self._cross_max_v >= CROSS_V_THRESHOLD:
            self._sc_sub = "TURNING_2"
            self._sc_tick = 0
            self.get_logger().info(
                f"CROSSROAD near (v={self._cross_max_v:.0f}) → TURNING_2")
            return

        # 종료 조건 2: hard timeout
        if self._sc_tick >= SC_FOLLOW_HARD_TIMEOUT_TICKS:
            self._sc_sub = "TURNING_2"
            self._sc_tick = 0
            self.get_logger().warn(
                f"FOLLOW timeout ({SC_FOLLOW_HARD_TIMEOUT_TICKS} ticks) → TURNING_2 (강제)")
            return

    def _exit_shortcut_to_lane(self):
        """SHORTCUT 종료 → LANE 복귀, 재진입 쿨다운 설정."""
        self.phase = "LANE"
        self._sc_sub = None
        self._sc_tick = 0
        self._sc_cooldown = SC_DONE_COOLDOWN_TICKS
        self._left_signal_hits = 0
        # 안전: emergency_stop 해제 (혹시 켜져있을 경우)
        self._pub_estop.publish(Bool(data=False))

    def _publish_left_arc(self, stamp):
        """원점에서 좌측으로 휘는 호를 /center_path와 /target으로 발행.

        motion_node가 이 path를 따라가면서 _compute_steering이 큰 음수 angle을
        자동 출력 (arctan2(-y, x), y>0이면 음수). _speed_from_angle이 |angle|↑
        에 따라 SPEED_MIN까지 자동 감속하므로 별도 속도 명령 불필요.
        """
        xs = np.linspace(SC_ARC_SAMPLE_X_START, SC_ARC_SAMPLE_X_END, SC_ARC_SAMPLE_N)
        # 원점에서 R 떨어진 좌측 원의 호: x = R sin θ, y = R(1 - cos θ)
        # θ = arcsin(x / R)
        ratio = np.clip(xs / SC_TURN_R, -1.0, 1.0)
        ys = SC_TURN_R * (1.0 - np.cos(np.arcsin(ratio)))   # y > 0 = 좌측

        self._pub_center.publish(_poses_from_xy(stamp, xs, ys))
        # 시각화용 좌/우 (참고용으로만)
        self._pub_left.publish(_poses_from_xy(stamp, xs, ys + 1.0))
        self._pub_right.publish(_poses_from_xy(stamp, xs, ys - 1.0))
        # /target은 호의 끝 점
        self._publish_target(stamp, float(xs[-1]), float(ys[-1]))

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
