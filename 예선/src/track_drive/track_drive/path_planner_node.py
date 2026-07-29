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

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String

# LANE 모드: 친구 plan() + 헬퍼 (lane_planner.py = 친구 path_planner_node.py 원본)
from .lane_planner import plan as lane_plan
from .lane_planner import (
    YELLOW_CLS_IDS, YELLOW_DASH_CLS_IDS, WHITE_CLS_IDS, PLAN_HZ, TARGET_X, TARGET_Y_LIMIT,
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
CONE_FIT_MAX_MISS = 39

CONE_SAMPLE_X_START = 0.5
CONE_SAMPLE_X_END = 6.0
CONE_SAMPLE_N = 25

CONE_TO_LANE_MISS = 40
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

EVT_NAMES = {
    6: "GREEN", 13: "RED", 15: "YELLOW", 10: "LEFT",
    12: "POLICE", 3: "CHILD_START", 1: "CHILD_END",
}

# ======================== SHORTCUT (지름길 좌회전) 상수 ========================
# 진입: 정지선 충분히 가까움 + POLICE 없음 → WAITING (정지, LEFT 대기)
# WAITING → LEFT 신호 검출 → TURNING_1 (좌측 호 강제, 2초)
# TURNING_1 → FOLLOW (지름길 차선 추종, _tick_lane 재사용)
# FOLLOW → CROSSROAD 가까움 OR 15s timeout → TURNING_2 (좌측 호, 2초) → LANE 복귀

STOP_V_THRESHOLD = 312.0          # 480p 기준, 정지선 bbox bottom v가 이 이상이면 가까움
CROSS_V_THRESHOLD = 278.0         # 480p 기준, CROSSROAD_OUT bbox bottom v 임계 (작을수록 멀리서 트리거)
STOP_HEADING_TARGET = 0.0         # 정지선 있는 직선 구간 heading (도)
STOP_HEADING_TOL = 10.0           # ±허용 범위 (도) — 이 밖이면 STOP 무시
STOP_BLOCK_270_DEG = 270.0        # heading(0~360 환산)이 이 부근을 지나면
STOP_BLOCK_270_TOL = 5.0         # ±허용(도)
STOP_BLOCK_270_TICKS = 60         # 이후 3초(20Hz)간 STOP 감지 차단
S_ZONE_HEADING_LO = 10.0
S_ZONE_HEADING_HI = 170.0
PURE_DRIVE_MODE = False           # True면 LANE에서 특수동작 전부 무시(순수주행 테스트)
IGNORE_PEDESTRIAN = False         # 보행자 감지 정상 동작 (사람 나오면 정지)
S_ZONE_BLOCK_TICKS = 160         # CONE→LANE 직후 8초(20Hz) S존 오인 차단
CORNER_BOOST_WINDOW_TICKS = 400   # CHILD 종료 후 이 시간(20s) 동안 코너 부스트

SC_TURN_R = 2.5                   # 강제 좌회전 호 반경 (m)
SC_TURN_DEG = 120.0               # 호 각도 (90보다 크게 → 실제 90도 정도 꺾임)
SC_TURN_TICKS = 50                # 강제 좌회전 지속 (50 ticks = 2.5초 @ 20Hz)
SC_ARC_SAMPLE_N = 15              # 호 샘플 점 개수

SC_FOLLOW_HARD_TIMEOUT_TICKS = 300  # FOLLOW에서 CROSSROAD 못 잡으면 15초 후 강제
SC_BYPASS_COOLDOWN_TICKS = 80       # POLICE 봤을 때 같은 정지선 재트리거 차단 (4초)
SC_DONE_COOLDOWN_TICKS = 200        # SHORTCUT 한 번 끝난 후 재진입 차단 (10초)

LEFT_SIGNAL_DEBOUNCE_TICKS = 2      # LEFT 신호 최소 검출 회수 (디바운싱)

# ======================== 사람 감지 (PEDESTRIAN) ========================

PED_X_MAX = 8.0             # 전방 이 거리 이내
PED_ROAD_HALF_WIDTH = 1.2   # center_path 기준 ± 이 폭 (1.75에서 축소, 도로 경계 오인식 방지)
PED_MIN_STOP_TICKS = 44     # 정지 시간 (2.2초, 20Hz)
PED_COOLDOWN_TICKS = 600    # 한번 감지 후 30초간 재감지 안 함 (20Hz)
PED_CAR_CLUSTER_COUNT = 3   # 도로 안 클러스터 이 이상이면 차 (사람 아님)
PED_CAR_SPREAD = 1.5        # 클러스터 간 거리 이 이내면 밀집 (차)

# ======================== 어린이 보호구역 (CHILD ZONE) ========================

CHILD_START_V_THR = 200.0        # CHILD_START bbox v 이 이상이면 트리거
CHILD_END_V_THR = 400.0          # CHILD_END bbox v 이 이상이면 트리거
CHILD_EXIT_DELAY_TICKS = 60      # END 감지 후 3초 뒤 속도 원복 (20Hz)
CHILD_ENTER_COOLDOWN_TICKS = 60  # START 후 3초간 END 무시 (오인식 방지)
CHILD_EXIT_COOLDOWN_TICKS = 200  # END 후 10초간 START 무시 (재진입 방지)
CHILD_HARD_TIMEOUT_TICKS = 340   # 17초 후 강제 해제 (20Hz)

# ======================== 추월 (OVERTAKE) ========================

OT_CAR_V_THRESHOLD = 250.0       # BLACK_CAR/GREEN_CAR bbox v 이 이상이면 트리거
OT_HEADING_MIN = 50.0            # 추월 허용 heading 최소 (도)
OT_HEADING_MAX = 170.0           # 추월 허용 heading 최대 (도)
OT_YELLOW_RIGHT_OFFSET = -1.5    # 노란선 기준 오른쪽 1.5m (2차선으로 이동)
OT_WHITE_LEFT_OFFSET = 0.8       # 오른쪽 흰선 기준 왼쪽 0.8m (2차선 주행)
OT_WHITE_LEFT_PASSING = 5.5      # 왼쪽 감지 후 흰선 기준 왼쪽 5.5m
OT_YELLOW_LEFT_OFFSET_1 = 2.0    # 왼쪽 사라진 후 노란선 기준 왼쪽 2.0m
OT_YELLOW_RIGHT_MERGE = -0.5     # 노란선 기준 오른쪽 0.5m (합류)
OT_MERGE_TICKS = 40              # 합류 주행 시간 (2초)
OT_LANE_TO_2ND_TICKS = 100       # 2차선 이동 시간 (5초)
OT_CAR_PASSED_DELAY_TICKS = 40   # 차 지나간 후 대기 (2초)
OT_RETURN_TICKS = 60             # 1차선 복귀 시간 (3초 → MERGE)
OT_LEFT_LIDAR_ANGLE = math.pi / 2   # 왼쪽 90도 (9시)
OT_LEFT_LIDAR_HALF = 0.0175        # ±각도 범위 (rad, ±1도)
OT_LEFT_LIDAR_R_MIN = 0.5         # 왼쪽 감지 최소 거리 (m)
OT_LEFT_LIDAR_R_MAX = 3.2         # 왼쪽 감지 최대 거리 (m)
OT_RIGHT_LIDAR_ANGLE = math.radians(-88.0)  # 오른쪽 88도
OT_RIGHT_LIDAR_HALF = math.radians(2.0)    # ±2도 (-90~-86도 커버)
OT_RIGHT_LIDAR_R_MIN = 0.01       # 오른쪽 감지 최소 거리 (m)
OT_RIGHT_LIDAR_R_MAX = 3.0        # 오른쪽 감지 최대 거리 (m)
OT_FIT_MIN_POINTS = 5
OT_OUTLIER_THR = 0.4
OT_EMA_ALPHA = 0.15
OT_X_BIN_SIZE = 0.5
OT_COOLDOWN_TICKS = 400          # 추월 후 20초 재트리거 방지
# ── 추월 재설계 (S자 끝 트리거 + 노란선 기준 offset 주행) 튜닝 상수 ──
OT_TRIGGER_HEADING = 140.0       # S자 후반에서 미리 추월 진입(150→140, 더 당김)
OT_LANE2_OFFSET = -1.07           # 2차선 = 노란선 기준 오른쪽 1.0m (Y 좌+ 이므로 음수)
OT_LANE1_OFFSET = 1.1            # 1차선 목표 = 노란선 기준 왼쪽 1.0m
OT_LANE1_REACH = 0.3             # 측정 좌측 오프셋이 이값 도달하면 안쪽 정착
OT_LANE1_SETTLE = 0.0            # 정착 = 목표와 동일(좌 0.9m)
OT_OFFSET_PROBE_X = 1.0          # 현재 오프셋 측정 지점 (m, lidar_frame x)
OT_PASS_DELAY_TICKS = 0          # 왼쪽 차 사라짐 확정 직후 바로 1차선 변경
OT_FRONT_GAP_MIN = 0.0           # 1차선 진입 전 정면 이격(m). 0=비활성(시간만으로 컷인)
OT_FRONT_HALF = math.radians(6.0)  # 정면 라이다 콘 반각
OT_FRONT_R_MIN = 0.3
OT_FRONT_R_MAX = 8.0
OT_LANE2_MAX_TICKS = 200         # 왼쪽 차 미감지 fallback (10초)
OT_LANE1_MAX_TICKS = 400         # child zone 미감지 fallback (20초)
OT_LANE2_SPEED = 10.0            # 2차선 단계 속도 a (너무 느려 타이밍 안나옴; 8→10)
OT_LANE1_SPEED = 16.0            # 1차선 단계 속도 (14→16, 더 상향)
OT_LANE2_MIN_TICKS = 20          # 2차선 진입 후 최소 안착(1초) — 그 전엔 옆차 감지 무시
OT_LEFT_ON_TICKS = 20            # 왼쪽 차 '감지' 디바운스(연속 N틱) — 1.0s(@20Hz)
OT_LEFT_OFF_TICKS = 6            # 왼쪽 차 '사라짐' 디바운스(연속 N틱)

BLACK_CAR_CLS_ID = 0
GREEN_CAR_CLS_ID = 7

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
        self._wait_green_ticks = 0       # WAIT에서 GREEN 감지 후 카운트
        self._start_grace_ticks = 0      # 출발 후 STOP 무시 타이머 (20초)

        # CONE 데이터
        self._obstacles = []
        self._cone_prev_fit = None
        self._cone_miss = 0
        self._cone_grace = 0

        # PEDESTRIAN
        self._ped_timer = 0
        self._ped_cooldown = 0
        # 보행자가 차선 가릴 때 center_path 흔들림 방지: 직전 경로를 약 1.5초 freeze
        self._path_freeze = 0
        self._frozen_center = None

        # SHORTCUT (지름길 좌회전)
        self._stop_max_v = None          # /detect/road_pixels의 STOP bbox bottom v 최댓값
        self._cross_max_v = None         # /detect/road_pixels의 CROSSROAD_OUT bbox bottom v 최댓값
        self._sc_sub = None              # None | "WAITING" | "TURNING_1" | "FOLLOW" | "TURNING_2"
        self._sc_tick = 0                # 현재 sub-state에서 머무른 틱 수
        self._sc_cooldown = 0            # 재트리거 차단 카운터
        self._left_signal_hits = 0       # LEFT 디바운싱용 누적 카운트
        self._red_stopping = False       # RED + 정지선 정지 중 플래그 (로그 중복 방지)
        self._stop_block_270 = 0         # 270도 부근 통과 후 STOP 차단 타이머
        self._police_seen_ago = 999      # POLICE 마지막 감지 후 틱 수

        # 어린이 보호구역
        self._child_zone = False
        self._child_start_max_v = None
        self._child_end_max_v = None
        self._child_enter_cooldown = 0   # START 후 END 무시 카운터
        self._child_exit_cooldown = 0    # END 후 START 무시 카운터
        self._child_exit_timer = 0       # END 감지 후 지연 원복 타이머
        self._child_total_ticks = 0      # child zone 진입 후 총 틱

        # OVERTAKE (추월)
        self._ot_sub = None              # None | LANE_TO_2ND | PASSING | CAR_BESIDE | CAR_PASSED | RETURN_TO_1ST
        self._ot_tick = 0
        self._ot_cooldown = 0
        self._ot_right_fit = None        # 오른쪽 흰 실선 피팅
        self._ot_yellow_fit = None       # 노란 중앙선 피팅
        self._car_max_v = None           # YOLO 차량 bbox v
        self._left_side_detected = False # 왼쪽 라이다 감지
        self._right_side_detected = False # 오른쪽 라이다 감지
        self._lane_offset = 0.0          # LANE/OVERTAKE 횡 오프셋 (좌+,우-)
        self._s_zone_seen = False        # S자 구간 통과 latch (추월 트리거용)
        self._front_min_range = float("inf")  # 정면 라이다 최소거리
        self._ot_left_on = 0             # 왼쪽 감지 연속 카운트
        self._ot_left_off = 0            # 왼쪽 소실 연속 카운트
        self._ot_lane1_reached = False   # 1차선 목표 오프셋 도달 latch(정착 전환용)

        # LANE 데이터 (친구 _on_lane과 동일 구조)
        self._yellow_xs = np.array([], dtype=np.float64)
        self._yellow_ys = np.array([], dtype=np.float64)
        self._yellow_dash_xs = np.array([], dtype=np.float64)
        self._yellow_dash_ys = np.array([], dtype=np.float64)
        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._has_lane = False

        # YOLO 이벤트
        self._events = []

        # LANE 모드 중앙선 (사람 감지 기준선)
        self._lane_center_coef = None
        self._last_lane_result = None

        # IMU heading
        self._heading_deg = 0.0
        self._s_zone_block = 0           # CONE→LANE 직후 S존 오인 차단 타이머
        self._s_zone_active = False      # 현재 S존 여부 (ped 오판 방지 게이트)
        self._manual_offset = 0.0        # ot_tune 수동 횡오프셋 (튜닝용)
        self._manual_hold = 0            # /manual_offset 수신 유지 카운트(끊기면 자동 해제)
        self._cb_window = 0              # CHILD 종료 후 코너 부스트 윈도우
        self._cb_prev_child = False      # child_zone 이전 상태(종료 전환 감지)
        self._corner_boost = False       # 어린이보호 종료 후 20s 코너 부스트

        # 구독
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/detect/events_raw", self._on_events, 10)
        # SHORTCUT용 YOLO 객체 픽셀 (STOP, CROSSROAD_OUT 거리 트리거)
        self.create_subscription(PoseArray, "/detect/road_pixels", self._on_road, 10)
        self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Float32, "/manual_offset", self._on_manual_offset, 10)
        self.create_subscription(String, "/detect/traffic_light", self._on_traffic_light, 10)
        # 수동 좌회전 트리거 (모델 LEFT 미인식 임시 대체)
        self.create_subscription(Bool, "/sc_go", self._on_sc_go, 10)
        self._sc_go_flag = False
        self._light_green = False        # legend_best.pt 신호등 green 상태

        # 발행
        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)
        self._pub_fits = self.create_publisher(PoseArray, "/lane_fits", 10)
        self._pub_estop = self.create_publisher(Bool, "/emergency_stop", 10)
        self._pub_lturn1 = self.create_publisher(Bool, "/left_turn1", 10)
        self._pub_lturn2 = self.create_publisher(Bool, "/left_turn2", 10)
        self._pub_child = self.create_publisher(Bool, "/child_zone", 10)
        self._pub_szone = self.create_publisher(Bool, "/s_zone", 10)
        self._pub_slow = self.create_publisher(Bool, "/slow_after_turn", 10)
        self._pub_slow_merge = self.create_publisher(Bool, "/slow_merge", 10)
        self._pub_ot_speed = self.create_publisher(Float32, "/overtake_speed", 10)
        self._pub_cone = self.create_publisher(Bool, "/cone_mode", 10)
        self._pub_sc_slow = self.create_publisher(Bool, "/shortcut_slow", 10)
        self._pub_corner_boost = self.create_publisher(Bool, "/corner_boost", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self._log_counter = 0
        self.get_logger().info("PLANNER")

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
            self._yellow_dash_xs = np.array([], dtype=np.float64)
            self._yellow_dash_ys = np.array([], dtype=np.float64)
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
        dm = np.isin(cls, list(YELLOW_DASH_CLS_IDS))
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        self._yellow_xs = xs[ym]
        self._yellow_ys = ys[ym]
        self._yellow_dash_xs = xs[dm]
        self._yellow_dash_ys = ys[dm]
        self._white_xs = xs[wm]
        self._white_ys = ys[wm]

    def _on_events(self, msg: PoseArray):
        self._events = [int(p.position.z) for p in msg.poses]

    def _on_road(self, msg: PoseArray):
        """YOLO 객체 픽셀 — bbox bottom v 추출 (거리 proxy)."""
        stop_v = None
        cross_v = None
        cs_v = None
        ce_v = None
        car_v = None
        for p in msg.poses:
            cls = int(p.position.z)
            v_bot = float(p.position.y)
            if cls == STOP_CLS_ID:
                if stop_v is None or v_bot > stop_v:
                    stop_v = v_bot
            elif cls == CROSSROAD_CLS_ID:
                if cross_v is None or v_bot > cross_v:
                    cross_v = v_bot
            elif cls == CHILD_START_CLS_ID:
                if cs_v is None or v_bot > cs_v:
                    cs_v = v_bot
            elif cls == CHILD_END_CLS_ID:
                if ce_v is None or v_bot > ce_v:
                    ce_v = v_bot
            elif cls in (BLACK_CAR_CLS_ID, GREEN_CAR_CLS_ID):
                if car_v is None or v_bot > car_v:
                    car_v = v_bot
        self._stop_max_v = stop_v
        self._cross_max_v = cross_v
        self._child_start_max_v = cs_v
        self._child_end_max_v = ce_v
        self._car_max_v = car_v

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._heading_deg = math.degrees(math.atan2(siny, cosy))

    def _on_manual_offset(self, msg):
        self._manual_offset = float(msg.data)
        self._manual_hold = 10   # 0.5초(20Hz) 유지 — 끊기면 자동 해제

    def _on_traffic_light(self, msg):
        self._light_green = (msg.data == "GREEN")

    def _on_scan(self, msg: LaserScan):
        """왼쪽/오른쪽 90도 방향 라이다 감지 (추월 시 옆 차량 확인)."""
        def _check_side(angle, half, r_min, r_max):
            lo = angle - half
            hi = angle + half
            idx_lo = max(0, int((lo - msg.angle_min) / msg.angle_increment))
            idx_hi = min(len(msg.ranges), int((hi - msg.angle_min) / msg.angle_increment))
            for i in range(idx_lo, idx_hi):
                r = msg.ranges[i]
                if np.isfinite(r) and r_min <= r <= r_max:
                    return True
            return False
        self._left_side_detected = _check_side(
            OT_LEFT_LIDAR_ANGLE, OT_LEFT_LIDAR_HALF,
            OT_LEFT_LIDAR_R_MIN, OT_LEFT_LIDAR_R_MAX)
        self._right_side_detected = _check_side(
            OT_RIGHT_LIDAR_ANGLE, OT_RIGHT_LIDAR_HALF,
            OT_RIGHT_LIDAR_R_MIN, OT_RIGHT_LIDAR_R_MAX)

        def _min_side(angle, half, r_min, r_max):
            lo = angle - half
            hi = angle + half
            idx_lo = max(0, int((lo - msg.angle_min) / msg.angle_increment))
            idx_hi = min(len(msg.ranges), int((hi - msg.angle_min) / msg.angle_increment))
            best = float("inf")
            for i in range(idx_lo, idx_hi):
                r = msg.ranges[i]
                if np.isfinite(r) and r_min <= r <= r_max and r < best:
                    best = r
            return best
        self._front_min_range = _min_side(
            0.0, OT_FRONT_HALF, OT_FRONT_R_MIN, OT_FRONT_R_MAX)

    def _on_sc_go(self, msg: Bool):
        if msg.data:
            self._sc_go_flag = True

    # ---- 메인 틱 ----

    def _tick(self):
        stamp = self.get_clock().now().to_msg()

        if self.phase == "IDLE":
            if len(self._obstacles) > 0 or self._has_lane:
                self.get_logger().info("WAIT")
                self.phase = "WAIT"
            return
        elif self.phase == "WAIT":
            self._tick_wait(stamp)
        elif self.phase == "CONE":
            self._tick_cone(stamp)
        elif self.phase == "LANE":
            if POLICE_CLS_ID in self._events:
                self._police_seen_ago = 0
            else:
                self._police_seen_ago += 1
            if self._start_grace_ticks > 0:
                self._start_grace_ticks -= 1
            if not PURE_DRIVE_MODE:
                self._tick_child_zone()
                if not IGNORE_PEDESTRIAN:
                    self._check_pedestrian()
                if self._start_grace_ticks <= 0:
                    if self._check_red_stop(stamp):
                        return
                    self._check_shortcut()
                    self._check_overtake()   # S자 끝(heading≈180) → 추월 진입
            if self.phase == "LANE":
                self._lane_offset = self._manual_offset if self._manual_hold > 0 else 0.0
            self._tick_lane(stamp)
        elif self.phase == "PEDESTRIAN":
            self._tick_pedestrian(stamp)
        elif self.phase == "SHORTCUT":
            self._tick_shortcut(stamp)
        elif self.phase == "OVERTAKE":
            self._tick_overtake(stamp)

        self._pub_cone.publish(Bool(data=(self.phase == "CONE")))
        self._pub_sc_slow.publish(Bool(data=(self.phase == "SHORTCUT")))

        # 로그
        if self._manual_hold > 0:
            self._manual_hold -= 1
        # heading 270도(0~360 환산) 부근 통과 → 이후 STOP_BLOCK_270_TICKS 동안 STOP 차단
        _h360 = self._heading_deg % 360.0
        _d270 = abs(_h360 - STOP_BLOCK_270_DEG)
        if _d270 > 180.0:
            _d270 = 360.0 - _d270
        if _d270 <= STOP_BLOCK_270_TOL:
            self._stop_block_270 = STOP_BLOCK_270_TICKS
        elif self._stop_block_270 > 0:
            self._stop_block_270 -= 1
        raw_szone = (S_ZONE_HEADING_LO <= self._heading_deg <= S_ZONE_HEADING_HI)
        if self._s_zone_block > 0:
            self._s_zone_block -= 1
        # CONE→LANE 직후 차단구간이거나 LANE 주행이 아니면 S존 무시(오인 방지)
        in_szone = raw_szone and self.phase == "LANE" and self._s_zone_block <= 0
        self._s_zone_active = in_szone   # ped 게이트용(다음 틱 참조)
        if in_szone:
            self._s_zone_seen = True   # S자 통과 latch (추월 트리거 조건)
        self._pub_szone.publish(Bool(data=in_szone))

        # 어린이보호 종료 후 20초간 코너 부스트 (motion에서 in_corner일 때만 적용, imu 조건 없음)
        if self._cb_prev_child and not self._child_zone:
            self._cb_window = CORNER_BOOST_WINDOW_TICKS
        self._cb_prev_child = self._child_zone
        if self._cb_window > 0:
            self._cb_window -= 1
        self._corner_boost = self._cb_window > 0
        self._pub_corner_boost.publish(Bool(data=self._corner_boost))
        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            phase_log = self.phase
            if self.phase == "SHORTCUT":
                phase_log = {
                    "WAITING": "SC WAIT",
                    "TURNING_1": "SC LT1",
                    "FOLLOW": "SC RUN",
                    "TURNING_2": "SC LT2",
                }.get(self._sc_sub, "SC")
            elif self.phase == "OVERTAKE":
                phase_log = {
                    "LANE_TO_2ND": "OT START",
                    "PASSING": "OT PASS",
                    "CAR_BESIDE": "OT SIDE",
                    "RETURN_TO_1ST": "OT RET",
                    "MERGE": "OT MERGE",
                }.get(self._ot_sub, "OT")
            elif self._child_zone:
                phase_log = "CHILD"
            if in_szone:
                phase_log += " +S"
            self.get_logger().info(phase_log)

    # ---- WAIT ----

    def _tick_wait(self, stamp):
        self._tick_lane(stamp)
        self._pub_estop.publish(Bool(data=True))

        if self._wait_green_ticks > 0:
            self._wait_green_ticks -= 1
            if self._wait_green_ticks <= 0:
                self.get_logger().info("CONE")
                self.phase = "CONE"
                self._cone_grace = CONE_GRACE_TICKS
                self._cone_miss = 0
                self._start_grace_ticks = 400  # 출발 후 20초간 STOP 무시
                self._pub_estop.publish(Bool(data=False))
            elif self._wait_green_ticks % PLAN_HZ == 0:
                self.get_logger().info("GREEN")
            return

        if self._light_green:
            self._wait_green_ticks = 6   # green 확인 후 약 0.3초 뒤 출발(legend 모델, 주황오인 없음 → 3.6초 대기 제거)

            self.get_logger().info("GREEN")

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
        self._cone_left_count = 0
        if all_x.size >= CONE_FIT_MIN_POINTS:
            if self._cone_prev_fit is not None:
                expected_y = np.polyval(self._cone_prev_fit, all_x)
                left_mask = np.abs(all_y - expected_y) < CONE_LEFT_ACCEPT_DIST
                method = "track"
            else:
                left_mask = all_y > 0
                method = "y>0"

            lx, ly = all_x[left_mask], all_y[left_mask]
            self._cone_left_count = lx.size
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
                    self.get_logger().info("LANE")
                    self.phase = "LANE"
                    self._s_zone_block = S_ZONE_BLOCK_TICKS   # CONE→LANE 직후 S존 오인 차단
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
        if self._cone_left_count <= 1:
            center_coef[1] -= 0.268  # 시계방향 15도 (tan15°≈0.268), 콘 1개 이하일 때만

        left_ys = np.polyval(left_fit, sample_xs)
        center_ys = np.clip(np.polyval(center_coef, sample_xs), -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        right_ys = center_ys - TRACK_HALF_WIDTH

        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, left_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, right_ys))
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        self._publish_target(stamp, TARGET_X, target_y)

    # ---- CHILD ZONE: 어린이 보호구역 속도 제한 ----

    def _tick_child_zone(self):
        """어린이 보호구역 진입/해제 판단. 매 틱 호출."""
        # 쿨다운 감소
        if self._child_enter_cooldown > 0:
            self._child_enter_cooldown -= 1
        if self._child_exit_cooldown > 0:
            self._child_exit_cooldown -= 1

        if not self._child_zone:
            # 진입 판단
            if self._child_exit_cooldown > 0:
                return
            if (self._child_start_max_v is not None
                    and self._child_start_max_v >= CHILD_START_V_THR):
                self._child_zone = True
                self._child_enter_cooldown = CHILD_ENTER_COOLDOWN_TICKS
                self._child_exit_timer = 0
                self._child_total_ticks = 0
                self._pub_child.publish(Bool(data=True))
                self.get_logger().info("CHILD ON")
        else:
            self._child_total_ticks += 1
            # hard timeout
            if self._child_total_ticks >= CHILD_HARD_TIMEOUT_TICKS:
                self._child_zone = False
                self._child_exit_cooldown = CHILD_EXIT_COOLDOWN_TICKS
                self._pub_child.publish(Bool(data=False))
                self.get_logger().info("CHILD OFF")
                return
            # 해제 판단
            if self._child_exit_timer > 0:
                self._child_exit_timer -= 1
                if self._child_exit_timer <= 0:
                    self._child_zone = False
                    self._child_exit_cooldown = CHILD_EXIT_COOLDOWN_TICKS
                    self._pub_child.publish(Bool(data=False))
                    self.get_logger().info("CHILD OFF")
                return
            if self._child_enter_cooldown > 0:
                return
            if (self._child_end_max_v is not None
                    and self._child_end_max_v >= CHILD_END_V_THR):
                self._child_exit_timer = CHILD_EXIT_DELAY_TICKS
                self.get_logger().info("CHILD END")

    # ---- PEDESTRIAN: 사람 감지 → 정지 → 도로 밖으로 나가면 출발 ----

    def _check_pedestrian(self):
        """LANE 주행 중 도로 안에 사람(작은 장애물) 있으면 정지. 밀집 클러스터면 차→무시."""
        if self._ped_cooldown > 0:
            self._ped_cooldown -= 1
            return
        if not self._s_zone_active:
            return  # 보행자는 S자 구간에만 등장 → 그 밖에선 ped 판정 안 함
        if self._lane_center_coef is None:
            return

        # 도로 안 작은 클러스터 모으기 (r < 0.4 = 사람 크기)
        road_obs = []
        for ox, oy, _r in self._obstacles:
            if ox > PED_X_MAX or ox < 1.5:
                continue
            if _r < 0.1 or _r >= 0.4:
                continue  # 너무 작으면(점1개=노이즈/꼬깔) 무시, 너무 크면(차/나무) 무시
            center_y = float(np.polyval(self._lane_center_coef, ox))
            dist = abs(oy - center_y)
            if dist <= PED_ROAD_HALF_WIDTH:
                road_obs.append((ox, oy, _r))

        if not road_obs:
            return

        # 도로 안 사람 후보 존재 → center_path를 직전값으로 약 1.5초 freeze
        # (보행자가 차선을 잠깐 가려 경로가 휙휙 바뀌는 것 방지)
        self._path_freeze = 30

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
        self.get_logger().info("PED")
        self._ped_timer = PED_MIN_STOP_TICKS
        self.phase = "PEDESTRIAN"
        from std_msgs.msg import Bool
        self._pub_estop.publish(Bool(data=True))

    def _tick_pedestrian(self, stamp):
        """정지. _tick_lane 호출해서 피팅 유지, estop으로 motion 정지.
        고정 3초(PED_MIN_STOP_TICKS) 정지 후 무조건 출발 (움직임 기준 X)."""
        from std_msgs.msg import Bool
        # 차선 피팅/발행 유지 (재출발 시 끊김 방지)
        self._tick_lane(stamp)
        # motion 정지
        self._pub_estop.publish(Bool(data=True))

        self._ped_timer -= 1

        if self._ped_timer > 0:
            return  # 최소 정지 시간 안 지남

        # === 고정 3초 경과 → 무조건 출발 ===
        self.get_logger().info("LANE")
        self._pub_estop.publish(Bool(data=False))
        self._ped_cooldown = PED_COOLDOWN_TICKS
        self.phase = "LANE"

        # --- 기존 사람-움직임 기준 재출발 로직 (주석처리: 고정 3초로 대체) ---
        # # 사람이 중앙선 왼쪽(y > center_y)으로 넘어갔나?
        # if self._lane_center_coef is not None:
        #     for ox, oy, _r in self._obstacles:
        #         if ox > PED_X_MAX or ox < 0.3:
        #             continue
        #         if _r < 0.1 or _r >= 0.4:
        #             continue
        #         center_y = float(np.polyval(self._lane_center_coef, ox))
        #         if abs(oy - center_y) < PED_ROAD_HALF_WIDTH:
        #             # 도로 안에 있음 — 중앙선 넘었나?
        #             if oy > center_y:
        #                 self.get_logger().info("LANE")
        #                 self._pub_estop.publish(Bool(data=False))
        #                 self._ped_cooldown = PED_COOLDOWN_TICKS
        #                 self.phase = "LANE"
        #                 return
        #             else:
        #                 return  # 아직 오른쪽 → 계속 정지
        # # 장애물 사라짐 → 출발
        # self.get_logger().info("LANE")
        # self._pub_estop.publish(Bool(data=False))
        # self._ped_cooldown = PED_COOLDOWN_TICKS
        # self.phase = "LANE"

    # ====================================================================
    # SHORTCUT: 정지선 좌회전 지름길 — PEDESTRIAN 패턴 그대로 따라감
    # 진입: LANE 안 _check_shortcut()가 STOP_v ≥ 임계 + POLICE 없음 확인
    # sub-state: WAITING → TURNING_1 → FOLLOW → TURNING_2 → LANE 복귀
    # ====================================================================

    def _heading_ok_for_stop(self):
        """heading이 정지선 방향 범위 안인지 확인."""
        diff = abs(self._heading_deg - STOP_HEADING_TARGET)
        if diff > 180:
            diff = 360 - diff
        return diff <= STOP_HEADING_TOL

    def _check_red_stop(self, stamp):
        """정지선 가까이 + GREEN 아님 → 정지. GREEN 뜰 때만 해제. True 반환 시 이번 틱 종료."""
        police_recent = self._police_seen_ago <= 10  # 0.5초 이내 POLICE
        # 이미 정지 중이면 GREEN + POLICE 최근으로 해제
        if self._red_stopping:
            if self._light_green and police_recent:
                self._pub_estop.publish(Bool(data=False))
                self.get_logger().info("GO")
                self._red_stopping = False
                return False
            if ((self._light_green or LEFT_SIGN_CLS_ID in self._events)
                    and not police_recent):
                # GREEN 또는 LEFT 화살표 + POLICE 아님 → SHORTCUT.WAITING 진입
                # (직진 GREEN만 기다리다 LEFT 화살표 타이밍 놓쳐 한 바퀴 도는 것 방지)
                self._red_stopping = False
                self.phase = "SHORTCUT"
                self._sc_sub = "WAITING"
                self._sc_tick = 0
                self._left_signal_hits = 0
                self._pub_estop.publish(Bool(data=True))
                self.get_logger().info("SC WAIT")
                return True
            self._pub_estop.publish(Bool(data=True))
            self._tick_lane(stamp)
            return True
        # 새로 진입: 정지선 + heading OK + GREEN 없음
        if (self._stop_max_v is None or self._stop_max_v < STOP_V_THRESHOLD
                or not self._heading_ok_for_stop()):
            return False
        if self._stop_block_270 > 0:
            return False  # 270도 부근 통과 후 3초간 STOP 차단
        if self._light_green and police_recent:
            return False  # POLICE 교차로 + 초록불 → 그냥 통과
        self.get_logger().info("STOP")
        self._red_stopping = True
        self._pub_estop.publish(Bool(data=True))
        self._tick_lane(stamp)
        return True

    def _check_shortcut(self):
        """LANE 주행 중 정지선이 충분히 가까우면 SHORTCUT 진입 판단."""
        if self._sc_cooldown > 0:
            self._sc_cooldown -= 1
            return
        if self._stop_max_v is None:
            return
        if self._stop_max_v < STOP_V_THRESHOLD:
            return
        if not self._heading_ok_for_stop():
            return

        # POLICE 보이면 지름길 포기 — 같은 정지선 다시 안 트리거하게 쿨다운
        if POLICE_CLS_ID in self._events:
            self._sc_cooldown = SC_BYPASS_COOLDOWN_TICKS
            self.get_logger().info("POLICE")
            return

        # 진입
        self.phase = "SHORTCUT"
        self._sc_sub = "WAITING"
        self._sc_tick = 0
        self._left_signal_hits = 0
        self._pub_estop.publish(Bool(data=True))
        self.get_logger().info("SC WAIT")

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
            self.get_logger().warn("SC ERR")
            self._exit_shortcut_to_lane()

    def _tick_sc_waiting(self, stamp):
        """정지하고 LEFT 신호 대기."""
        self._pub_estop.publish(Bool(data=True))
        self._tick_lane(stamp)
        self._sc_tick += 1

        if LEFT_SIGN_CLS_ID in self._events:
            self._left_signal_hits += 1
        else:
            self._left_signal_hits = max(0, self._left_signal_hits - 1)

        if self._sc_tick % PLAN_HZ == 0:
            self.get_logger().info("SC WAIT")

        if self._left_signal_hits >= LEFT_SIGNAL_DEBOUNCE_TICKS:
            self._pub_estop.publish(Bool(data=False))
            self._sc_sub = "TURNING_1"
            self._sc_tick = 0
            self.get_logger().info("LT1")

    def _tick_sc_turning(self, stamp, after):
        """좌회전: motion_node에 /left_turn1 or /left_turn2 신호."""
        if self._sc_tick == 0:
            if after == "FOLLOW":
                self._pub_lturn1.publish(Bool(data=True))
            else:
                self._pub_lturn2.publish(Bool(data=True))
        self._sc_tick += 1

        ticks = SC_TURN_TICKS
        if self._sc_tick % PLAN_HZ == 0:
            self.get_logger().info("SC TURN")

        if self._sc_tick >= ticks:
            if after == "FOLLOW":
                self._sc_sub = "FOLLOW"
                self._sc_tick = 0
                self.get_logger().info("SC FOLLOW")
            elif after == "LANE":
                self.get_logger().info("LANE")
                self._exit_shortcut_to_lane()

    def _tick_sc_follow(self, stamp):
        """지름길 위 차선 추종. CROSSROAD 가까워지거나 timeout이면 TURNING_2."""
        self._tick_lane(stamp)
        self._sc_tick += 1

        if self._sc_tick % PLAN_HZ == 0:
            self.get_logger().info("SC FOLLOW")

        if self._cross_max_v is not None and self._cross_max_v >= CROSS_V_THRESHOLD:
            self._sc_sub = "TURNING_2"
            self._sc_tick = 0
            self.get_logger().info("LT2")
            return

        # # 종료 조건 2: hard timeout (임시 비활성화)
        # if self._sc_tick >= SC_FOLLOW_HARD_TIMEOUT_TICKS:
        #     self._sc_sub = "TURNING_2"
        #     self._sc_tick = 0
        #     self.get_logger().warn(
        #         f"FOLLOW timeout ({SC_FOLLOW_HARD_TIMEOUT_TICKS} ticks) → TURNING_2 (강제)")
        #     return

    def _exit_shortcut_to_lane(self):
        """SHORTCUT 종료 → LANE 복귀, 재진입 쿨다운 설정."""
        self.get_logger().info("LANE")
        self.phase = "LANE"
        self._sc_sub = None
        self._sc_tick = 0
        self._sc_cooldown = SC_DONE_COOLDOWN_TICKS
        self._left_signal_hits = 0
        self._pub_estop.publish(Bool(data=False))
        # (slow_after_turn 제거: CHILD 속도보다 낮은 5로 캡돼 "예전 속도"가 나오던 원인.
        #  지름길 진입도 일반 child zone과 동일하게 CHILD_ZONE_SPEED 적용)
        # 좌회전 직후 무조건 어린이 보호구역 진입
        self._child_zone = True
        self._child_enter_cooldown = CHILD_ENTER_COOLDOWN_TICKS
        self._child_exit_timer = 0
        self._child_total_ticks = 0
        self._pub_child.publish(Bool(data=True))
        self.get_logger().info("CHILD ON")

    def _publish_left_arc(self, stamp):
        """원점에서 좌측으로 휘는 호를 /center_path와 /target으로 발행.
        theta 기반 파라미터화로 90도 이상 회전 가능.
        """
        thetas = np.linspace(0.05, np.radians(SC_TURN_DEG), SC_ARC_SAMPLE_N)
        xs = SC_TURN_R * np.sin(thetas)          # 전방
        ys = SC_TURN_R * (1.0 - np.cos(thetas))  # 좌측 (y > 0)

        self._pub_center.publish(_poses_from_xy(stamp, xs, ys))
        self._pub_left.publish(_poses_from_xy(stamp, xs, ys + 1.0))
        self._pub_right.publish(_poses_from_xy(stamp, xs, ys - 1.0))
        self._publish_target(stamp, float(xs[-1]), float(ys[-1]))

    # ---- OVERTAKE: 추월 ----

    def _check_overtake(self):
        """S자 종료(heading≈180) 감지 → OVERTAKE 진입.
        차량은 라이다로만 보이므로 트리거는 순수 기하: s_zone 통과 뒤 heading 도달."""
        if self._ot_cooldown > 0:
            self._ot_cooldown -= 1
            return
        if self._child_zone:
            return
        if not self._s_zone_seen:
            return
        if self._heading_deg < OT_TRIGGER_HEADING:
            return
        self.phase = "OVERTAKE"
        self._ot_sub = "IN_LANE2"
        self._ot_tick = 0
        self._lane_offset = OT_LANE2_OFFSET
        self._s_zone_seen = False
        self._ot_left_on = 0
        self._ot_left_off = 0
        self._ot_lane1_reached = False
        self.get_logger().info("OVERTAKE 진입 → 2차선(우 1.5m)")

    def _tick_overtake(self, stamp):
        """OVERTAKE: 노란 중앙선 plan() + self._lane_offset 로 주행."""
        self._tick_child_zone()              # 추월 중에도 child zone 감지 유지(종료 조건)
        # 속도 캡: 2차선 단계=a(OT_LANE2_SPEED), 1차선 단계=a보다 조금 빠르게(OT_LANE1_SPEED)
        if self._ot_sub == "CUT_TO_LANE1":
            self._pub_ot_speed.publish(Float32(data=OT_LANE1_SPEED))
        else:
            self._pub_ot_speed.publish(Float32(data=OT_LANE2_SPEED))
        if self._ot_sub == "IN_LANE2":
            self._tick_ot_in_lane2()
        elif self._ot_sub == "LANE1_BESIDE":
            self._tick_ot_lane1_beside()
        elif self._ot_sub == "LANE1_PASSED":
            self._tick_ot_lane1_passed()
        elif self._ot_sub == "CUT_TO_LANE1":
            self._tick_ot_cut_to_lane1()
        else:
            self._exit_overtake()
            return
        self._tick_lane(stamp)

    def _tick_ot_in_lane2(self):
        """2차선(우) 안착 → 왼쪽 라이다에 1차선 차가 '확실히' 잡힐 때까지."""
        self._ot_tick += 1
        self._lane_offset = OT_LANE2_OFFSET
        if self._ot_tick < OT_LANE2_MIN_TICKS:   # 먼저 2차선에 안착
            self._ot_left_on = 0
            return
        self._ot_left_on = self._ot_left_on + 1 if self._left_side_detected else 0
        if self._ot_left_on >= OT_LEFT_ON_TICKS:
            self._ot_sub = "LANE1_BESIDE"
            self._ot_tick = 0
            self._ot_left_off = 0
            self.get_logger().info("OT: 왼쪽 차 옆(확정)")
        elif self._ot_tick >= OT_LANE2_MAX_TICKS:
            # 왼쪽 차 끝내 미감지 → 거짓 추월판정 대신 추월 취소(LANE 복귀)
            self.get_logger().info("OT: 왼쪽 차 미감지 → 추월 취소")
            self._exit_overtake()

    def _tick_ot_lane1_beside(self):
        """1차선 차가 옆에 있는 동안 2차선 유지. '확실히' 사라지면(지나치면) 대기."""
        self._ot_tick += 1
        self._lane_offset = OT_LANE2_OFFSET
        self._ot_left_off = self._ot_left_off + 1 if not self._left_side_detected else 0
        if self._ot_left_off >= OT_LEFT_OFF_TICKS:
            self._ot_sub = "LANE1_PASSED"
            self._ot_tick = 0
            self.get_logger().info("OT: 왼쪽 차 지나침(확정)")

    def _tick_ot_lane1_passed(self):
        """지나친 뒤 2초 대기(내 차가 앞서도록) + 정면(2차선 차) 충분히 멀어야 1차선 진입."""
        self._ot_tick += 1
        self._lane_offset = OT_LANE2_OFFSET
        front_ok = (OT_FRONT_GAP_MIN <= 0.0) or (self._front_min_range >= OT_FRONT_GAP_MIN)
        if self._ot_tick >= OT_PASS_DELAY_TICKS and front_ok:
            self._ot_sub = "CUT_TO_LANE1"
            self._ot_tick = 0
            self.get_logger().info("OT: 1차선 진입")

    def _tick_ot_cut_to_lane1(self):
        """1차선(좌) 주행. 좌 1.0 목표로 가다 측정 좌측 오프셋이 REACH(0.9) 넘으면
        0.7로 당겨 정착(왼쪽 꼬깔 충돌 방지). child zone 진입하면 종료."""
        self._ot_tick += 1
        ego_off = 0.0
        if self._lane_center_coef is not None:
            ego_off = -float(np.polyval(self._lane_center_coef, OT_OFFSET_PROBE_X))
        if not self._ot_lane1_reached and ego_off >= OT_LANE1_REACH:
            self._ot_lane1_reached = True
            self.get_logger().info(f"OT: 1차선 도달 → 좌 {OT_LANE1_SETTLE} 정착")
        self._lane_offset = OT_LANE1_SETTLE if self._ot_lane1_reached else OT_LANE1_OFFSET
        # child_zone가 컷인 '도중'에 떠도 바로 abort하지 않고 1차선 도달(_ot_lane1_reached) 후에만
        # 종료 → 횡이동 방향 급반전(추월↔child 충돌/버벅) 방지. 미도달이면 MAX_TICKS로만 탈출.
        if (self._child_zone and self._ot_lane1_reached) or self._ot_tick >= OT_LANE1_MAX_TICKS:
            self.get_logger().info("OVERTAKE 완료 → LANE")
            self._exit_overtake()

    def _exit_overtake(self):
        self.phase = "LANE"
        self._ot_sub = None
        self._ot_tick = 0
        self._ot_cooldown = OT_COOLDOWN_TICKS
        self._lane_offset = 0.0
        self._pub_ot_speed.publish(Float32(data=0.0))

    def _ot_update_fits(self):
        """추월 중 흰선/노란선 피팅 업데이트."""
        # 오른쪽 흰 실선
        if self._white_xs.size >= 3:
            rx, ry = self._ot_extract_right(self._white_xs, self._white_ys)
            new_fit = self._ot_fit(rx, ry, self._ot_right_fit)
            if new_fit is not None:
                if self._ot_right_fit is not None:
                    self._ot_right_fit = ((1.0 - OT_EMA_ALPHA) * self._ot_right_fit
                                          + OT_EMA_ALPHA * new_fit)
                else:
                    self._ot_right_fit = new_fit
        # 노란선
        if self._yellow_xs.size >= OT_FIT_MIN_POINTS:
            new_fit = self._ot_fit(self._yellow_xs, self._yellow_ys, self._ot_yellow_fit)
            if new_fit is not None:
                if self._ot_yellow_fit is not None:
                    self._ot_yellow_fit = ((1.0 - OT_EMA_ALPHA) * self._ot_yellow_fit
                                           + OT_EMA_ALPHA * new_fit)
                else:
                    self._ot_yellow_fit = new_fit

    @staticmethod
    def _ot_extract_right(xs, ys):
        """x 구간별 최소 Y → 오른쪽 실선."""
        bins = np.arange(0.5, 8.0, OT_X_BIN_SIZE)
        rx, ry = [], []
        for bx in bins:
            mask = (xs >= bx) & (xs < bx + OT_X_BIN_SIZE)
            if not mask.any():
                continue
            idx = np.argmin(ys[mask])
            rx.append(float(xs[mask][idx]))
            ry.append(float(ys[mask][idx]))
        return np.array(rx), np.array(ry)

    @staticmethod
    def _ot_fit(xs, ys, prev_fit):
        """아웃라이어 제거 + 2차 피팅."""
        if prev_fit is not None and xs.size >= 3:
            expected = np.polyval(prev_fit, xs)
            inlier = np.abs(ys - expected) < OT_OUTLIER_THR
            if inlier.sum() >= OT_FIT_MIN_POINTS:
                xs, ys = xs[inlier], ys[inlier]
        if xs.size < OT_FIT_MIN_POINTS:
            return None
        x_span = float(xs.max() - xs.min())
        if x_span < 1.0:
            return None
        try:
            deg = 2 if xs.size >= 6 and x_span >= 2.0 else 1
            coef = np.polyfit(xs, ys, deg)
            if deg == 1:
                coef = np.array([0.0, coef[0], coef[1]])
            return coef
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _ot_publish_path(self, stamp, fit, offset):
        """피팅 + 오프셋으로 center_path 발행."""
        center_coef = fit.copy()
        center_coef[2] += offset
        sample_xs = np.linspace(0.5, 8.0, 20)
        center_ys = np.clip(np.polyval(center_coef, sample_xs),
                            -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))
        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        self._publish_target(stamp, TARGET_X, target_y)

    # ---- LANE (친구 plan() 그대로) ----

    def _tick_lane(self, stamp):
        if not self._has_lane:
            if self.phase == "LANE" and self._publish_last_lane(stamp):
                return
            return

        # OVERTAKE 중엔 plan() 자체 회피를 끄고(빈 장애물) 오프셋 주행만 — 이중 회피 방지
        lane_obs = [] if self.phase == "OVERTAKE" else self._obstacles
        result = lane_plan(self._yellow_xs, self._yellow_ys,
                           self._white_xs, self._white_ys,
                           lane_obs,
                           child_zone=self._child_zone,
                           yellow_dash_xs=self._yellow_dash_xs,
                           yellow_dash_ys=self._yellow_dash_ys)

        if result is None:
            if self.phase == "LANE" and self._publish_last_lane(stamp):
                return
            return

        self._last_lane_result = self._copy_lane_result(result)
        self._publish_lane_result(stamp, result)

    @staticmethod
    def _copy_lane_result(result):
        return {
            "target": tuple(result["target"]),
            "sample_xs": np.asarray(result["sample_xs"], dtype=np.float64).copy(),
            "sample_ys": np.asarray(result["sample_ys"], dtype=np.float64).copy(),
            "left_ys": np.asarray(result["left_ys"], dtype=np.float64).copy(),
            "right_ys": np.asarray(result["right_ys"], dtype=np.float64).copy(),
            "all_fits": [
                np.asarray(coef, dtype=np.float64).copy()
                for coef in result.get("all_fits", [])
            ],
        }

    def _publish_last_lane(self, stamp):
        if self._last_lane_result is None:
            return False
        self._publish_lane_result(stamp, self._last_lane_result)
        return True

    def _publish_lane_result(self, stamp, result):
        # center_path의 coef 저장 (사람 감지 기준선으로 사용)
        all_fits = result.get("all_fits", [])
        if all_fits:
            self._lane_center_coef = all_fits[0]  # yellow_fit = center

        target_x, target_y = result["target"]
        sample_xs = result["sample_xs"]

        # center_path freeze: 보행자 감지 중이면 직전 경로를 재발행 (흔들림 방지)
        if self._path_freeze > 0 and self._frozen_center is not None:
            pub_xs, pub_ys = self._frozen_center
            self._path_freeze -= 1
        else:
            pub_xs, pub_ys = sample_xs, result["sample_ys"]
            self._frozen_center = (pub_xs, pub_ys)

        # OVERTAKE 횡 오프셋 (노란 중앙선 기준 좌+/우- 시프트). LANE 기본은 0.
        offset = getattr(self, "_lane_offset", 0.0)
        left_ys = result["left_ys"]
        right_ys = result["right_ys"]
        if offset != 0.0:
            pub_ys = np.clip(np.asarray(pub_ys, dtype=np.float64) + offset,
                             -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
            target_y = float(np.clip(target_y + offset, -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
            left_ys = np.clip(np.asarray(left_ys, dtype=np.float64) + offset,
                              -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
            right_ys = np.clip(np.asarray(right_ys, dtype=np.float64) + offset,
                               -TARGET_Y_LIMIT, TARGET_Y_LIMIT)

        self._publish_target(stamp, target_x, target_y)
        self._pub_center.publish(_make_pose_array(stamp, pub_xs, pub_ys))
        self._pub_left.publish(_make_pose_array(stamp, sample_xs, left_ys))
        self._pub_right.publish(_make_pose_array(stamp, sample_xs, right_ys))

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
