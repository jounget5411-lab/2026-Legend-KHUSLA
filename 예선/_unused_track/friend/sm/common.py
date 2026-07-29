"""공용 상수 + 검증된 자율주행 로직.

기능 모음:
  1. BEV 호모그래피·그리드 상수 (lane_detect, viewer 공통)
  2. 차선 마스크 추출 / 차선 픽셀 → lidar_frame 좌표 / 차선 품질 점수
  3. 라이다 클러스터링 (장애물 ROI + 가까운 점 묶기)
  4. 오른쪽 실선 2차 fit + preview path 생성 + EMA memory + path reanchor
  5. stale/quality 게이트 헬퍼
  6. legacy 단일 타겟 조향 / 속도 계산
  7. 쿼터니언 → 회전행렬 / 오일러 변환 (viewer 호모그래피·IMU 용)

planner / motion 분리:
  - planner: plan_drive_target() — 오른쪽 실선 fit, preview path, memory reanchor까지
  - motion:  motion_node.py — /motion/path 다점 pursuit + heading 가중평균
"""

import math
import time

import cv2
import numpy as np


# ============================================================
# BEV / 라이다 좌표계
# ============================================================

LIDAR_FRAME = "lidar_frame"
CAM_FRONT_FRAME = "usb_cam_front"
LIDAR_HEIGHT_ABOVE_GROUND_M = 0.228
GROUND_Z_IN_LIDAR = -LIDAR_HEIGHT_ABOVE_GROUND_M

BEV_RES = 0.05
BEV_X_MIN, BEV_X_MAX = -2.0, 12.0
BEV_Y_HALF = 5.0
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))

M_GRID = np.array([
    [ 0.0,         -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,         BEV_X_MAX  / BEV_RES],
    [ 0.0,           0.0,          1.0],
], dtype=np.float64)

R_ROBOT_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

# lane_detect의 fallback 호모그래피 (캘리브된 값, tf 없을 때 사용)
H_PIX2LIDAR_FALLBACK = np.array([
    [ 0.00000000000000e+00,  4.63055528548851e-03, -2.73689017508590e+00],
    [ 4.18097842466549e-03,  1.87301533010043e-04, -1.37819064484720e+00],
    [ 0.00000000000000e+00, -2.40130177039888e-03,  5.16378846754945e-01],
], dtype=np.float64)


# ============================================================
# 차선 마스크 파라미터
# ============================================================

LANE_WHITE_V_MIN  = 200
LANE_WHITE_S_MAX  = 50
LANE_YELLOW_H_MIN = 18
LANE_YELLOW_H_MAX = 38
LANE_YELLOW_S_MIN = 80
LANE_YELLOW_V_MIN = 130
LANE_PIX_SUBSAMPLE = 3
LANE_RUN_MIN_PX = 2
LANE_RUN_MAX_PX = 35
LANE_CC_MIN_AREA = 6
LANE_CC_MIN_H = 3


# ============================================================
# 라이다 / 장애물 파라미터
# ============================================================

OBSTACLE_X_MIN, OBSTACLE_X_MAX = 0.1, 10.0
OBSTACLE_Y_HALF = 4.5
CLUSTER_GAP = 0.25
CLUSTER_MIN_PTS = 3
OBSTACLE_DRAW_PADDING = 0.05


# ============================================================
# stale / 품질 게이트
# ============================================================

PERCEPTION_STALE_SEC = 0.5
LANE_SOURCE_MIN_POINTS = 12
LANE_SOURCE_GOOD_POINTS = 250
LANE_SOURCE_GOOD_SPAN_M = 4.0
LANE_QUALITY_MIN_FOR_DRIVE = 0.15


# ============================================================
# 차선 fit / path / memory 파라미터
# ============================================================

LANE_WIDTH_SINGLE = 3.0
LANE_MODEL_X_MIN = 0.4
LANE_MODEL_X_MAX = 9.0
LANE_SIDE_SPLIT_Y = 0.15
LANE_SIDE_MIN_POINTS = 6
LANE_PAIR_CLUSTER_GAP = 0.45
LANE_PAIR_ASSIGN_HALF_WIDTH = 0.70
LANE_KEEP_MARGIN = 0.20
LANE_MEMORY_MAX_AGE = 20
LANE_COEF_EMA_ALPHA = 0.35

# 차선 추종 모드 — 오른쪽 실선을 기준으로 차량 중심 경로를 만든다.
RIGHT_SOLID_FOLLOW_OFFSET = 0.5 * LANE_WIDTH_SINGLE
LANE_FOLLOW_OFFSET = RIGHT_SOLID_FOLLOW_OFFSET
CENTER_BAND_M = 0.12              # 중앙 근처 조향 저감 폭. 0으로 죽이지 않고 부드럽게 따라감.

# 직각 / 연속 커브 대응 — raw 픽셀 직접 연결 대신 x-bin median 경로 + Pure Pursuit
MIN_POINTS_FOR_DIRECT_PATH = 15   # sampled path 시도 최소 raw 점 수
PATH_BIN_STEP_M = 0.25
PATH_BIN_MIN_POINTS = 2
PATH_MIN_SAMPLED_POINTS = 6
PATH_BIN_MAX_GAP_M = 1.25
PATH_BIN_MAX_LATERAL_JUMP_M = 0.55
PATH_BIN_MAX_SLOPE = 1.15
PATH_SMOOTH_WINDOW = 5
CURVE_PREVIEW_EXTRA_MAX_M = 1.20   # 커브 전에 lookahead를 더 앞쪽으로 보내 미리 진입
CURVE_SIDE_LATERAL_DELTA_M = 0.60  # 옆 차선이 이만큼 횡방향으로 벌어지면 커브로 본다
CURVE_SIDE_HEADING_RAD = 0.28      # 선 자체가 옆으로 누우면 곡률이 작아도 커브로 본다
CURVE_ENTER_KAPPA = 0.030          # 이 이상일 때만 직선에서 커브 모드 진입
CURVE_EXIT_KAPPA = 0.012           # 이 밑으로 충분히 내려가면 오른쪽 실선으로 복귀
CURVE_ENTER_TURN_SCORE = 1.15
CURVE_EXIT_TURN_SCORE = 0.45
CURVE_HOLD_FRAMES = 8              # 커브 종료/끊김에서 바로 직선화되지 않게 유지
CURVE_MEMORY_MAX_AGE = 28
PATH_REANCHOR_DISTANCE_M = 0.45    # 차량 원점에서 경로까지 이 이상 떨어지면 복귀 모드 표시
CORNER_PRETURN_EXTRA_MAX_M = 1.00
CORNER_PRETURN_PREVIEW_M = 2.50
OUTER_SOLID_NEAR_MIN_ABS_Y = 0.65  # 중앙 점선을 실선으로 오인하지 않기 위한 near-field 기준
VIRTUAL_CENTER_MERGE_MAX_M = 0.60  # 좌/우 실선 기반 중앙선이 이 안이면 평균
CENTER_DASH_SOLID_OFFSET = LANE_WIDTH_SINGLE  # 실선에서 실제 중앙 점선까지 거리
CENTER_DASH_ASSIGN_HALF_WIDTH = 0.45  # 실선 기반 중앙 예측선 주변 실제 점선 픽셀 허용 폭
CENTER_DASH_DIRECT_MAX_ABS_Y = 1.80
CENTER_DASH_MIN_POINTS = 8


# ============================================================
# 제어 (motion_node) 파라미터
# ============================================================

ANGLE_MIN, ANGLE_MAX = -100.0, 100.0
LOOKAHEAD_X = 3.0
TARGET_X_MIN = 1.2
LOOKAHEAD_CURVE_LIGHT = 2.5
LOOKAHEAD_CURVE_MID = 2.0
LOOKAHEAD_CURVE_TIGHT = 1.5

STEER_GAIN = 4.0
CURVE_GAIN_SCALE = 14.0
CURVE_GAIN_MAX = 1.8
CURV_TIGHT = 0.08
CURV_MID = 0.04
CURV_LIGHT = 0.015
STRAIGHT_TY_DEADBAND = 0.04
STRAIGHT_KAPPA_DEADBAND = 0.010
CENTER_LOW_GAIN_M = 0.45
CENTER_LOW_GAIN_SCALE = 0.45
BOUNDARY_GAIN_NEAR_M = 0.30
BOUNDARY_GAIN_MAX = 0.9
BASE_SPEED = 10.0


# ============================================================
# 기본 수학 헬퍼
# ============================================================

def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = (math.copysign(math.pi / 2.0, sinp)
             if abs(sinp) >= 1.0 else math.asin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def stamp_msg_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def is_perception_fresh(stamp, now=None):
    if stamp <= 0.0:
        return False
    if now is None:
        now = time.monotonic()
    return (now - stamp) <= PERCEPTION_STALE_SEC


# ============================================================
# 차선 마스크 추출
# ============================================================

def extract_lane_mask(bev_img):
    if bev_img is None or bev_img.size == 0:
        return None
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (V >= LANE_WHITE_V_MIN) & (S <= LANE_WHITE_S_MAX)
    yellow = ((H >= LANE_YELLOW_H_MIN) & (H <= LANE_YELLOW_H_MAX)
              & (S >= LANE_YELLOW_S_MIN) & (V >= LANE_YELLOW_V_MIN))
    valid = V > 5
    mask = ((white | yellow) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    thin = np.zeros_like(mask)
    for v in range(mask.shape[0]):
        cols = np.flatnonzero(mask[v] > 0)
        if cols.size == 0:
            continue
        cuts = np.where(np.diff(cols) > 1)[0] + 1
        for run in np.split(cols, cuts):
            if LANE_RUN_MIN_PX <= run.size <= LANE_RUN_MAX_PX:
                thin[v, run] = 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    filtered = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= LANE_CC_MIN_AREA and h >= LANE_CC_MIN_H:
            filtered[labels == i] = 255
    return filtered


def lane_mask_to_lidar_xy(mask):
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    right_m = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    forward_m = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return forward_m, -right_m


def extract_cv_lane_data(mask):
    """마스크 → (lane_xs, lane_ys, point_count, quality 0~1)."""
    lx, ly = lane_mask_to_lidar_xy(mask)
    if lx.size == 0:
        return np.array([]), np.array([]), 0, 0.0

    valid = (np.isfinite(lx) & np.isfinite(ly)
             & (lx >= BEV_X_MIN) & (lx <= BEV_X_MAX)
             & (np.abs(ly) <= BEV_Y_HALF))
    lx = lx[valid]
    ly = ly[valid]
    point_count = int(lx.size)
    if point_count < LANE_SOURCE_MIN_POINTS:
        return np.array([]), np.array([]), point_count, 0.0

    span_x = float(np.max(lx) - np.min(lx)) if point_count > 1 else 0.0
    count_score = min(1.0, point_count / float(LANE_SOURCE_GOOD_POINTS))
    span_score = min(1.0, span_x / LANE_SOURCE_GOOD_SPAN_M)
    quality = 0.65 * count_score + 0.35 * span_score
    return lx, ly, point_count, float(np.clip(quality, 0.0, 1.0))


# ============================================================
# 라이다 클러스터링
# ============================================================

def cluster_scan_obstacles(xs, ys):
    if xs.size == 0:
        return []
    pts = np.column_stack([xs, ys])
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    splits = np.where(d > CLUSTER_GAP)[0] + 1
    groups = np.split(pts, splits)
    out = []
    for g in groups:
        if len(g) < CLUSTER_MIN_PTS:
            continue
        cx, cy = g.mean(axis=0)
        r = float(np.linalg.norm(g - [cx, cy], axis=1).max()) + OBSTACLE_DRAW_PADDING
        out.append((float(cx), float(cy), r))
    return out


def extract_lidar_obstacle_data(xs, ys):
    if xs.size == 0:
        return [], 0, 0.0
    valid = np.isfinite(xs) & np.isfinite(ys)
    roi = (valid
           & (xs >= OBSTACLE_X_MIN) & (xs <= OBSTACLE_X_MAX)
           & (np.abs(ys) <= OBSTACLE_Y_HALF))
    point_count = int(np.count_nonzero(roi))
    if point_count == 0:
        return [], 0, 0.0

    clusters = cluster_scan_obstacles(xs[roi], ys[roi])
    if not clusters:
        return [], point_count, 0.0

    quality = min(1.0, point_count / float(max(CLUSTER_MIN_PTS * 4, 1)))
    return clusters, point_count, float(quality)


# ============================================================
# 차선 fit / lookahead / path / EMA memory
# ============================================================
# planner는 LaneMemory 인스턴스를 하나 들고 다니며 plan_drive_target에 넘긴다.
# (전역 dict 대신 인스턴스 — 노드 재시작 시 자동 리셋)
# ============================================================

class LaneMemory:
    def __init__(self):
        self.left_coef = None
        self.right_coef = None
        self.center_coef = None
        self.width = LANE_WIDTH_SINGLE
        self.last_target = None
        self.last_path = None
        self.last_path_age = 999
        self.left_age = 999
        self.right_age = 999
        self.center_age = 999
        self.followed_side = None  # "left" / "right" / None — 마지막에 추종한 쪽
        self.curve_hold = 0
        self.last_curve_path = None
        self.last_curve_age = 999
        self.last_curve_side = None

    def age(self):
        self.left_age += 1
        self.right_age += 1
        self.center_age += 1
        self.last_path_age += 1
        self.last_curve_age += 1
        if self.curve_hold > 0:
            self.curve_hold -= 1

    def fresh(self, side):
        return getattr(self, f"{side}_age") <= LANE_MEMORY_MAX_AGE


def _pad_coef(coef, deg=2):
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=np.float64)
    if coef.size >= deg + 1:
        return coef[-(deg + 1):]
    return np.pad(coef, (deg + 1 - coef.size, 0))


def _ema_coef(prev, new):
    new = _pad_coef(new)
    if prev is None:
        return new
    return LANE_COEF_EMA_ALPHA * new + (1.0 - LANE_COEF_EMA_ALPHA) * _pad_coef(prev)


def _eval_lane(coef, x):
    if coef is None:
        return None
    return float(np.polyval(coef, x))


def _curve_mag(coef):
    coef = _pad_coef(coef)
    if coef is None:
        return 0.0
    return abs(float(coef[0]))


def _lookahead_for_curve(coef):
    kappa = _curve_mag(coef)
    if kappa > CURV_TIGHT:
        return LOOKAHEAD_CURVE_TIGHT
    if kappa > CURV_MID:
        return LOOKAHEAD_CURVE_MID
    if kappa > CURV_LIGHT:
        return LOOKAHEAD_CURVE_LIGHT
    return LOOKAHEAD_X


def _fit_lane_side(xs, ys):
    if xs.size < LANE_SIDE_MIN_POINTS:
        return None
    deg = 2 if xs.size >= 10 else 1
    weights = 1.0 + 0.2 * np.clip(xs, 0.0, LANE_MODEL_X_MAX)
    try:
        return np.polyfit(xs, ys, deg=deg, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _reference_center_y(memory, x_value):
    if memory.fresh("center") and memory.center_coef is not None:
        return _eval_lane(memory.center_coef, x_value)
    if memory.last_target is not None:
        return float(memory.last_target[1])
    return 0.0


def _visible_path_xmax(lane_xs):
    if lane_xs is None or len(lane_xs) == 0:
        return LOOKAHEAD_X
    xs = np.asarray(lane_xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    xs = xs[(xs >= LANE_MODEL_X_MIN) & (xs <= LANE_MODEL_X_MAX)]
    if xs.size == 0:
        return LOOKAHEAD_X
    return float(np.clip(np.percentile(xs, 95), LOOKAHEAD_X, LANE_MODEL_X_MAX))


# ============================================================
# 게인 (motion에서 사용)
# ============================================================

def boundary_gain(target_y, lane_bounds):
    if lane_bounds is None:
        return 1.0
    left_y, right_y = lane_bounds
    if (left_y is None or right_y is None
            or not math.isfinite(left_y) or not math.isfinite(right_y)
            or left_y <= right_y):
        return 1.0
    min_gap = min(left_y - target_y, target_y - right_y)
    if min_gap >= BOUNDARY_GAIN_NEAR_M:
        return 1.0
    closeness = max(0.0, BOUNDARY_GAIN_NEAR_M - min_gap) / BOUNDARY_GAIN_NEAR_M
    return 1.0 + BOUNDARY_GAIN_MAX * closeness


def center_gain(target_y, kappa):
    if kappa >= CURV_LIGHT or abs(target_y) >= CENTER_LOW_GAIN_M:
        return 1.0
    ratio = abs(target_y) / CENTER_LOW_GAIN_M
    return CENTER_LOW_GAIN_SCALE + (1.0 - CENTER_LOW_GAIN_SCALE) * ratio


# ============================================================
# planner 진입점 — 차선 fit → target/path/bounds 산출
# ============================================================

class DriveTarget:
    """plan_drive_target 결과 묶음."""
    __slots__ = ("ok", "target_x", "target_y", "kappa",
                 "path_xs", "path_ys", "lane_bounds", "mode")

    def __init__(self, ok, target_x, target_y, kappa,
                 path_xs, path_ys, lane_bounds, mode):
        self.ok = ok
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.kappa = float(kappa)
        self.path_xs = path_xs
        self.path_ys = path_ys
        self.lane_bounds = lane_bounds
        self.mode = mode


def _stop_target(mode, fallback_x=LOOKAHEAD_X):
    return DriveTarget(
        ok=False,
        target_x=fallback_x,
        target_y=0.0,
        kappa=0.0,
        path_xs=np.array([], dtype=np.float32),
        path_ys=np.array([], dtype=np.float32),
        lane_bounds=None,
        mode=mode,
    )


def _target_from_memory_path(memory: LaneMemory, mode):
    """검출이 끊겼을 때 마지막 preview path 위 가까운 점으로 다시 붙는다.

    커브 중 끊긴 경우에는 마지막 커브 path를 앞쪽으로 연장해서 계속 돌게 한다.
    """
    use_curve_memory = (
        memory.last_curve_path is not None
        and memory.last_curve_age <= CURVE_MEMORY_MAX_AGE
        and (memory.curve_hold > 0
             or memory.followed_side == "center"
             or _looks_like_curve_path(*memory.last_curve_path, enter=False))
    )

    if use_curve_memory:
        path_src = memory.last_curve_path
        age = memory.last_curve_age
        max_age = CURVE_MEMORY_MAX_AGE
    else:
        path_src = memory.last_path
        age = memory.last_path_age
        max_age = LANE_MEMORY_MAX_AGE

    if path_src is None or age > max_age:
        return _stop_target(mode)

    path_xs, path_ys = path_src
    if path_xs is None or path_ys is None:
        return _stop_target(mode)
    path_xs = np.asarray(path_xs, dtype=np.float32)
    path_ys = np.asarray(path_ys, dtype=np.float32)
    if path_xs.size < 2 or path_ys.size != path_xs.size:
        return _stop_target(mode)

    if use_curve_memory:
        extended_xs, extended_ys = _extend_curve_path(path_xs, path_ys)
        if extended_xs is not None and extended_ys is not None:
            path_xs, path_ys = extended_xs, extended_ys

    path_kappa = _path_preview_curve_mag(path_xs, path_ys)
    turn_score = _path_turn_score(path_xs, path_ys)
    anchor_idx, anchor_dist = _nearest_path_index(path_xs, path_ys, 0.0, 0.0)
    L = _curve_preview_lookahead(LOOKAHEAD_X, path_kappa)
    if use_curve_memory:
        L += CURVE_PREVIEW_EXTRA_MAX_M * float(np.clip(turn_score, 0.0, 1.0))
    L += _corner_preturn_extra(path_xs, path_ys, anchor_idx)
    target_x, target_y = _target_at_lookahead_from_index(
        path_xs, path_ys, anchor_idx, L)
    if target_x is None:
        return _stop_target(mode)

    mode_out = mode + "-memory-reanchor"
    if use_curve_memory:
        mode_out += "-curve-hold"
    if anchor_dist > PATH_REANCHOR_DISTANCE_M:
        mode_out += "-far"
    if L > LOOKAHEAD_X + 0.1:
        mode_out += "-preview"

    bound_half = CENTER_DASH_SOLID_OFFSET if use_curve_memory else LANE_FOLLOW_OFFSET
    lane_bounds = (
        float(target_y + bound_half),
        float(target_y - bound_half),
    )
    memory.last_target = (float(target_x), float(target_y))

    return DriveTarget(
        ok=True,
        target_x=max(TARGET_X_MIN, float(target_x)),
        target_y=float(target_y),
        kappa=float(path_kappa),
        path_xs=path_xs,
        path_ys=path_ys,
        lane_bounds=lane_bounds,
        mode=mode_out,
    )


def _outer_lane_mask_by_x_bins(x, y, side):
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    bin_ids = np.floor((x - LANE_MODEL_X_MIN) / PATH_BIN_STEP_M).astype(np.int32)
    mask = np.zeros(x.size, dtype=bool)
    for bid in np.unique(bin_ids):
        idx = np.flatnonzero(bin_ids == bid)
        if idx.size < PATH_BIN_MIN_POINTS:
            continue
        yb = y[idx]
        if side == "right":
            edge = float(np.min(yb))
            keep = yb <= edge + LANE_PAIR_CLUSTER_GAP
        else:
            edge = float(np.max(yb))
            keep = yb >= edge - LANE_PAIR_CLUSTER_GAP
        mask[idx[keep]] = True
    return mask


def _fit_rightmost_lane(lane_xs, lane_ys, memory: LaneMemory):
    """오른쪽 실선 기반 중앙 점선 보강용: 현재 보이는 가장 오른쪽 라인만 fit."""
    fallback = memory.right_coef if memory.fresh("right") else None
    empty = np.array([], dtype=np.float64)
    if lane_xs is None or lane_ys is None or lane_xs.size <= 10:
        return empty, empty, fallback

    x = np.asarray(lane_xs, dtype=np.float64)
    y = np.asarray(lane_ys, dtype=np.float64)
    valid = (np.isfinite(x) & np.isfinite(y)
             & (x >= LANE_MODEL_X_MIN)
             & (x <= LANE_MODEL_X_MAX)
             & (np.abs(y) <= BEV_Y_HALF - 0.2))
    x = x[valid]
    y = y[valid]
    if x.size <= 10:
        return empty, empty, fallback

    mask = _outer_lane_mask_by_x_bins(x, y, "right")
    if np.count_nonzero(mask) < LANE_SIDE_MIN_POINTS:
        band = ((x >= max(LANE_MODEL_X_MIN, LOOKAHEAD_X - 2.0))
                & (x <= min(LANE_MODEL_X_MAX, LOOKAHEAD_X + 2.0)))
        y_ref = y[band] if np.count_nonzero(band) >= LANE_SIDE_MIN_POINTS else y
        if y_ref.size < LANE_SIDE_MIN_POINTS:
            return empty, empty, fallback

        y_sorted = np.sort(y_ref)
        splits = np.where(np.diff(y_sorted) > LANE_PAIR_CLUSTER_GAP)[0] + 1
        groups = np.split(y_sorted, splits)
        centers = [
            float(np.median(g)) for g in groups
            if g.size >= LANE_SIDE_MIN_POINTS
        ]
        if not centers:
            return empty, empty, fallback
        right_center = min(centers)
        mask = np.abs(y - right_center) <= LANE_PAIR_ASSIGN_HALF_WIDTH

    right_xs = x[mask].astype(np.float64)
    right_ys = y[mask].astype(np.float64)
    coef = _fit_lane_side(right_xs, right_ys)
    if coef is None:
        coef = fallback
    return right_xs, right_ys, coef


def _fit_leftmost_lane(lane_xs, lane_ys, memory: LaneMemory):
    """왼쪽 실선 기반 중앙점선 생성용: 현재 보이는 차선 중 가장 왼쪽 라인만 fit."""
    fallback = memory.left_coef if memory.fresh("left") else None
    empty = np.array([], dtype=np.float64)
    if lane_xs is None or lane_ys is None or lane_xs.size <= 10:
        return empty, empty, fallback

    x = np.asarray(lane_xs, dtype=np.float64)
    y = np.asarray(lane_ys, dtype=np.float64)
    valid = (np.isfinite(x) & np.isfinite(y)
             & (x >= LANE_MODEL_X_MIN)
             & (x <= LANE_MODEL_X_MAX)
             & (np.abs(y) <= BEV_Y_HALF - 0.2))
    x = x[valid]
    y = y[valid]
    if x.size <= 10:
        return empty, empty, fallback

    mask = _outer_lane_mask_by_x_bins(x, y, "left")
    if np.count_nonzero(mask) < LANE_SIDE_MIN_POINTS:
        band = ((x >= max(LANE_MODEL_X_MIN, LOOKAHEAD_X - 2.0))
                & (x <= min(LANE_MODEL_X_MAX, LOOKAHEAD_X + 2.0)))
        y_ref = y[band] if np.count_nonzero(band) >= LANE_SIDE_MIN_POINTS else y
        if y_ref.size < LANE_SIDE_MIN_POINTS:
            return empty, empty, fallback

        y_sorted = np.sort(y_ref)
        splits = np.where(np.diff(y_sorted) > LANE_PAIR_CLUSTER_GAP)[0] + 1
        groups = np.split(y_sorted, splits)
        centers = [
            float(np.median(g)) for g in groups
            if g.size >= LANE_SIDE_MIN_POINTS
        ]
        if not centers:
            return empty, empty, fallback
        left_center = max(centers)
        mask = np.abs(y - left_center) <= LANE_PAIR_ASSIGN_HALF_WIDTH

    left_xs = x[mask].astype(np.float64)
    left_ys = y[mask].astype(np.float64)
    coef = _fit_lane_side(left_xs, left_ys)
    if coef is None:
        coef = fallback
    return left_xs, left_ys, coef


def _looks_like_outer_solid(coef, side, raw_xs=None, raw_ys=None):
    if raw_xs is not None and raw_ys is not None and raw_xs.size > 0:
        x = np.asarray(raw_xs, dtype=np.float64)
        y = np.asarray(raw_ys, dtype=np.float64)
        near = (np.isfinite(x) & np.isfinite(y)
                & (x >= LANE_MODEL_X_MIN)
                & (x <= LOOKAHEAD_CURVE_MID))
        if np.count_nonzero(near) >= LANE_SIDE_MIN_POINTS:
            lane_y = float(np.median(y[near]))
            if side == "right":
                return lane_y <= -OUTER_SOLID_NEAR_MIN_ABS_Y
            return lane_y >= OUTER_SOLID_NEAR_MIN_ABS_Y

    lane_y = _eval_lane(coef, TARGET_X_MIN)
    if lane_y is None:
        return False
    if side == "right":
        return lane_y <= -OUTER_SOLID_NEAR_MIN_ABS_Y
    return lane_y >= OUTER_SOLID_NEAR_MIN_ABS_Y


def _points_near_lane_fit(lane_xs, lane_ys, coef):
    if coef is None or lane_xs is None or lane_ys is None or lane_xs.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = np.asarray(lane_xs, dtype=np.float64)
    y = np.asarray(lane_ys, dtype=np.float64)
    valid = (np.isfinite(x) & np.isfinite(y)
             & (x >= LANE_MODEL_X_MIN)
             & (x <= LANE_MODEL_X_MAX)
             & (np.abs(y) <= BEV_Y_HALF - 0.2))
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    fit_y = np.polyval(_pad_coef(coef), x)
    near = np.abs(y - fit_y) <= LANE_PAIR_ASSIGN_HALF_WIDTH
    return x[near].astype(np.float64), y[near].astype(np.float64)


def _fit_direct_center_dash(lane_xs, lane_ys, memory: LaneMemory):
    """실제 중앙 점선 직접 추출. 실선 기반 보정이 없을 때도 중앙 점선을 우선한다."""
    fallback = memory.center_coef if memory.fresh("center") else None
    empty = np.array([], dtype=np.float64)
    if lane_xs is None or lane_ys is None or lane_xs.size <= 10:
        return empty, empty, fallback

    x = np.asarray(lane_xs, dtype=np.float64)
    y = np.asarray(lane_ys, dtype=np.float64)
    valid = (np.isfinite(x) & np.isfinite(y)
             & (x >= LANE_MODEL_X_MIN)
             & (x <= LANE_MODEL_X_MAX)
             & (np.abs(y) <= BEV_Y_HALF - 0.2))
    x = x[valid]
    y = y[valid]
    if x.size <= 10:
        return empty, empty, fallback

    band = ((x >= max(LANE_MODEL_X_MIN, LOOKAHEAD_X - 2.0))
            & (x <= min(LANE_MODEL_X_MAX, LOOKAHEAD_X + 2.0)))
    y_ref = y[band] if np.count_nonzero(band) >= LANE_SIDE_MIN_POINTS else y
    if y_ref.size < LANE_SIDE_MIN_POINTS:
        return empty, empty, fallback

    y_sorted = np.sort(y_ref)
    splits = np.where(np.diff(y_sorted) > LANE_PAIR_CLUSTER_GAP)[0] + 1
    groups = np.split(y_sorted, splits)
    centers = [
        float(np.median(g)) for g in groups
        if g.size >= LANE_SIDE_MIN_POINTS
    ]
    if not centers:
        return empty, empty, fallback

    ref_y = _reference_center_y(memory, LOOKAHEAD_X)
    center_y = min(centers, key=lambda c: abs(c - ref_y))
    if abs(center_y - ref_y) > CENTER_DASH_DIRECT_MAX_ABS_Y:
        return empty, empty, fallback

    mask = np.abs(y - center_y) <= LANE_PAIR_ASSIGN_HALF_WIDTH
    dash_xs = x[mask].astype(np.float64)
    dash_ys = y[mask].astype(np.float64)
    coef = _fit_lane_side(dash_xs, dash_ys)
    if coef is None:
        coef = fallback
    return dash_xs, dash_ys, coef


def _smooth_series(values, window):
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    window = int(window)
    if n < 3 or window <= 1:
        return values
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _make_sampled_offset_path(raw_xs, raw_ys, visible_xmax, offset_m):
    """x-bin median 기반 차선 경로.

    두꺼운 픽셀 구름은 그대로 이으면 루프가 생긴다. 대신 전방 x축으로
    잘라 각 구간 median y만 쓰면 순서는 안정되고 S자 굴곡은 유지된다.
    """
    if raw_xs is None or raw_ys is None or raw_xs.size < MIN_POINTS_FOR_DIRECT_PATH:
        return None, None

    raw_xs = np.asarray(raw_xs, dtype=np.float64)
    raw_ys = np.asarray(raw_ys, dtype=np.float64)
    valid = (np.isfinite(raw_xs) & np.isfinite(raw_ys)
             & (raw_xs >= LANE_MODEL_X_MIN)
             & (raw_xs <= LANE_MODEL_X_MAX)
             & (np.abs(raw_ys) <= BEV_Y_HALF - 0.2))
    x = raw_xs[valid]
    y = raw_ys[valid]
    if x.size < MIN_POINTS_FOR_DIRECT_PATH:
        return None, None

    x_end = max(TARGET_X_MIN, float(visible_xmax))
    x_end = float(np.clip(x_end, TARGET_X_MIN, LANE_MODEL_X_MAX))
    bin_ids = np.floor((x - LANE_MODEL_X_MIN) / PATH_BIN_STEP_M).astype(np.int32)

    bx = []
    by = []
    for bid in np.unique(bin_ids):
        mask = bin_ids == bid
        if np.count_nonzero(mask) < PATH_BIN_MIN_POINTS:
            continue
        mx = float(np.median(x[mask]))
        if mx > x_end:
            continue
        bx.append(mx)
        by.append(float(np.median(y[mask])))

    if len(bx) < PATH_MIN_SAMPLED_POINTS:
        return None, None

    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    order = np.argsort(bx)
    bx = bx[order]
    by = by[order]

    clean_x = [float(bx[0])]
    clean_y = [float(by[0])]
    for px, py in zip(bx[1:], by[1:]):
        dx = float(px - clean_x[-1])
        dy = float(py - clean_y[-1])
        if dx <= 1e-6:
            continue
        if dx > PATH_BIN_MAX_GAP_M and len(clean_x) >= PATH_MIN_SAMPLED_POINTS:
            break
        max_jump = PATH_BIN_MAX_LATERAL_JUMP_M + PATH_BIN_MAX_SLOPE * dx
        if abs(dy) > max_jump:
            continue
        clean_x.append(float(px))
        clean_y.append(float(py))

    if len(clean_x) < PATH_MIN_SAMPLED_POINTS:
        return None, None

    clean_x = np.asarray(clean_x, dtype=np.float64)
    clean_y = _smooth_series(np.asarray(clean_y, dtype=np.float64), PATH_SMOOTH_WINDOW)
    if clean_x[-1] - clean_x[0] < max(LOOKAHEAD_CURVE_TIGHT, PATH_BIN_STEP_M * 4):
        return None, None

    out_x = np.arange(clean_x[0], clean_x[-1] + PATH_BIN_STEP_M * 0.5,
                      PATH_BIN_STEP_M, dtype=np.float64)
    out_y = np.interp(out_x, clean_x, clean_y) + float(offset_m)
    return out_x.astype(np.float32), out_y.astype(np.float32)


def _path_preview_curve_mag(path_xs, path_ys):
    if path_xs is None or path_ys is None or path_xs.size < 5:
        return 0.0
    xs = path_xs.astype(np.float64)
    ys = path_ys.astype(np.float64)
    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
        return 0.0
    if np.any(np.diff(xs) <= 1e-6):
        return 0.0
    try:
        dy = np.gradient(ys, xs)
        ddy = np.gradient(dy, xs)
    except (FloatingPointError, ValueError):
        return 0.0
    curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
    curv = curv[np.isfinite(curv)]
    if curv.size == 0:
        return 0.0
    return float(np.percentile(curv, 85))


def _path_turn_score(path_xs, path_ys):
    """짧은 커브 진입부처럼 거의 직선 fit이어도 옆으로 뻗으면 커브로 잡는다."""
    if path_xs is None or path_ys is None or path_xs.size < 2:
        return 0.0
    xs = np.asarray(path_xs, dtype=np.float64)
    ys = np.asarray(path_ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 2:
        return 0.0
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    dx = max(float(xs[-1] - xs[0]), 1e-3)
    lateral = float(np.max(np.abs(ys - ys[0])))
    heading = abs(math.atan2(float(ys[-1] - ys[0]), dx))
    return max(
        lateral / max(CURVE_SIDE_LATERAL_DELTA_M, 1e-6),
        heading / max(CURVE_SIDE_HEADING_RAD, 1e-6),
    )


def _looks_like_curve_path(path_xs, path_ys, enter=True):
    kappa = _path_preview_curve_mag(path_xs, path_ys)
    turn_score = _path_turn_score(path_xs, path_ys)
    if enter:
        return kappa >= CURVE_ENTER_KAPPA or turn_score >= CURVE_ENTER_TURN_SCORE
    return kappa >= CURVE_EXIT_KAPPA or turn_score >= CURVE_EXIT_TURN_SCORE


def _extend_curve_path(path_xs, path_ys):
    """차선이 잠깐 끊겨도 마지막 커브 방향으로 path를 앞으로 연장한다."""
    if path_xs is None or path_ys is None:
        return None, None
    xs = np.asarray(path_xs, dtype=np.float64)
    ys = np.asarray(path_ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= LANE_MODEL_X_MIN)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 3:
        return None, None

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    keep = np.concatenate(([True], np.diff(xs) > 1e-4))
    xs = xs[keep]
    ys = ys[keep]
    if xs.size < 3:
        return None, None

    deg = 2 if xs.size >= 5 else 1
    weights = 1.0 + 0.35 * np.clip(xs, 0.0, LANE_MODEL_X_MAX)
    try:
        coef = np.polyfit(xs, ys, deg=deg, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    start_x = max(0.5, float(xs[0]))
    x_end = LANE_MODEL_X_MAX
    out_x = np.arange(start_x, x_end + PATH_BIN_STEP_M * 0.5,
                      PATH_BIN_STEP_M, dtype=np.float64)
    out_y = np.polyval(coef, out_x)

    measured = out_x <= float(xs[-1])
    if np.count_nonzero(measured) > 0:
        out_y[measured] = np.interp(out_x[measured], xs, ys)

    out_y = _smooth_series(
        np.clip(out_y, -BEV_Y_HALF + 0.25, BEV_Y_HALF - 0.25),
        min(PATH_SMOOTH_WINDOW, 3),
    )
    return out_x.astype(np.float32), out_y.astype(np.float32)


def _curve_preview_lookahead(base_L, path_kappa):
    if path_kappa <= CURV_LIGHT:
        return float(base_L)
    denom = max(CURV_TIGHT - CURV_LIGHT, 1e-6)
    ratio = np.clip((path_kappa - CURV_LIGHT) / denom, 0.0, 1.0)
    extra = CURVE_PREVIEW_EXTRA_MAX_M * float(ratio)
    return float(min(LOOKAHEAD_X + CURVE_PREVIEW_EXTRA_MAX_M, base_L + extra))


def _nearest_path_index(path_xs, path_ys, anchor_x=0.0, anchor_y=0.0):
    if path_xs is None or path_ys is None or path_xs.size == 0:
        return 0, float("inf")
    xs = path_xs.astype(np.float64)
    ys = path_ys.astype(np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0)
    if not valid.any():
        valid = np.isfinite(xs) & np.isfinite(ys)
    cand = np.flatnonzero(valid)
    if cand.size == 0:
        return 0, float("inf")
    dx = xs[cand] - float(anchor_x)
    dy = ys[cand] - float(anchor_y)
    local = int(np.argmin(dx * dx + dy * dy))
    idx = int(cand[local])
    return idx, float(math.hypot(float(dx[local]), float(dy[local])))


def _target_at_lookahead_from_index(path_xs, path_ys, start_idx, L):
    if path_xs is None or path_ys is None or path_xs.size == 0:
        return None, None
    n = int(path_xs.size)
    start_idx = int(np.clip(start_idx, 0, n - 1))
    if start_idx >= n - 1:
        return float(path_xs[start_idx]), float(path_ys[start_idx])

    xs = path_xs.astype(np.float64)
    ys = path_ys.astype(np.float64)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    dist = np.concatenate(([0.0], np.cumsum(seg)))
    start_dist = float(dist[start_idx])
    target_dist = start_dist + max(0.0, float(L))

    idx = int(np.searchsorted(dist, target_dist, side="left"))
    if idx <= start_idx:
        return float(path_xs[start_idx]), float(path_ys[start_idx])
    if idx >= n:
        return float(path_xs[-1]), float(path_ys[-1])

    d_prev = float(dist[idx - 1])
    d_curr = float(dist[idx])
    frac = (target_dist - d_prev) / max(d_curr - d_prev, 1e-6)
    x = float(path_xs[idx - 1]) + frac * float(path_xs[idx] - path_xs[idx - 1])
    y = float(path_ys[idx - 1]) + frac * float(path_ys[idx] - path_ys[idx - 1])
    return x, y


def _corner_preturn_extra(path_xs, path_ys, start_idx):
    if path_xs is None or path_ys is None or path_xs.size < 4:
        return 0.0
    n = int(path_xs.size)
    start_idx = int(np.clip(start_idx, 0, n - 2))
    xs = path_xs.astype(np.float64)
    ys = path_ys.astype(np.float64)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    dist = np.concatenate(([0.0], np.cumsum(seg)))
    start_dist = float(dist[start_idx])
    end_dist = min(float(dist[-1]), start_dist + CORNER_PRETURN_PREVIEW_M)
    end_idx = int(np.searchsorted(dist, end_dist, side="left"))
    end_idx = int(np.clip(end_idx, start_idx + 1, n - 1))

    dx = max(float(xs[end_idx] - xs[start_idx]), 1e-3)
    lateral = abs(float(ys[end_idx] - ys[start_idx]))
    heading = abs(math.atan2(float(ys[end_idx] - ys[start_idx]), dx))
    score = max(lateral / 0.9, heading / 0.55)
    return float(np.clip(score, 0.0, 1.0) * CORNER_PRETURN_EXTRA_MAX_M)


def _make_offset_path(lane_coef, visible_xmax, target_x, offset_m):
    """차선 polynomial에 lateral offset 더한 차량 중심 경로.

    offset_m > 0 → 좌측(+y)으로 이동 (오른쪽 차선 추종 시)
    offset_m < 0 → 우측(-y)으로 이동 (왼쪽 차선 추종 시)
    """
    x_end = max(target_x, visible_xmax)
    x_end = float(np.clip(x_end, target_x, LANE_MODEL_X_MAX))
    n = max(2, int((x_end - 0.5) / 0.25) + 1)
    xs = np.linspace(0.5, x_end, n, dtype=np.float32)
    lane_ys = np.polyval(_pad_coef(lane_coef), xs)
    ys = (lane_ys + offset_m).astype(np.float32)
    return xs, ys


def _make_center_candidate(raw_xs, raw_ys, coef, lane_xs, lane_ys,
                           visible_xmax, offset, side_label):
    if coef is None:
        return None

    fit_xs, fit_ys = _points_near_lane_fit(lane_xs, lane_ys, coef)
    if fit_xs.size > raw_xs.size:
        raw_xs, raw_ys = fit_xs, fit_ys

    hint_x = max(TARGET_X_MIN, float(_lookahead_for_curve(coef)))
    path_xs, path_ys = _make_sampled_offset_path(
        raw_xs, raw_ys, visible_xmax, offset)
    source = "sampled"
    if path_xs is None or path_ys is None:
        path_xs, path_ys = _make_offset_path(coef, visible_xmax, hint_x, offset)
        source = "poly"
    if path_xs is None or path_ys is None or path_xs.size == 0:
        return None

    score = float(raw_xs.size) + 20.0 * max(0.0, float(path_xs[-1] - path_xs[0]))
    return {
        "side": side_label,
        "source": source,
        "coef": coef,
        "raw_xs": raw_xs,
        "raw_ys": raw_ys,
        "path_xs": path_xs,
        "path_ys": path_ys,
        "score": score,
    }


def _merge_center_candidates(right_cand, left_cand):
    if right_cand is None:
        return left_cand, "left"
    if left_cand is None:
        return right_cand, "right"

    rx = right_cand["path_xs"]
    ry = right_cand["path_ys"]
    lx = left_cand["path_xs"]
    ly = left_cand["path_ys"]

    if rx.size < 2 or lx.size < 2:
        return (right_cand, "right") if right_cand["score"] >= left_cand["score"] else (left_cand, "left")

    if right_cand["score"] >= left_cand["score"]:
        primary, other = right_cand, left_cand
    else:
        primary, other = left_cand, right_cand

    px = primary["path_xs"]
    py = primary["path_ys"].astype(np.float32).copy()
    ox = other["path_xs"]
    oy = other["path_ys"]
    overlap = (px >= float(ox[0])) & (px <= float(ox[-1]))
    if np.count_nonzero(overlap) >= 4:
        interp = np.interp(px[overlap], ox, oy).astype(np.float32)
        diff = np.abs(py[overlap] - interp)
        if float(np.median(diff)) <= VIRTUAL_CENTER_MERGE_MAX_M:
            py[overlap] = 0.5 * (py[overlap] + interp)
            merged = dict(primary)
            merged["path_ys"] = py.astype(np.float32)
            merged["source"] = "both-" + primary["source"]
            return merged, "both"

    return primary, primary["side"]


def _extract_center_dash_points(lane_xs, lane_ys, center_xs, center_ys,
                                right_coef=None, left_coef=None):
    if (lane_xs is None or lane_ys is None or center_xs is None or center_ys is None
            or center_xs.size < 2):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x = np.asarray(lane_xs, dtype=np.float64)
    y = np.asarray(lane_ys, dtype=np.float64)
    valid = (np.isfinite(x) & np.isfinite(y)
             & (x >= float(center_xs[0]))
             & (x <= float(center_xs[-1]))
             & (np.abs(y) <= BEV_Y_HALF - 0.2))
    x = x[valid]
    y = y[valid]
    if x.size < CENTER_DASH_MIN_POINTS:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    center_y = np.interp(x, center_xs, center_ys)
    near_center = np.abs(y - center_y) <= CENTER_DASH_ASSIGN_HALF_WIDTH

    between_outer = np.ones(x.size, dtype=bool)
    if right_coef is not None:
        right_y = np.polyval(_pad_coef(right_coef), x)
        between_outer &= y >= right_y + LANE_KEEP_MARGIN
    if left_coef is not None:
        left_y = np.polyval(_pad_coef(left_coef), x)
        between_outer &= y <= left_y - LANE_KEEP_MARGIN

    mask = near_center & between_outer
    if np.count_nonzero(mask) < CENTER_DASH_MIN_POINTS:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return x[mask].astype(np.float64), y[mask].astype(np.float64)


def _fill_center_path_with_actual_dash(virtual_xs, virtual_ys, dash_xs, dash_ys):
    if (virtual_xs is None or virtual_ys is None or virtual_xs.size < 2
            or dash_xs is None or dash_ys is None
            or dash_xs.size < CENTER_DASH_MIN_POINTS):
        return virtual_xs, virtual_ys, False

    vx = virtual_xs.astype(np.float64)
    vy = virtual_ys.astype(np.float64)
    dx = np.asarray(dash_xs, dtype=np.float64)
    dy = np.asarray(dash_ys, dtype=np.float64)

    valid = (np.isfinite(dx) & np.isfinite(dy)
             & (dx >= float(vx[0])) & (dx <= float(vx[-1])))
    dx = dx[valid]
    dy = dy[valid]
    if dx.size < CENTER_DASH_MIN_POINTS:
        return virtual_xs, virtual_ys, False

    bin_ids = np.floor((dx - LANE_MODEL_X_MIN) / PATH_BIN_STEP_M).astype(np.int32)
    used_x = []
    used_y = []

    for bid in np.unique(bin_ids):
        mask = bin_ids == bid
        if np.count_nonzero(mask) < PATH_BIN_MIN_POINTS:
            continue
        mx = float(np.median(dx[mask]))
        my = float(np.median(dy[mask]))
        idx = int(np.argmin(np.abs(vx - mx)))
        if abs(float(vx[idx]) - mx) > PATH_BIN_STEP_M:
            continue
        used_x.append(float(vx[idx]))
        used_y.append(my)

    if len(used_x) < PATH_MIN_SAMPLED_POINTS:
        return virtual_xs, virtual_ys, False

    used_x = np.asarray(used_x, dtype=np.float64)
    used_y = np.asarray(used_y, dtype=np.float64)
    order = np.argsort(used_x)
    used_x = used_x[order]
    used_y = used_y[order]
    used_y = _smooth_series(used_y, min(PATH_SMOOTH_WINDOW, 3))

    # 실제 중앙 점선이 보이는 구간은 그 선을 그대로 경로로 쓰고,
    # 보이지 않는 앞뒤 구간만 solid 기반 가상 중앙선으로 남긴다.
    filled = vy.copy()
    in_actual_span = (vx >= float(used_x[0])) & (vx <= float(used_x[-1]))
    filled[in_actual_span] = np.interp(vx[in_actual_span], used_x, used_y)
    filled = _smooth_series(filled, min(PATH_SMOOTH_WINDOW, 3))
    return virtual_xs, filled.astype(np.float32), True


def plan_drive_target(lane_xs, lane_ys, memory: LaneMemory):
    """기본은 오른쪽 실선, 곡선에서는 좌/우 실선 기준 3m 중앙선 path를 만든다.

    1. OpenCV 차선 구름에서 가장 오른쪽/왼쪽 실선을 fit한다.
    2. 직선은 오른쪽 실선에서 half lane width만큼 offset해 차선 중심을 따른다.
    3. 곡선은 오른쪽 실선 +3m 또는 왼쪽 실선 -3m로 도로 중앙선을 강제 추종한다.
    4. 현재 차량 원점에서 path 위 가장 가까운 점으로 reanchor하고 앞쪽 preview target을 잡는다.
    5. 검출이 끊긴 동안은 마지막으로 미리 봐둔 path에서 같은 reanchor를 수행한다.
    """
    memory.age()

    live_lane = (lane_xs is not None and lane_ys is not None
                 and np.asarray(lane_xs).size >= LANE_SOURCE_MIN_POINTS
                 and np.asarray(lane_ys).size >= LANE_SOURCE_MIN_POINTS)
    if not live_lane:
        return _target_from_memory_path(memory, "NO LIVE LANE")

    right_xs, right_ys, right_coef = _fit_rightmost_lane(lane_xs, lane_ys, memory)
    right_ok = _looks_like_outer_solid(right_coef, "right", right_xs, right_ys)
    right_mode = "right-solid"
    if not right_ok:
        if memory.fresh("right") and _looks_like_outer_solid(memory.right_coef, "right"):
            right_xs = np.array([], dtype=np.float64)
            right_ys = np.array([], dtype=np.float64)
            right_coef = memory.right_coef
            right_mode = "right-solid-memory"
        elif right_coef is not None and right_xs.size >= LANE_SIDE_MIN_POINTS:
            # 검증이 빡세게 실패해도 현재 보이는 오른쪽 최외곽 선을 오른쪽 실선으로 고정 추종한다.
            right_mode = "right-solid-relaxed"
        else:
            right_coef = None

    left_xs, left_ys, left_coef = _fit_leftmost_lane(lane_xs, lane_ys, memory)
    left_ok = _looks_like_outer_solid(left_coef, "left", left_xs, left_ys)
    left_mode = "left-solid"
    if not left_ok:
        if memory.fresh("left") and _looks_like_outer_solid(memory.left_coef, "left"):
            left_xs = np.array([], dtype=np.float64)
            left_ys = np.array([], dtype=np.float64)
            left_coef = memory.left_coef
            left_mode = "left-solid-memory"
        elif left_coef is not None and left_xs.size >= LANE_SIDE_MIN_POINTS:
            left_mode = "left-solid-relaxed"
        else:
            left_coef = None

    if right_coef is None and left_coef is None:
        return _target_from_memory_path(memory, "NO OUTER SOLID")

    visible_xmax = _visible_path_xmax(lane_xs)

    path_cand = None
    mode_stem = right_mode
    if right_coef is not None:
        path_cand = _make_center_candidate(
            right_xs, right_ys, right_coef, lane_xs, lane_ys,
            visible_xmax, RIGHT_SOLID_FOLLOW_OFFSET, right_mode)

    probe_path_kappa = (
        _path_preview_curve_mag(path_cand["path_xs"], path_cand["path_ys"])
        if path_cand is not None else 0.0
    )
    probe_turn_score = (
        _path_turn_score(path_cand["path_xs"], path_cand["path_ys"])
        if path_cand is not None else 0.0
    )

    right_center = None
    left_center = None
    if right_coef is not None:
        right_center = _make_center_candidate(
            right_xs, right_ys, right_coef, lane_xs, lane_ys,
            visible_xmax, CENTER_DASH_SOLID_OFFSET, "right")
    if left_coef is not None:
        left_center = _make_center_candidate(
            left_xs, left_ys, left_coef, lane_xs, lane_ys,
            visible_xmax, -CENTER_DASH_SOLID_OFFSET, "left")

    center_path_kappa = max(
        _path_preview_curve_mag(right_center["path_xs"], right_center["path_ys"])
        if right_center is not None else 0.0,
        _path_preview_curve_mag(left_center["path_xs"], left_center["path_ys"])
        if left_center is not None else 0.0,
    )
    center_turn_score = max(
        _path_turn_score(right_center["path_xs"], right_center["path_ys"])
        if right_center is not None else 0.0,
        _path_turn_score(left_center["path_xs"], left_center["path_ys"])
        if left_center is not None else 0.0,
    )
    lane_curve = max(_curve_mag(right_coef), _curve_mag(left_coef),
                     probe_path_kappa, center_path_kappa)
    turn_score = max(probe_turn_score, center_turn_score)
    curve_enter = (
        lane_curve >= CURVE_ENTER_KAPPA
        or turn_score >= CURVE_ENTER_TURN_SCORE
    )
    curve_still_visible = (
        lane_curve >= CURVE_EXIT_KAPPA
        or turn_score >= CURVE_EXIT_TURN_SCORE
    )
    if curve_enter:
        memory.curve_hold = CURVE_HOLD_FRAMES
    elif not curve_still_visible and memory.curve_hold <= 0:
        memory.curve_hold = 0
    curve_section = curve_enter or memory.curve_hold > 0

    if curve_section:
        center_cand, center_side = _merge_center_candidates(right_center, left_center)
        if center_cand is not None:
            path_cand = center_cand
            mode_stem = "curve-center-" + center_side
        elif path_cand is None:
            return _target_from_memory_path(memory, "NO CURVE CENTER PATH")

    if path_cand is None:
        return _target_from_memory_path(memory, "NO RIGHT PATH")

    path_xs = path_cand["path_xs"]
    path_ys = path_cand["path_ys"]
    path_source = path_cand["source"]
    base_coef = path_cand["coef"]

    path_kappa = _path_preview_curve_mag(path_xs, path_ys)
    kappa = max(_curve_mag(base_coef), path_kappa)

    # 현재 차량 위치에서 path 위 가장 가까운 점으로 다시 붙고, 커브가 보이면 더 앞을 본다.
    anchor_idx, anchor_dist = _nearest_path_index(path_xs, path_ys, 0.0, 0.0)
    anchor_lateral_dist = (
        abs(float(path_ys[anchor_idx]))
        if path_ys is not None and path_ys.size > anchor_idx else anchor_dist
    )
    base_L = _lookahead_for_curve(base_coef)
    L = _curve_preview_lookahead(base_L, path_kappa)
    if mode_stem.startswith("curve-center"):
        L += CURVE_PREVIEW_EXTRA_MAX_M * float(np.clip(_path_turn_score(path_xs, path_ys), 0.0, 1.0))
    L += _corner_preturn_extra(path_xs, path_ys, anchor_idx)
    target_x, target_y = _target_at_lookahead_from_index(
        path_xs, path_ys, anchor_idx, L)
    if target_x is None:
        return _target_from_memory_path(memory, "NO RIGHT TARGET")

    target_x = max(TARGET_X_MIN, float(target_x))
    target_y = float(target_y)

    if mode_stem.startswith("curve-center"):
        right_y_at_target = (
            float(np.polyval(_pad_coef(right_coef), target_x))
            if right_coef is not None else None
        )
        left_y_at_target = (
            float(np.polyval(_pad_coef(left_coef), target_x))
            if left_coef is not None else None
        )
        if (left_y_at_target is not None and right_y_at_target is not None
                and left_y_at_target > right_y_at_target):
            lane_bounds = (left_y_at_target, right_y_at_target)
        elif right_y_at_target is not None:
            lane_bounds = (
                right_y_at_target + 2.0 * CENTER_DASH_SOLID_OFFSET,
                right_y_at_target,
            )
        elif left_y_at_target is not None:
            lane_bounds = (
                left_y_at_target,
                left_y_at_target - 2.0 * CENTER_DASH_SOLID_OFFSET,
            )
        else:
            lane_bounds = (
                float(target_y + CENTER_DASH_SOLID_OFFSET),
                float(target_y - CENTER_DASH_SOLID_OFFSET),
            )
    elif right_coef is not None:
        right_y_at_target = float(np.polyval(_pad_coef(right_coef), target_x))
        lane_bounds = (
            right_y_at_target + max(LANE_WIDTH_SINGLE, RIGHT_SOLID_FOLLOW_OFFSET * 2.0),
            right_y_at_target,
        )
    else:
        lane_bounds = (
            float(target_y + LANE_FOLLOW_OFFSET),
            float(target_y - LANE_FOLLOW_OFFSET),
        )

    mode = mode_stem + "-" + path_source
    if anchor_lateral_dist > PATH_REANCHOR_DISTANCE_M:
        mode += "-reanchor"
    if L > base_L + 0.1:
        mode += "-preview"

    if right_coef is not None:
        memory.right_coef = _ema_coef(memory.right_coef, right_coef)
        memory.right_age = 0
    if left_coef is not None:
        memory.left_coef = _ema_coef(memory.left_coef, left_coef)
        memory.left_age = 0
    if mode_stem.startswith("curve-center"):
        center_coef = _fit_lane_side(
            np.asarray(path_xs, dtype=np.float64),
            np.asarray(path_ys, dtype=np.float64),
        )
        if center_coef is not None:
            memory.center_coef = _ema_coef(memory.center_coef, center_coef)
            memory.center_age = 0
        memory.followed_side = "center"
        memory.last_curve_path = (
            np.asarray(path_xs, dtype=np.float32),
            np.asarray(path_ys, dtype=np.float32),
        )
        memory.last_curve_age = 0
        memory.last_curve_side = mode_stem.rsplit("-", 1)[-1]
    elif right_coef is not None:
        memory.followed_side = "right"

    memory.last_target = (float(target_x), float(target_y))
    memory.last_path = (
        np.asarray(path_xs, dtype=np.float32),
        np.asarray(path_ys, dtype=np.float32),
    )
    memory.last_path_age = 0

    return DriveTarget(
        ok=True,
        target_x=float(target_x),
        target_y=float(target_y),
        kappa=float(kappa),
        path_xs=path_xs,
        path_ys=path_ys,
        lane_bounds=lane_bounds,
        mode=mode,
    )


# ============================================================
# legacy 단일 target 제어식 — 현재 motion_node는 /motion/path 다점 추종을 사용
# ============================================================

def _shrink_to_center_band(target_y, band):
    """중앙 근처 조향을 낮추되 0으로 죽이지 않는 soft band.

    하드 데드밴드는 라인이 보여도 차가 늦게 따라붙는 원인이 된다.
    작은 오차는 낮은 gain으로, 큰 오차는 거의 원래 오차에 가깝게 보낸다.
    """
    target_y = float(target_y)
    band = max(0.0, float(band))
    abs_y = abs(target_y)
    if band <= 1e-6:
        return target_y
    if abs_y <= band:
        return target_y * CENTER_LOW_GAIN_SCALE
    softened = (abs_y - band) + band * CENTER_LOW_GAIN_SCALE
    return math.copysign(softened, target_y)


def compute_steer_command(target_x, target_y, kappa=0.0, lane_bounds=None):
    """project _finish_drive_command의 게인 부분 + 두꺼운 중앙선 deadband."""
    target_x = max(TARGET_X_MIN, float(target_x))
    target_y = float(target_y)
    kappa = max(0.0, float(kappa))

    # 중앙선을 두껍게 — 작은 이탈에는 조향 안 함 (떨림 방지)
    effective_y = _shrink_to_center_band(target_y, CENTER_BAND_M)

    gain = (STEER_GAIN
            * (1.0 + min(CURVE_GAIN_MAX, kappa * CURVE_GAIN_SCALE))
            * center_gain(effective_y, kappa)
            * boundary_gain(target_y, lane_bounds))
    angle = gain * math.degrees(math.atan2(-effective_y, target_x))
    if abs(target_y) < STRAIGHT_TY_DEADBAND and kappa < STRAIGHT_KAPPA_DEADBAND:
        angle = 0.0
    angle = float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))
    speed = BASE_SPEED * max(0.30, 1.0 - abs(angle) / 80.0 - min(0.25, kappa * 3.0))
    return float(angle), float(speed)
