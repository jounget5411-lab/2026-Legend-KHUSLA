#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================================================
# 국민대 자율주행 - 통합 + 자동주행 노드 (가독성 재구성판)
#   카메라(front) + 라이다(/scan) + IMU + tf_static 을 받아서
#   1) Ground Homography 로 카메라 픽셀 -> 차량기준 미터 좌표(BEV) 변환
#   2) 차선 마스크 추출 + 라이다 장애물 클러스터링
#   3) 차선/장애물 기반 조향각·속도 계산 (path planner)
#   4) /xycar_motor 발행 + matplotlib 통합 뷰어 표시
#   * 좌표계: X = 전방+, Y = 좌+  (lidar_frame, ROS 표준)
#=============================================================================

import math
import threading
import os
import time
from dataclasses import dataclass

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, Imu, CameraInfo
from xycar_msgs.msg import XycarMotor
from tf2_msgs.msg import TFMessage
import tf2_ros

import matplotlib
matplotlib.use("TkAgg")
# matplotlib 기본 단축키 비활성화 (w/a/s/d 등이 그래프 조작에 먹히지 않도록)
for _k in ("save", "quit", "quit_all", "fullscreen", "home", "back", "forward",
           "pan", "zoom", "grid", "grid_minor", "yscale", "xscale", "copy",
           "help"):
    if f"keymap.{_k}" in matplotlib.rcParams:
        matplotlib.rcParams[f"keymap.{_k}"] = []
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


#=============================================================================
# 1. 토픽 이름 / 프레임 이름 상수
#=============================================================================
CAMERA_TOPICS = {
    "front":  "/usb_cam/image_raw/front",
    "left":   "/usb_cam/image_raw/left",
    "right":  "/usb_cam/image_raw/right",
    "behind": "/usb_cam/image_raw/behind",
}
CAM_INFO_FRONT = "/usb_cam/camera_info/front"
SCAN_TOPIC = "/scan"
IMU_TOPIC = "/imu"
XYCAR_TOPIC = "/xycar_motor"
LIDAR_FRAME = "lidar_frame"
CAM_FRONT_FRAME = "usb_cam_front"


#=============================================================================
# 2. Homography(BEV 변환) 관련 상수
#   - 라이다는 지면에서 0.228m 위에 장착 -> 지면 평면은 lidar_frame 기준 z=-0.228
#   - BEV 격자: 해상도/범위 정의. M_GRID 는 (미터 -> BEV 이미지 픽셀) 변환 행렬
#=============================================================================
LIDAR_HEIGHT_ABOVE_GROUND_M = 0.228
GROUND_Z_IN_LIDAR = -LIDAR_HEIGHT_ABOVE_GROUND_M

BEV_RES   = 0.05                      # 1픽셀 = 0.05m
BEV_X_MIN, BEV_X_MAX = -2.0, 12.0     # 전방 범위 (m)
BEV_Y_HALF           = 5.0            # 좌우 ±5m
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))

# 미터(X=전방, Y=좌) -> BEV 이미지 픽셀(가로 us, 세로 vs) 변환 행렬
M_GRID = np.array([
    [ 0.0,         -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,         BEV_X_MAX  / BEV_RES],
    [ 0.0,           0.0,          1.0],
], dtype=np.float64)

# 카메라 컨벤션 판별용: robot(x=전방) -> optical(z=전방) 회전
R_ROBOT_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


#=============================================================================
# 3. 차선 검출(HSV) 상수
#=============================================================================
LANE_WHITE_V_MIN  = 200    # 흰색: 밝기 하한
LANE_WHITE_S_MAX  = 50     # 흰색: 채도 상한
LANE_YELLOW_H_MIN = 18     # 노란색: 색상 범위
LANE_YELLOW_H_MAX = 38
LANE_YELLOW_S_MIN = 80
LANE_YELLOW_V_MIN = 130
LANE_PIX_SUBSAMPLE = 3     # 차선 픽셀 1/3만 사용 (과밀 방지)
LANE_RUN_MIN_PX = 2        # 한 행에서 차선으로 인정할 연속폭 최소
LANE_RUN_MAX_PX = 35       # 〃 최대 (너무 넓으면 노면/배경)
LANE_CC_MIN_AREA = 6       # 연결성분 최소 면적
LANE_CC_MIN_H = 3          # 연결성분 최소 높이


#=============================================================================
# 4. 라이다 장애물 클러스터링 상수
#=============================================================================
OBSTACLE_X_MIN, OBSTACLE_X_MAX = 0.1, 10.0   # 전방 거리 ROI
OBSTACLE_Y_HALF = 4.5                        # 좌우 ROI
CLUSTER_GAP = 0.25          # 점 간격 이보다 크면 다른 클러스터
CLUSTER_MIN_PTS = 3         # 클러스터 최소 점 개수
OBSTACLE_DRAW_PADDING = 0.05


#=============================================================================
# 5. 주행(path planner) 상수
#=============================================================================
ANGLE_MIN, ANGLE_MAX = -100.0, 100.0
ANIM_INTERVAL_MS = 100

LANE_HALF_WIDTH   = 1.5     # 차선 절반폭 가정 (한쪽만 보일 때 반대쪽 추정)
LANE_MODEL_X_MAX  = 9.0     # 차선 모델링 전방 한계
SAFE_GAP_MIN      = 0.3     # 통과 가능 최소 간격
LANE_OBS_MARGIN   = 0.15    # 장애물 회피 여유
LANE_KEEP_MARGIN  = 0.20    # 차선 경계 안쪽 여유
CORRIDOR_X_TOL    = 1.8     # 목표 전방거리 ±이 범위 장애물만 고려
VIRTUAL_WALL_Y    = 3.8     # 차선 전혀 없을 때 가상 벽

LANE_EMA_ALPHA    = 0.55    # 차선 계수 지수평활 비중
LANE_MAX_AGE      = 25      # 차선 계수 유지 최대 프레임
LANE_LOST_GRACE   = 40      # 차선 놓쳐도 직전 값 쓰는 유예 프레임
LANE_PIX_AVOID_OBS_M = 0.35 # 장애물 근처 차선픽셀 제거 반경

# 슬라이딩 윈도우(차선 중심 추적) 상수
SW_BIN_SIZE       = 0.4
SW_INIT_X_LO      = 0.5
SW_INIT_X_HI      = 1.8
SW_SEARCH_Y       = 0.9
SW_MIN_BIN_PTS    = 2
SW_MIN_CENTERS    = 3
LANE_WIDTH_MIN    = 1.4
LANE_WIDTH_MAX    = 6.0

# 곡률(kappa) 구간별 전방주시거리(look-ahead) / 조향게인
CURV_TIGHT   = 0.07
CURV_MID     = 0.035
CURV_LIGHT   = 0.012
LA_STRAIGHT  = 3.2
LA_LIGHT     = 2.6
LA_MID       = 2.0
LA_TIGHT     = 1.4
GAIN_STRAIGHT = 1.0
GAIN_LIGHT    = 1.35
GAIN_MID      = 1.85
GAIN_TIGHT    = 2.60
STEER_GAIN_BASE = 4.5

STRAIGHT_TY_DEADBAND     = 0.06   # 거의 직진이면 조향 0 처리
STRAIGHT_KAPPA_DEADBAND  = 0.005


#=============================================================================
# 6. 주행 모듈이 프레임 간 기억하는 전역 상태
#   (차선 계수 평활 / 차선 놓쳤을 때 직전값 유지)
#=============================================================================
_LANE_STATE = {"center_coef": None, "age": 999}
_LANE_MEMORY = {
    "last_ly": None, "last_ry": None,
    "lost_frames": 999,
    "ever_seen": False,
}


#=============================================================================
# 7. 공유 상태 컨테이너
#   ROS 콜백(쓰기)과 뷰어/주행(읽기)이 같이 접근하는 최신 데이터 보관소
#=============================================================================
@dataclass
class Latest:
    cam_front: np.ndarray = None
    cam_left: np.ndarray = None
    cam_right: np.ndarray = None
    cam_behind: np.ndarray = None
    scan_ranges: np.ndarray = None
    scan_angle_min: float = 0.0
    scan_angle_inc: float = 0.0
    scan_range_max: float = 100.0
    imu_rpy: tuple = (0.0, 0.0, 0.0)
    K: np.ndarray = None              # 카메라 내부행렬
    H_pix2bev: np.ndarray = None      # 픽셀 -> BEV 이미지픽셀 변환
    cam_height_m: float = float("nan")
    convention: str = "?"
    lane_xs: np.ndarray = None        # 차선 점 (전방 X, m)
    lane_ys: np.ndarray = None        # 차선 점 (좌 Y, m)
    obstacles: list = None            # [(cx, cy, r), ...]
    auto_target: tuple = None         # 주행 목표점 (tx, ty)
    auto_mode_label: str = ""


#=============================================================================
#=============================================================================
#   기하 유틸 함수들 (쿼터니언/회전행렬)
#=============================================================================
#=============================================================================

#=============================================================================
# 쿼터니언 -> 오일러각(roll, pitch, yaw)  (IMU 표시용)
#=============================================================================
def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi/2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


#=============================================================================
# 쿼터니언 -> 3x3 회전행렬  (tf 의 카메라 자세 -> Homography 계산용)
#=============================================================================
def quat_to_R(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


#=============================================================================
#=============================================================================
#   지각(Perception) 함수들 : 카메라 차선 추출 / 라이다 클러스터링
#=============================================================================
#=============================================================================

#=============================================================================
# 차선 마스크 추출
#   BEV 이미지 -> 흰/노란 픽셀 -> 가로 연속폭 필터 -> 연결성분 필터
#   (얇고 길쭉한 차선만 남기고 노면/배경 덩어리 제거)
#=============================================================================
def extract_lane_mask(bev_img):
    if bev_img is None or bev_img.size == 0:
        return None
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white  = (V >= LANE_WHITE_V_MIN) & (S <= LANE_WHITE_S_MAX)
    yellow = ((H >= LANE_YELLOW_H_MIN) & (H <= LANE_YELLOW_H_MAX)
              & (S >= LANE_YELLOW_S_MIN) & (V >= LANE_YELLOW_V_MIN))
    valid = V > 5
    mask = ((white | yellow) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 각 행에서 연속폭이 차선 두께 범위인 구간만 남김
    thin = np.zeros_like(mask)
    for v in range(mask.shape[0]):
        cols = np.flatnonzero(mask[v] > 0)
        if cols.size == 0:
            continue
        cuts = np.where(np.diff(cols) > 1)[0] + 1
        for run in np.split(cols, cuts):
            w = run.size
            if LANE_RUN_MIN_PX <= w <= LANE_RUN_MAX_PX:
                thin[v, run] = 255

    # 연결성분 중 너무 작거나 짧은 것 제거
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    filtered = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= LANE_CC_MIN_AREA and h >= LANE_CC_MIN_H:
            filtered[labels == i] = 255
    return filtered


#=============================================================================
# 차선 마스크(BEV 이미지픽셀) -> 차량기준 미터 좌표 (전방 X, 좌 Y)
#   us(가로),vs(세로) 픽셀을 BEV 정의 역산으로 미터로 환산
#=============================================================================
def lane_mask_to_lidar_xy(mask):
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    right_m = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    forward_m = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return forward_m, -right_m     # (X=전방, Y=좌)


#=============================================================================
# 라이다 점(xs, ys) -> 장애물 클러스터 [(중심x, 중심y, 반지름), ...]
#   인접 점을 거리로 묶고, 점 개수 충족하는 그룹만 채택
#=============================================================================
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


#=============================================================================
#=============================================================================
#   주행(Path Planner) 함수들
#   차선 중심 추적 -> 목표점 -> 장애물 회피 -> 조향각/속도
#=============================================================================
#=============================================================================

#=============================================================================
# 다항식 계수 길이 맞추기 (1차/2차 섞일 때 평활용)
#=============================================================================
def _pad_coef(c, deg=2):
    if c is None: return None
    c = np.asarray(c, dtype=np.float64)
    return c[-(deg+1):] if c.size >= deg+1 else np.pad(c, (deg+1 - c.size, 0))


#=============================================================================
# 차선 중심 계수 지수평활 (이전값과 새값 섞기 -> 떨림 완화)
#=============================================================================
def _ema_coefs(prev, new, alpha=LANE_EMA_ALPHA):
    if prev is None: return new
    if new is None:  return prev
    return alpha * _pad_coef(new) + (1 - alpha) * _pad_coef(prev)


#=============================================================================
# 슬라이딩 윈도우로 차선 중심선 추적
#   가까운 구간에서 좌/우 차선 시작점 잡고, 전방으로 윈도우 옮기며
#   좌/우 차선점 추적 -> 중심선/좌선/우선 다항식 계수 반환
#=============================================================================
def _trace_lane_center(lane_xs, lane_ys):
    if lane_xs is None or lane_xs.size < 8:
        return None, None, None

    init_mask = (lane_xs >= SW_INIT_X_LO) & (lane_xs < SW_INIT_X_HI)
    init_ys = lane_ys[init_mask]
    if init_ys.size < 3:
        return None, None, None

    init_left  = init_ys[init_ys >  0.10]
    init_right = init_ys[init_ys < -0.10]

    if init_left.size >= 2 and init_right.size >= 2:
        prev_left  = float(np.min(init_left))
        prev_right = float(np.max(init_right))
    elif init_left.size >= 2:
        prev_left  = float(np.min(init_left))
        prev_right = prev_left - 2 * LANE_HALF_WIDTH
    elif init_right.size >= 2:
        prev_right = float(np.max(init_right))
        prev_left  = prev_right + 2 * LANE_HALF_WIDTH
    else:
        return None, None, None

    prev_center = (prev_left + prev_right) / 2.0

    centers_x = [(SW_INIT_X_LO + SW_INIT_X_HI) / 2.0]
    centers_y = [prev_center]
    lefts_x  = [centers_x[0]]; lefts_y  = [prev_left]
    rights_x = [centers_x[0]]; rights_y = [prev_right]

    x_cur = SW_INIT_X_HI + SW_BIN_SIZE * 0.5
    while x_cur + SW_BIN_SIZE * 0.5 < LANE_MODEL_X_MAX:
        lo = x_cur - SW_BIN_SIZE * 0.5
        hi = x_cur + SW_BIN_SIZE * 0.5
        mask = (lane_xs >= lo) & (lane_xs < hi) & (np.abs(lane_ys) < BEV_Y_HALF - 0.2)
        ys = lane_ys[mask]
        if ys.size < SW_MIN_BIN_PTS:
            x_cur += SW_BIN_SIZE
            continue

        near_left  = ys[(ys >= prev_center) & (np.abs(ys - prev_left)  < SW_SEARCH_Y)]
        near_right = ys[(ys <= prev_center) & (np.abs(ys - prev_right) < SW_SEARCH_Y)]

        new_left = new_right = None
        if near_left.size >= 1:
            new_left = float(np.median(near_left))
        if near_right.size >= 1:
            new_right = float(np.median(near_right))

        if new_left is not None and new_right is not None:
            w = new_left - new_right
            if not (LANE_WIDTH_MIN <= w <= LANE_WIDTH_MAX):
                if abs(new_left - prev_left) < abs(new_right - prev_right):
                    new_right = new_left - 2 * LANE_HALF_WIDTH
                else:
                    new_left = new_right + 2 * LANE_HALF_WIDTH
            new_center = (new_left + new_right) / 2
            prev_left, prev_right, prev_center = new_left, new_right, new_center
            centers_x.append(x_cur); centers_y.append(new_center)
            lefts_x.append(x_cur);   lefts_y.append(new_left)
            rights_x.append(x_cur);  rights_y.append(new_right)
        elif new_left is not None:
            new_center = new_left - LANE_HALF_WIDTH
            prev_left = new_left
            prev_right = new_left - 2 * LANE_HALF_WIDTH
            prev_center = new_center
            centers_x.append(x_cur); centers_y.append(new_center)
            lefts_x.append(x_cur);   lefts_y.append(new_left)
        elif new_right is not None:
            new_center = new_right + LANE_HALF_WIDTH
            prev_right = new_right
            prev_left = new_right + 2 * LANE_HALF_WIDTH
            prev_center = new_center
            centers_x.append(x_cur); centers_y.append(new_center)
            rights_x.append(x_cur);  rights_y.append(new_right)
        x_cur += SW_BIN_SIZE

    if len(centers_x) < SW_MIN_CENTERS:
        return None, None, None

    cx_arr = np.array(centers_x); cy_arr = np.array(centers_y)
    deg = 2 if len(centers_x) >= 5 else 1
    try:
        center_coef = np.polyfit(cx_arr, cy_arr, deg=deg)
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None

    left_coef = right_coef = None
    if len(lefts_x) >= 3:
        try:
            left_coef = np.polyfit(np.array(lefts_x), np.array(lefts_y),
                                   deg=2 if len(lefts_x) >= 5 else 1)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if len(rights_x) >= 3:
        try:
            right_coef = np.polyfit(np.array(rights_x), np.array(rights_y),
                                    deg=2 if len(rights_x) >= 5 else 1)
        except (np.linalg.LinAlgError, ValueError):
            pass

    return center_coef, left_coef, right_coef


#=============================================================================
# 차선 중심계수 메모리 갱신 (새 값 있으면 평활, 없으면 일정 프레임 유지)
#=============================================================================
def _update_center_memory(center_new):
    s = _LANE_STATE
    if center_new is not None:
        s["center_coef"] = _ema_coefs(s["center_coef"], center_new)
        s["age"] = 0
    else:
        s["age"] += 1
        if s["age"] > LANE_MAX_AGE:
            s["center_coef"] = None
    return s["center_coef"]


#=============================================================================
# 중심선 따라 전방주시거리(look_ahead)에 해당하는 목표점 찾기
#=============================================================================
def _find_lookahead_target(coef, look_ahead, x_max=9.0):
    if coef is None:
        return None
    xs = np.linspace(0.5, x_max, 90)
    for x in xs:
        y = float(np.polyval(coef, x))
        if math.hypot(x, y) >= look_ahead:
            return float(x), y
    x = float(x_max)
    return x, float(np.polyval(coef, x))


#=============================================================================
# 장애물 근처 차선픽셀 제거 (장애물을 차선으로 오인하지 않도록)
#=============================================================================
def _strip_lane_pixels_near_obstacles(lane_xs, lane_ys, obstacles):
    if lane_xs is None or lane_xs.size == 0 or not obstacles:
        return lane_xs, lane_ys
    keep = np.ones(lane_xs.size, dtype=bool)
    for ox, oy, rr in obstacles:
        thresh = rr + LANE_PIX_AVOID_OBS_M
        dx = lane_xs - ox
        dy = lane_ys - oy
        keep &= (dx*dx + dy*dy) > (thresh * thresh)
    return lane_xs[keep], lane_ys[keep]


#=============================================================================
# 목표 전방거리에서 좌/우 차선 경계 안의 가장 넓은 통과 가능 구간 중심 찾기
#   장애물들을 막힌 구간으로 보고, 빈 구간 중 제일 넓은 곳의 중앙 반환
#=============================================================================
def _widest_corridor_at(look_ahead, left_bound, right_bound, obstacles):
    inner_low  = right_bound + LANE_KEEP_MARGIN
    inner_high = left_bound  - LANE_KEEP_MARGIN
    if inner_high - inner_low < SAFE_GAP_MIN:
        return None
    obs_in = [(oy, rr) for ox, oy, rr in obstacles
              if abs(ox - look_ahead) < CORRIDOR_X_TOL
              and 0.2 < ox < OBSTACLE_X_MAX]
    if not obs_in:
        return (inner_low + inner_high) / 2
    blocked = sorted([(oy - rr - LANE_OBS_MARGIN, oy + rr + LANE_OBS_MARGIN)
                      for oy, rr in obs_in])
    merged = [blocked[0]]
    for lo, hi in blocked[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    free = []
    prev = inner_low
    for lo, hi in merged:
        if lo > prev:
            free.append((prev, min(lo, inner_high)))
        if hi > prev: prev = hi
        if prev >= inner_high: break
    if prev < inner_high:
        free.append((prev, inner_high))
    free = [(lo, hi) for lo, hi in free if hi - lo >= SAFE_GAP_MIN]
    if not free: return None
    widest = max(free, key=lambda f: f[1] - f[0])
    return (widest[0] + widest[1]) / 2


#=============================================================================
# 목표 전방거리 차선 안에 장애물이 있는지 판정
#=============================================================================
def _any_obstacle_in_lane(obstacles, ly, ry, look_ahead):
    for ox, oy, rr in obstacles:
        if abs(ox - look_ahead) < CORRIDOR_X_TOL and 0.2 < ox < OBSTACLE_X_MAX:
            if (ry - rr * 0.3) < oy < (ly + rr * 0.3):
                return True
    return False


#=============================================================================
# [핵심] 주행 명령 계산
#   입력: 차선점(lane_xs/ys), 장애물(obstacles)
#   출력: (속도, 조향각, 모드라벨, 목표점)
#   흐름: 차선중심 추적 -> 목표점 -> 차선/장애물로 목표 보정 -> 조향/속도
#=============================================================================
def compute_drive_command(lane_xs, lane_ys, obstacles):
    lane_xs, lane_ys = _strip_lane_pixels_near_obstacles(lane_xs, lane_ys, obstacles)

    center_raw, left_coef, right_coef = _trace_lane_center(lane_xs, lane_ys)
    center_coef = _update_center_memory(center_raw)

    # 곡률 kappa 로 전방주시거리 결정 (급커브일수록 가까이 봄)
    kappa = 0.0
    if center_coef is not None and len(center_coef) >= 3:
        kappa = abs(float(center_coef[0]))

    if   kappa > CURV_TIGHT: look_ahead = LA_TIGHT
    elif kappa > CURV_MID:   look_ahead = LA_MID
    elif kappa > CURV_LIGHT: look_ahead = LA_LIGHT
    else:                    look_ahead = LA_STRAIGHT

    def ev(coef, xv):
        return None if coef is None else float(np.polyval(coef, xv))

    mem = _LANE_MEMORY
    have_lane = False
    target_x = target_y = None
    mode = "no perception"
    ly = ry = None

    # ---- 차선 있음: 중심선에서 목표점 + 좌우 경계 산출 ----
    if center_coef is not None:
        target = _find_lookahead_target(center_coef, look_ahead)
        if target is not None:
            target_x, target_y = target

        ly_eval = ev(left_coef,  target_x) if left_coef  is not None else None
        ry_eval = ev(right_coef, target_x) if right_coef is not None else None

        if ly_eval is not None and ry_eval is not None:
            ly, ry = ly_eval, ry_eval
            mode = "lane both"
        elif ly_eval is not None:
            ly = ly_eval
            ry = ly - 2 * LANE_HALF_WIDTH
            mode = "lane left"
        elif ry_eval is not None:
            ry = ry_eval
            ly = ry + 2 * LANE_HALF_WIDTH
            mode = "lane right"
        else:
            ly = target_y + LANE_HALF_WIDTH
            ry = target_y - LANE_HALF_WIDTH
            mode = "lane center"

        mem["last_ly"] = ly; mem["last_ry"] = ry
        mem["lost_frames"] = 0; mem["ever_seen"] = True
        have_lane = True
    # ---- 차선 없음: 직전 차선 잠깐 유지하거나, 가상 벽으로 대체 ----
    else:
        mem["lost_frames"] += 1
        if mem["ever_seen"] and mem["lost_frames"] <= LANE_LOST_GRACE \
                and mem["last_ly"] is not None:
            ly = mem["last_ly"]; ry = mem["last_ry"]
            target_x = LA_STRAIGHT
            target_y = (ly + ry) / 2
            mode = f"lane memory ({mem['lost_frames']})"
            have_lane = True
        else:
            ly, ry = +VIRTUAL_WALL_Y, -VIRTUAL_WALL_Y
            target_x = LA_STRAIGHT
            target_y = 0.0
            mode = "obs-as-lane" if obstacles else "no perception"
            have_lane = False

    # ---- 차선 안에 장애물 있으면 가장 넓은 통로로 목표 보정 ----
    if have_lane and _any_obstacle_in_lane(obstacles, ly, ry, target_x):
        corr_y = _widest_corridor_at(target_x, ly, ry, obstacles)
        if corr_y is None:
            found = False
            for la2 in (target_x * 0.7, target_x * 0.5, 1.4):
                t = _widest_corridor_at(la2, ly, ry, obstacles)
                if t is not None:
                    target_y, target_x = t, la2
                    found = True
                    break
            if not found:
                return 0.0, 0.0, "BLOCKED", (1.5, 0.0)
        else:
            target_y = corr_y
        mode += " | avoid"
    elif not have_lane:
        corr_y = _widest_corridor_at(target_x, ly, ry, obstacles)
        if corr_y is None:
            found = False
            for la2 in (target_x * 0.7, target_x * 0.5, 1.4):
                t = _widest_corridor_at(la2, ly, ry, obstacles)
                if t is not None:
                    target_y, target_x = t, la2
                    found = True
                    break
            if not found:
                return 0.0, 0.0, "BLOCKED", (1.5, 0.0)
        else:
            target_y = corr_y

    # 목표 Y를 차선 경계 안쪽으로 제한
    target_y = float(np.clip(target_y, ry + LANE_KEEP_MARGIN, ly - LANE_KEEP_MARGIN))

    # 코앞 정면 장애물이면 급정거
    for ox, oy, _rr in obstacles:
        if 0.05 < ox < 1.0 and abs(oy) < 0.5:
            return 0.0, 0.0, mode + " BRAKE", (target_x, target_y)

    # 곡률 구간별 조향 게인
    if   kappa > CURV_TIGHT: gain_mult = GAIN_TIGHT
    elif kappa > CURV_MID:   gain_mult = GAIN_MID
    elif kappa > CURV_LIGHT: gain_mult = GAIN_LIGHT
    else:                    gain_mult = GAIN_STRAIGHT

    # 목표점 방향 -> 조향각 (Pure-Pursuit 류)
    raw_deg = math.degrees(math.atan2(-target_y, max(target_x, 0.5)))
    angle = STEER_GAIN_BASE * gain_mult * raw_deg

    # 거의 직진이면 미세조향 죽이기
    if have_lane and "avoid" not in mode and "memory" not in mode:
        if abs(target_y) < STRAIGHT_TY_DEADBAND and kappa < STRAIGHT_KAPPA_DEADBAND:
            angle = 0.0

    angle = float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    # 조향클수록/급커브일수록 감속
    steer_factor = abs(angle) / 50.0
    curve_factor = min(1.0, kappa * 6.0)
    speed_mult = max(0.30,
                     1.0
                     - 0.35 * min(1.0, steer_factor)
                     - 0.45 * curve_factor)
    speed = 10.0 * speed_mult

    return speed, angle, mode, (target_x, target_y)


#=============================================================================
#=============================================================================
#   ROS2 노드 : 토픽 구독/발행 + Homography 빌드 + 주행명령 발행
#=============================================================================
#=============================================================================
class TestViewerNode(Node):

    #=============================================================================
    # 생성자: 구독/발행 설정, tf 리스너, 타이머 등록
    #=============================================================================
    def __init__(self, latest: Latest):
        super().__init__("kookmin9_viewer")
        self.latest = latest
        self._lock = threading.Lock()

        # 카메라 4개 구독
        for slot, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, slot=slot: self._on_image(slot, msg),
                qos_profile_sensor_data,
            )
        # 카메라 내부행렬 / 라이다 / IMU 구독
        self.create_subscription(CameraInfo, CAM_INFO_FRONT, self._on_caminfo,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, IMU_TOPIC, self._on_imu,
                                 qos_profile_sensor_data)

        # 모터 발행
        self._motor_pub = self.create_publisher(XycarMotor, XYCAR_TOPIC, 10)

        # tf_static 구독 (카메라<->라이다 자세) - latched 성격이라 RELIABLE
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        tf_static_qos = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            TFMessage, '/tf_static', self._on_tf_static, tf_static_qos,
        )

        # 타이머: 0.1초마다 주행명령 발행 / 0.5초마다 Homography 빌드 시도
        self.target_speed = 0.0
        self.target_angle = 0.0
        self.create_timer(0.1, self._publish_motor)

        self._H_built = False
        self.create_timer(0.5, self._maybe_build_homography)

        self.get_logger().info("kookmin9_viewer 시작.")

    #-------------------------------------------------------------------------
    # 콜백: 들어온 센서 데이터를 latest 에 저장만 (가공은 뷰어/주행에서)
    #-------------------------------------------------------------------------
    def _on_image(self, slot, msg):
        if msg.encoding != "rgb8":
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except ValueError:
            return
        with self._lock:
            setattr(self.latest, f"cam_{slot}", arr)

    def _on_caminfo(self, msg):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self._lock:
            self.latest.K = K

    def _on_scan(self, msg):
        with self._lock:
            self.latest.scan_ranges = np.asarray(msg.ranges, dtype=np.float32)
            self.latest.scan_angle_min = msg.angle_min
            self.latest.scan_angle_inc = msg.angle_increment
            self.latest.scan_range_max = msg.range_max

    def _on_imu(self, msg):
        q = msg.orientation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        with self._lock:
            self.latest.imu_rpy = (roll, pitch, yaw)

    def _on_tf_static(self, msg):
        for tr in msg.transforms:
            self.tf_buffer.set_transform_static(tr, "kookmin9_viewer")

    #-------------------------------------------------------------------------
    # Homography 빌드 (한 번만)
    #   tf(카메라<->라이다) + K 로 픽셀->지면 미터 변환행렬 계산
    #   지면 평면을 라이다높이(0.228m)만큼 내려서 = Ground H
    #   optical/robot 두 컨벤션 시도 후 합리적인 쪽 자동 채택
    #-------------------------------------------------------------------------
    def _maybe_build_homography(self):
        if self._H_built:
            return
        with self._lock:
            K = self.latest.K
        if K is None:
            return
        try:
            tr = self.tf_buffer.lookup_transform(
                LIDAR_FRAME, CAM_FRONT_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        t = np.array([tr.transform.translation.x,
                      tr.transform.translation.y,
                      tr.transform.translation.z], dtype=np.float64)
        q = tr.transform.rotation
        R_raw = quat_to_R(q.x, q.y, q.z, q.w)

        # 지면까지 카메라 높이 + 지면평면 스케일
        cam_ground_height = float(t[2] - GROUND_Z_IN_LIDAR)
        if abs(cam_ground_height) < 1e-3:
            self.get_logger().warn("카메라 z≈ground. BEV 비활성.")
            return
        plane_scale = GROUND_Z_IN_LIDAR - float(t[2])

        T_homog = np.array([
            [plane_scale, 0.0, t[0]],
            [0.0, plane_scale, t[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        # 화면 중앙-하단 픽셀이 전방으로 매핑되는 컨벤션을 채택
        u0 = K[0, 2]
        v0 = K[1, 2] + K[1, 1] * 0.30
        test_pix = np.array([u0, v0, 1.0])

        candidates = [("optical", R_raw),
                      ("robot",   R_raw @ R_ROBOT_TO_OPTICAL)]
        best = None
        for label, R_eff in candidates:
            H_pl = T_homog @ R_eff @ K_inv
            p = H_pl @ test_pix
            if p[2] == 0:
                continue
            X_chk = p[0] / p[2]; Y_chk = p[1] / p[2]
            score = X_chk - 2.0 * abs(Y_chk)
            self.get_logger().info(
                f"BEV 시도 [{label}]: 중앙-하단 → "
                f"(X={X_chk:+.2f}, Y={Y_chk:+.2f}) m   score={score:+.2f}"
            )
            if 0.5 < X_chk < 20 and abs(Y_chk) < 2.0:
                if best is None or score > best[0]:
                    best = (score, label, H_pl, X_chk, Y_chk)

        if best is None:
            self.get_logger().warn("어떤 컨벤션도 합리적 BEV 안 나옴.")
            return

        _, label, H_pix2lidar, X_chk, Y_chk = best
        # 미터 -> BEV 이미지픽셀 까지 합쳐 최종 warp 행렬로 저장
        H_pix2bev = M_GRID @ H_pix2lidar

        with self._lock:
            self.latest.H_pix2bev = H_pix2bev
            self.latest.cam_height_m = cam_ground_height
            self.latest.convention = label
        self._H_built = True
        self.get_logger().info(
            f"BEV 호모그래피 빌드 완료 [{label}]. cam_h={cam_ground_height:+.3f} m"
        )

    #-------------------------------------------------------------------------
    # 주행명령 계산 (지각결과 -> compute_drive_command)
    #-------------------------------------------------------------------------
    def _compute_auto_drive(self):
        with self._lock:
            lane_xs = self.latest.lane_xs
            lane_ys = self.latest.lane_ys
            obstacles = list(self.latest.obstacles or [])
        if lane_xs is None:
            lane_xs = np.array([])
            lane_ys = np.array([])
        speed, angle, mode, target = compute_drive_command(
            lane_xs, lane_ys, obstacles)
        self.target_speed = speed
        self.target_angle = angle
        with self._lock:
            self.latest.auto_target = target
            self.latest.auto_mode_label = mode

    #-------------------------------------------------------------------------
    # 0.1초 타이머: 주행명령 계산 후 /xycar_motor 발행
    #-------------------------------------------------------------------------
    def _publish_motor(self):
        self._compute_auto_drive()
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self.target_speed)
        msg.angle = float(self.target_angle)
        self._motor_pub.publish(msg)


#=============================================================================
#=============================================================================
#   matplotlib 통합 뷰어
#   카메라4 + BEV지도(차선/라이다/장애물/목표) + IMU + 제어 표시
#   * 지각 가공(차선 미터변환, 클러스터링)도 여기 _update 에서 일어남
#=============================================================================
#=============================================================================
class Viewer:

    #=============================================================================
    # 생성자: figure/axes 구성, 애니메이션 타이머 등록
    #=============================================================================
    def __init__(self, node, latest):
        self.node = node
        self.latest = latest
        self._last_bev_rgb = None
        self._last_mask = None

        # --- 메인 figure (카메라4 + BEV + IMU + 제어) ---
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title("kookmin9_viewer")
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[1, 1, 0.8])

        # --- 별도 figure: 차선 마스크만 크게 ---
        self.lane_fig = plt.figure(figsize=(5, 8))
        self.lane_fig.canvas.manager.set_window_title("lane_mask_view")
        self.ax_lane_mask = self.lane_fig.add_subplot(111)
        self.ax_lane_mask.set_title("lane mask only")
        self.ax_lane_mask.set_aspect("equal")
        self.ax_lane_mask.set_xlim(-BEV_Y_HALF, +BEV_Y_HALF)
        self.ax_lane_mask.set_ylim(BEV_X_MIN, BEV_X_MAX)
        self.ax_lane_mask.set_xlabel("lateral [m]  (right +)")
        self.ax_lane_mask.set_ylabel("forward [m]")
        self.lane_mask_im = self.ax_lane_mask.imshow(
            np.zeros((BEV_H, BEV_W), dtype=np.uint8),
            cmap="gray", vmin=0, vmax=255,
            extent=[-BEV_Y_HALF, +BEV_Y_HALF, BEV_X_MIN, BEV_X_MAX],
            origin="upper", interpolation="nearest",
        )
        self._lane_mask_status = self.ax_lane_mask.text(
            0.02, 0.98, "waiting for BEV lane mask",
            transform=self.ax_lane_mask.transAxes, color="yellow", fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2),
        )
        self.lane_fig.canvas.mpl_connect("key_press_event", self._on_key)

        # --- 카메라 4분할 ---
        self.ax_cam = {
            "front":  self.fig.add_subplot(gs[0, 0]),
            "left":   self.fig.add_subplot(gs[0, 1]),
            "right":  self.fig.add_subplot(gs[1, 0]),
            "behind": self.fig.add_subplot(gs[1, 1]),
        }
        self.cam_im = {}
        for slot, ax in self.ax_cam.items():
            ax.set_title(f"camera/{slot}")
            ax.set_xticks([]); ax.set_yticks([])
            self.cam_im[slot] = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        # --- BEV 통합 지도 ---
        self.ax_map = self.fig.add_subplot(gs[0:2, 2:4])
        self.ax_map.set_title(
            "BEV map  —  camera + LiDAR + lanes + obstacles + AUTO target  (lidar_frame, m)"
        )
        self.ax_map.set_aspect("equal")
        self.ax_map.set_xlim(-BEV_Y_HALF, +BEV_Y_HALF)
        self.ax_map.set_ylim(BEV_X_MIN, BEV_X_MAX)
        self.ax_map.set_xlabel("lateral [m]  (right +)")
        self.ax_map.set_ylabel("forward [m]")

        self.bev_im = self.ax_map.imshow(
            np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8),
            extent=[-BEV_Y_HALF, +BEV_Y_HALF, BEV_X_MIN, BEV_X_MAX],
            origin="upper", interpolation="nearest", zorder=1,
        )

        # 거리 가이드 원
        theta = np.linspace(0.0, 2 * math.pi, 128)
        for r in (1.0, 2.0, 3.0, 5.0, 8.0):
            self.ax_map.plot(-r * np.sin(theta), r * np.cos(theta),
                             color="white", alpha=0.22, linewidth=0.7, zorder=2)
            self.ax_map.text(0.05, r + 0.05, f"{int(r)}m",
                             color="white", alpha=0.55, fontsize=7, zorder=2)

        self.lane_pts, = self.ax_map.plot([], [], '.', color="#33ff66",
                                           markersize=2.5, zorder=3, label="lanes")
        self.lidar_pts, = self.ax_map.plot([], [], '.', color="#00ffff",
                                            markersize=2.0, zorder=4, label="/scan")
        self._obstacle_artists = []
        veh = np.array([[0.0, 0.45], [-0.20, -0.25], [0.20, -0.25]])
        self.ax_map.add_patch(plt.Polygon(
            veh, closed=True, color="#4ec3ff", alpha=0.9, zorder=6))

        self._target_line, = self.ax_map.plot(
            [], [], '-', color="yellow", alpha=0.75, linewidth=1.8, zorder=7)
        self._target_marker, = self.ax_map.plot(
            [], [], '*', color="magenta", markersize=18,
            markeredgecolor="black", markeredgewidth=0.5,
            zorder=8, label="AUTO target")

        self._bev_status = self.ax_map.text(
            0.02, 0.98, "BEV: waiting for CameraInfo + /tf_static …",
            transform=self.ax_map.transAxes, color="yellow", fontsize=9,
            verticalalignment="top", zorder=10,
        )
        self._det_status = self.ax_map.text(
            0.02, 0.02, "",
            transform=self.ax_map.transAxes, color="white", fontsize=9,
            verticalalignment="bottom", zorder=10,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2),
        )
        self.ax_map.legend(loc="lower right", fontsize=8, framealpha=0.7)

        # --- IMU 표시 ---
        self.ax_imu = self.fig.add_subplot(gs[2, 0:2])
        self.ax_imu.set_xlim(-1.2, 1.2); self.ax_imu.set_ylim(-1.2, 1.2)
        self.ax_imu.set_aspect("equal")
        self.ax_imu.set_xticks([]); self.ax_imu.set_yticks([])
        self.ax_imu.set_title("/imu — roll/pitch/yaw")
        self._arrows = []
        self._imu_text = self.ax_imu.text(
            -1.15, 1.05, "", fontsize=9, family="monospace",
            verticalalignment="top",
        )

        # --- 제어 상태 텍스트 ---
        self.ax_ctrl = self.fig.add_subplot(gs[2, 2:4])
        self.ax_ctrl.set_xticks([]); self.ax_ctrl.set_yticks([])
        self.ax_ctrl.set_title("/xycar_motor — control")
        self._ctrl_text = self.ax_ctrl.text(
            0.02, 0.95, "", fontsize=10, family="monospace",
            transform=self.ax_ctrl.transAxes, verticalalignment="top",
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(
            self.fig, self._update, interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )

    #-------------------------------------------------------------------------
    # 키 입력: ESC 종료 / M 디버그 덤프
    #-------------------------------------------------------------------------
    def _on_key(self, event):
        k = event.key
        if k == "escape":      plt.close(self.fig); return
        if k in ("m", "M"):    self._dump_debug(); return

    #-------------------------------------------------------------------------
    # 디버그 덤프: 현재 front/bev/mask 를 ~/ 에 PNG 저장
    #-------------------------------------------------------------------------
    def _dump_debug(self):
        ts = time.strftime("%H%M%S")
        out_dir = os.path.expanduser("~")
        saved = []
        front = self.latest.cam_front
        if front is not None:
            p = os.path.join(out_dir, f"front_{ts}.png")
            cv2.imwrite(p, cv2.cvtColor(front, cv2.COLOR_RGB2BGR)); saved.append(p)
        if self._last_bev_rgb is not None:
            p = os.path.join(out_dir, f"bev_{ts}.png")
            cv2.imwrite(p, cv2.cvtColor(self._last_bev_rgb, cv2.COLOR_RGB2BGR)); saved.append(p)
        if self._last_mask is not None:
            p = os.path.join(out_dir, f"mask_{ts}.png")
            cv2.imwrite(p, self._last_mask); saved.append(p)
        print(f"[dump] saved: {saved}")

    #-------------------------------------------------------------------------
    # [핵심] 화면 갱신 (애니메이션 콜백, 0.1초마다)
    #   1) 카메라 표시  2) BEV warp + 차선마스크  3) 차선 미터변환
    #   4) 라이다 점/클러스터  5) latest 에 지각결과 저장(주행이 사용)
    #   6) 목표점/IMU/제어 표시
    #-------------------------------------------------------------------------
    def _update(self, _frame):
        # 1) 카메라 원본 4개
        for slot, im in self.cam_im.items():
            arr = getattr(self.latest, f"cam_{slot}")
            if arr is not None:
                im.set_data(arr)

        # 2) front 를 BEV 로 warp -> 차선마스크 추출
        H = self.latest.H_pix2bev
        front = self.latest.cam_front
        bev_img = None
        mask = None
        if H is not None and front is not None:
            bev_img = cv2.warpPerspective(
                front, H, (BEV_W, BEV_H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask = extract_lane_mask(bev_img)
            disp = bev_img.copy()
            if mask is not None and mask.any():
                disp[mask > 0] = [255, 0, 255]   # 차선 마젠타 강조
            self.bev_im.set_data(disp)
            self._bev_status.set_text(
                f"BEV ON [{self.latest.convention}]  "
                f"(cam_h={self.latest.cam_height_m:+.2f} m)"
            )
            self._last_bev_rgb = bev_img
            self._last_mask = mask
            self.lane_mask_im.set_data(mask if mask is not None else np.zeros((BEV_H, BEV_W), dtype=np.uint8))
        else:
            missing = []
            if self.latest.K is None: missing.append("CameraInfo")
            if self.latest.K is not None and H is None: missing.append("/tf_static")
            if front is None: missing.append("/usb_cam/image_raw/front")
            self._bev_status.set_text(
                "BEV: waiting for " + ", ".join(missing) if missing else "")
            self.lane_mask_im.set_data(np.zeros((BEV_H, BEV_W), dtype=np.uint8))

        # 3) 차선마스크 -> 미터좌표 (전방 lx, 좌 ly)
        if mask is not None:
            lx, ly = lane_mask_to_lidar_xy(mask)
            self.lane_pts.set_data(-ly, lx)
            n_lane = lx.size
        else:
            lx = np.array([]); ly = np.array([])
            self.lane_pts.set_data([], [])
            n_lane = 0
        self._lane_mask_status.set_text(f"lane pixels: {n_lane}")
        self.lane_fig.canvas.draw_idle()

        # 4) 라이다 점 (거리/각도 -> x,y) + 표시
        ranges = self.latest.scan_ranges
        xs = ys = np.array([])
        if ranges is not None and ranges.size > 0:
            n = ranges.size
            angles = self.latest.scan_angle_min + np.arange(n) * self.latest.scan_angle_inc
            valid = (np.isfinite(ranges)
                     & (ranges > 0.05)
                     & (ranges < self.latest.scan_range_max - 1e-3))
            if valid.any():
                xs = ranges[valid] * np.cos(angles[valid])
                ys = ranges[valid] * np.sin(angles[valid])
                self.lidar_pts.set_data(-ys, xs)
            else:
                self.lidar_pts.set_data([], [])

        # 4-2) 라이다 ROI 클러스터링 -> 장애물 원 표시
        for art in self._obstacle_artists:
            art.remove()
        self._obstacle_artists = []
        clusters = []
        if xs.size > 0:
            roi = ((xs >= OBSTACLE_X_MIN) & (xs <= OBSTACLE_X_MAX)
                   & (np.abs(ys) <= OBSTACLE_Y_HALF))
            if roi.any():
                clusters = cluster_scan_obstacles(xs[roi], ys[roi])
                for (cx, cy, r) in clusters:
                    sx, sy = -cy, cx
                    circ = plt.Circle((sx, sy), r, fill=False,
                                       edgecolor="red", linewidth=1.5, zorder=5)
                    self.ax_map.add_patch(circ)
                    self._obstacle_artists.append(circ)
                    txt = self.ax_map.text(
                        sx, sy + r + 0.1, f"{math.hypot(cx, cy):.1f}m",
                        color="red", fontsize=7, ha="center", zorder=5,
                    )
                    self._obstacle_artists.append(txt)

        # 5) 지각결과를 latest 에 저장 -> 노드의 주행 타이머가 사용
        with self.node._lock:
            self.latest.lane_xs = lx if lx.size > 0 else None
            self.latest.lane_ys = ly if ly.size > 0 else None
            self.latest.obstacles = clusters

        # 6) 주행 목표점 표시
        target = self.latest.auto_target
        if target is not None:
            tx, ty = target
            sx, sy = -ty, tx
            self._target_marker.set_data([sx], [sy])
            self._target_line.set_data([0, sx], [0, sy])
        else:
            self._target_marker.set_data([], [])
            self._target_line.set_data([], [])

        mode_txt = f"AUTO [{self.latest.auto_mode_label}]"
        self._det_status.set_text(
            f"{mode_txt}    lanes: {n_lane:4d} pts   obstacles: {len(clusters):2d}"
        )

        # 6-2) IMU 화살표
        roll, pitch, yaw = self.latest.imu_rpy
        for art in self._arrows:
            art.remove()
        self._arrows = []
        L = 0.9
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L*math.cos(roll), L*math.sin(roll),
            color="red", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L*math.cos(pitch+math.pi/2), L*math.sin(pitch+math.pi/2),
            color="green", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L*math.cos(yaw), L*math.sin(yaw),
            color="blue", width=0.02, head_width=0.06, length_includes_head=True))
        self._imu_text.set_text(
            f"roll  (red)   = {math.degrees(roll):+7.2f} deg\n"
            f"pitch (green) = {math.degrees(pitch):+7.2f} deg\n"
            f"yaw   (blue)  = {math.degrees(yaw):+7.2f} deg"
        )

        # 6-3) 제어 상태 텍스트
        self._ctrl_text.set_text(
            f"speed = {self.node.target_speed:+6.2f}    angle = {self.node.target_angle:+6.2f}\n"
            f"AUTO  = ON\n"
            f"\n"
            f"  M       : dump BEV/mask/front to ~/\n"
            f"  ESC     : exit"
        )

    def show(self):
        plt.show()


#=============================================================================
#=============================================================================
#   메인: 노드 생성 -> spin 스레드 -> 뷰어 표시
#=============================================================================
#=============================================================================
def main():
    rclpy.init()
    latest = Latest()
    node = TestViewerNode(latest)

    # ROS 콜백은 별도 스레드에서 spin (뷰어는 메인스레드 GUI 점유)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viewer = Viewer(node, latest)
    try:
        viewer.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()