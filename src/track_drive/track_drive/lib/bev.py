"""BEV 호모그래피 + 어안 undistort + 차선점 추출 (순수 파이썬, rclpy 금지).

역할: 카메라 캘리브(config/camera.yaml) 로드, 170도 어안 undistort,
      이미지 마스크 → BEV warp → lidar 좌표 차선 중심점 추출, 품질 점수.
입력: camera.yaml (K/D/H/calibrated), BGR 이미지, 차선 마스크(uint8, 이미지 픽셀 좌표).
출력: (xs, ys) lidar_frame 미터 (x전방/y좌), lane_quality 0~1.
예선 대비 변경점: lane_detect_node 의 BEV 상수·_mask_to_centerline_xy 를 lib 로
      승격(원문 유지). H 는 yaml 에서 로드하고 미캘리브 시 예선 시뮬 폴백 H 사용.
      Undistorter(fisheye) 신설 — calibrated=False 면 no-op 패스스루.
"""

import os

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # PyYAML 없는 환경(순수 유닛테스트)에서도 import 는 되게
    yaml = None


# ======================== BEV 상수 (예선 common.py 와 동일 유지) ========================

BEV_RES = 0.05                      # [SIM] m/px
BEV_X_MIN, BEV_X_MAX = -2.0, 12.0   # [SIM] 전방 범위 m
BEV_Y_HALF = 5.0                    # [SIM] 좌우 반폭 m
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))          # 200
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))   # 280

# lidar (x전방/y좌, m) → BEV 픽셀 (u,v) 그리드 행렬
M_GRID = np.array([
    [ 0.0,           -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,           BEV_X_MAX  / BEV_RES],
    [ 0.0,            0.0,           1.0],
], dtype=np.float64)

# 예선 시뮬 캘리브 호모그래피 (이미지 픽셀 → lidar 미터). 실차 캘리브 전 폴백.
H_PIX2LIDAR_SIM_FALLBACK = np.array([  # [SIM] common.H_PIX2LIDAR_FALLBACK 복사본
    [ 0.00000000000000e+00,  4.63055528548851e-03, -2.73689017508590e+00],
    [ 4.18097842466549e-03,  1.87301533010043e-04, -1.37819064484720e+00],
    [ 0.00000000000000e+00, -2.40130177039888e-03,  5.16378846754945e-01],
], dtype=np.float64)

LANE_PIX_SUBSAMPLE = 3          # [SIM] BEV 마스크 픽셀 서브샘플 간격
LANE_CENTERLINE_BIN_M = 0.18    # [SIM] 전방 x-bin 폭 (bin당 중심점 1개)

# lane_quality 등급 기준 (본선 스펙: n/20 × x스팬/3.0)
LANE_QUALITY_GOOD_POINTS = 20.0
LANE_QUALITY_GOOD_SPAN_M = 3.0


# ======================== 캘리브 로드 ========================

def load_camera_config(path):
    """camera.yaml → dict(K, D, H, calibrated).

    파일 없음 / 파싱 실패 / calibrated:false 면 예선 시뮬 H 로 폴백하고
    calibrated=False 반환 — 호출측(노드)이 반드시 경고 로그를 남길 것.
    """
    fallback = {
        "K": np.eye(3, dtype=np.float64),
        "D": np.zeros(4, dtype=np.float64),
        "H": H_PIX2LIDAR_SIM_FALLBACK.copy(),
        "calibrated": False,
    }

    if yaml is None or not path or not os.path.isfile(path):
        return fallback

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return fallback
        calibrated = bool(data.get("calibrated", False))
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["D"], dtype=np.float64).reshape(-1)[:4]
        H = np.asarray(data["H"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
        return fallback

    if not calibrated:
        # K/D 자리는 읽되 H 는 검증된 시뮬 폴백을 쓴다 (미캘리브 H 신뢰 금지)
        return fallback

    return {"K": K, "D": D, "H": H, "calibrated": True}


# ======================== 어안 undistort ========================

class Undistorter:
    """170도 어안 undistort. 리맵 테이블 1회 생성 후 캐시.

    calibrated=False 면 no-op 패스스루 (시뮬 폴백 H 는 원본 픽셀 기준이므로
    undistort 없이 그대로 써야 좌표가 맞는다).
    """

    def __init__(self, K, D, size, calibrated=True):
        # size = (width, height) 예: (640, 480)
        self.enabled = False
        self.new_K = None if K is None else np.asarray(K, dtype=np.float64)
        if not calibrated or K is None or D is None:
            return
        try:
            K = np.asarray(K, dtype=np.float64).reshape(3, 3)
            D = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
            # new_K = K 유지 → yaml 의 H(undistort 이미지 기준)와 좌표 일치
            self.new_K = K.copy()
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D.reshape(4, 1), np.eye(3), self.new_K,
                (int(size[0]), int(size[1])), cv2.CV_16SC2)
            self.enabled = True
        except cv2.error:
            # 잘못된 K/D — 패스스루 유지 (호출측이 enabled 확인해 경고 로그)
            self.enabled = False

    def apply(self, bgr):
        if not self.enabled or bgr is None:
            return bgr
        return cv2.remap(bgr, self._map1, self._map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


# ======================== 마스크 → lidar 좌표 차선점 ========================

def _lane_mask_to_lidar_xy(mask):
    """BEV 마스크 픽셀 → lidar_frame (x전방, y좌) 미터. 예선 원문 유지."""
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    right_m = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    forward_m = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return forward_m, -right_m


def _mask_to_centerline_xy(mask):
    """YOLO seg 출력처럼 전방 x-bin마다 차선 중심점 하나만 만든다.

    원 YOLO 노드는 각 mask row에서 u 중심점만 발행했다. OpenCV 마스크 픽셀을
    그대로 넘기면 차선 두께/노이즈가 2차식 fit에 들어가 직선 와리가리가 생긴다.
    (예선 lane_detect_node._mask_to_centerline_xy 원문 이식)
    """
    xs, ys = _lane_mask_to_lidar_xy(mask)
    if xs.size == 0:
        return xs, ys

    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return xs, ys

    bins = np.floor(xs / LANE_CENTERLINE_BIN_M).astype(np.int32)
    out_x = []
    out_y = []
    for bid in np.unique(bins):
        mask_bin = bins == bid
        if int(np.count_nonzero(mask_bin)) < 2:
            continue
        out_x.append(float(np.median(xs[mask_bin])))
        out_y.append(float(np.median(ys[mask_bin])))

    if not out_x:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float32)
    out_y = np.asarray(out_y, dtype=np.float32)
    order = np.argsort(out_x)
    return out_x[order], out_y[order]


def mask_to_bev_points(mask, H):
    """이미지 픽셀 마스크 → BEV warp → lidar 좌표 차선 중심점 (xs, ys).

    mask: uint8 이진 마스크 (undistort 된 이미지와 같은 픽셀 좌표계).
    H: 이미지 픽셀 → lidar 미터 호모그래피 (load_camera_config 의 "H").
    """
    if mask is None or mask.size == 0:
        return (np.array([], dtype=np.float32),
                np.array([], dtype=np.float32))

    H_pix2bev = M_GRID @ np.asarray(H, dtype=np.float64)
    bev_mask = cv2.warpPerspective(
        mask.astype(np.uint8), H_pix2bev, (BEV_W, BEV_H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _mask_to_centerline_xy(bev_mask)


# ======================== 차선 품질 (등급형 0~1) ========================

def lane_quality(xs, ys):
    """중심점 개수 × 전방 x스팬 기반 등급 점수 (본선 /detect/lane_quality 규격)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = int(xs.size)
    if n == 0:
        return 0.0
    span_x = float(np.max(xs) - np.min(xs)) if n > 1 else 0.0
    count_score = float(np.clip(n / LANE_QUALITY_GOOD_POINTS, 0.0, 1.0))
    span_score = float(np.clip(span_x / LANE_QUALITY_GOOD_SPAN_M, 0.0, 1.0))
    return count_score * span_score
