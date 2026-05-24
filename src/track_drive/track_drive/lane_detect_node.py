#!/usr/bin/env python3
"""
차선 검출 노드 — sumin.py의 BEV warp + 차선 마스크 로직 분리.

구독: /usb_cam/image_raw/front (RGB 640×480)
발행: /detect/lane (PoseArray, lidar_frame 미터: x=전방, y=좌)

나중에 YOLO 노드로 통째 교체 시 /detect/lane 출력 형식만 유지.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray

# ======================== 상수 ========================

# Ground H: pixel → lidar_frame 미터 (X=전방, Y=좌)
# tf_static 기반, 지면 투영 (z = -0.228 in lidar frame)
# homography_ground.npz의 H를 lidar_frame 좌표로 변환한 값
H_PIX2LIDAR = np.array([
    [ 0.00000000000000e+00,  4.63055528548851e-03, -2.73689017508590e+00],
    [ 4.18097842466549e-03,  1.87301533010043e-04, -1.37819064484720e+00],
    [ 0.00000000000000e+00, -2.40130177039888e-03,  5.16378846754945e-01],
], dtype=np.float64)

# BEV 그리드 파라미터 (sumin.py 기준)
BEV_RES = 0.05          # m/pixel
BEV_X_MIN = -2.0        # 후방 한계 (m)
BEV_X_MAX = 12.0        # 전방 한계 (m)
BEV_Y_HALF = 7.0        # 좌우 한계 (m)
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))   # 200
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))  # 280

# lidar_frame 미터 → BEV 픽셀 변환 행렬
M_GRID = np.array([
    [ 0.0,           -1.0 / BEV_RES,  BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,            BEV_X_MAX / BEV_RES],
    [ 0.0,            0.0,            1.0],
], dtype=np.float64)

# warpPerspective용: pixel → BEV pixel (= M_GRID @ H_PIX2LIDAR)
H_PIX2BEV = M_GRID @ H_PIX2LIDAR

# 차선 HSV 임계값
LANE_WHITE_V_MIN = 200
LANE_WHITE_S_MAX = 50
LANE_YELLOW_H_MIN = 18
LANE_YELLOW_H_MAX = 38
LANE_YELLOW_S_MIN = 80
LANE_YELLOW_V_MIN = 130

# 차선 필터링
LANE_PIX_SUBSAMPLE = 3
LANE_RUN_MIN_PX = 2
LANE_RUN_MAX_PX = 35
LANE_CC_MIN_AREA = 6
LANE_CC_MIN_H = 3

# ======================== 차선 검출 함수 (sumin.py 원본) ========================

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
    """BEV 마스크 → lidar_frame 미터 (x=전방, y=좌)."""
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    right_m = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    forward_m = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return forward_m, -right_m  # (X_lidar=전방, Y_lidar=좌)


# ======================== ROS 노드 ========================

class LaneDetectNode(Node):
    def __init__(self):
        super().__init__("lane_detect_node")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)

        self._pub = self.create_publisher(PoseArray, "/detect/lane", 10)
        self.get_logger().info("lane_detect_node started")

    def _on_image(self, msg):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return

        bev = cv2.warpPerspective(
            img, H_PIX2BEV, (BEV_W, BEV_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        mask = extract_lane_mask(bev)
        fwd, left = lane_mask_to_lidar_xy(mask)

        out = PoseArray()
        out.header = msg.header
        out.header.frame_id = "lidar_frame"
        for x, y in zip(fwd, left):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            out.poses.append(p)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
