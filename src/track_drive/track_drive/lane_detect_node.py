#!/usr/bin/env python3
"""
차선 검출 노드 — BEV warp + 차선 마스크 + 품질 점수 발행.

구독: /usb_cam/image_raw/front (rgb8/bgr8 계열)
발행:
  /detect/lane (PoseArray, lidar_frame 미터: x=전방, y=좌)
  /detect/lane_quality (Float32, 0~1)

호모그래피: common.H_PIX2LIDAR_FALLBACK (캘리브된 하드코딩 값)을 BEV 픽셀로 변환.
나중에 YOLO 노드로 통째 교체 시 출력 토픽 두 개만 유지하면 됨.
"""

import numpy as np
import cv2

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32
from tf2_msgs.msg import TFMessage
import tf2_ros

from .common import (
    BEV_H,
    BEV_W,
    CAM_FRONT_FRAME,
    GROUND_Z_IN_LIDAR,
    H_PIX2LIDAR_FALLBACK,
    LANE_CC_MIN_AREA,
    LANE_CC_MIN_H,
    LANE_RUN_MAX_PX,
    LANE_RUN_MIN_PX,
    LANE_WHITE_S_MAX,
    LANE_WHITE_V_MIN,
    LANE_YELLOW_H_MAX,
    LANE_YELLOW_H_MIN,
    LANE_YELLOW_S_MIN,
    LANE_YELLOW_V_MIN,
    LIDAR_FRAME,
    M_GRID,
    R_ROBOT_TO_OPTICAL,
    extract_cv_lane_data,
    extract_lane_mask,
    lane_mask_to_lidar_xy,
    quat_to_R,
)


# ======================== 상수 ========================

# 픽셀 → BEV 픽셀 워프 행렬 (M_GRID @ H_PIX2LIDAR_FALLBACK)
H_PIX2BEV = M_GRID @ H_PIX2LIDAR_FALLBACK

CLS_WHITE_LANE = 6
CLS_YELLOW_MID = 8
LANE_CENTERLINE_BIN_M = 0.18


def _filter_lane_mask(mask):
    if mask is None or mask.size == 0:
        return None
    mask = mask.astype(np.uint8)
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


def _extract_colored_lane_masks(bev_img):
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = V > 5
    white = ((V >= LANE_WHITE_V_MIN) & (S <= LANE_WHITE_S_MAX) & valid)
    yellow = ((H >= LANE_YELLOW_H_MIN) & (H <= LANE_YELLOW_H_MAX)
              & (S >= LANE_YELLOW_S_MIN) & (V >= LANE_YELLOW_V_MIN)
              & valid)
    return (
        _filter_lane_mask(white.astype(np.uint8) * 255),
        _filter_lane_mask(yellow.astype(np.uint8) * 255),
    )


def _mask_to_centerline_xy(mask):
    """YOLO seg 출력처럼 전방 x-bin마다 차선 중심점 하나만 만든다.

    원 YOLO 노드는 각 mask row에서 u 중심점만 발행했다. OpenCV 마스크 픽셀을
    그대로 넘기면 차선 두께/노이즈가 2차식 fit에 들어가 직선 와리가리가 생긴다.
    """
    xs, ys = lane_mask_to_lidar_xy(mask)
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


# ======================== ROS 노드 ========================

class LaneDetectNode(Node):
    def __init__(self):
        super().__init__("lane_detect_node")

        self._K = None
        self._H_pix2bev = H_PIX2BEV
        self._dynamic_h_built = False

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, "/usb_cam/camera_info/front",
            self._on_caminfo, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        tf_static_qos = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(TFMessage, "/tf_static",
                                 self._on_tf_static, tf_static_qos)
        self.create_timer(0.5, self._maybe_build_homography)

        self._pub_lane = self.create_publisher(PoseArray, "/detect/lane", 10)
        self._pub_quality = self.create_publisher(Float32, "/detect/lane_quality", 10)
        self.get_logger().info(
            "lane_detect_node started (dynamic BEV + fallback homography)")

    def _on_caminfo(self, msg):
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_tf_static(self, msg):
        for tr in msg.transforms:
            self.tf_buffer.set_transform_static(tr, "lane_detect_node")

    def _maybe_build_homography(self):
        if self._dynamic_h_built or self._K is None:
            return
        try:
            tr = self.tf_buffer.lookup_transform(
                LIDAR_FRAME, CAM_FRONT_FRAME, Time(),
                timeout=Duration(seconds=0.1),
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

        cam_ground_height = float(t[2] - GROUND_Z_IN_LIDAR)
        if abs(cam_ground_height) < 1e-3:
            self.get_logger().warn("camera z≈ground; keep fallback BEV homography")
            return

        plane_scale = GROUND_Z_IN_LIDAR - float(t[2])
        T_homog = np.array([
            [plane_scale, 0.0, t[0]],
            [0.0, plane_scale, t[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        K_inv = np.linalg.inv(self._K)

        u0 = self._K[0, 2]
        v0 = self._K[1, 2] + self._K[1, 1] * 0.30
        test_pix = np.array([u0, v0, 1.0])

        best = None
        for label, R_eff in (("optical", R_raw),
                             ("robot", R_raw @ R_ROBOT_TO_OPTICAL)):
            H_pl = T_homog @ R_eff @ K_inv
            p = H_pl @ test_pix
            if abs(float(p[2])) < 1e-9:
                continue
            x_chk = float(p[0] / p[2])
            y_chk = float(p[1] / p[2])
            score = x_chk - 2.0 * abs(y_chk)
            if 0.5 < x_chk < 20.0 and abs(y_chk) < 2.0:
                if best is None or score > best[0]:
                    best = (score, label, H_pl, x_chk, y_chk)

        if best is None:
            return

        _, label, H_pix2lidar, x_chk, y_chk = best
        self._H_pix2bev = M_GRID @ H_pix2lidar
        self._dynamic_h_built = True
        self.get_logger().info(
            f"dynamic BEV homography ready [{label}], "
            f"check=({x_chk:+.2f}, {y_chk:+.2f}) m")

    def _on_image(self, msg):
        img = self._image_to_rgb(msg)
        if img is None:
            return

        bev = cv2.warpPerspective(
            img, self._H_pix2bev, (BEV_W, BEV_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        mask = extract_lane_mask(bev)
        _lane_xs, _lane_ys, point_count, quality = extract_cv_lane_data(mask)
        white_mask, yellow_mask = _extract_colored_lane_masks(bev)
        white_xs, white_ys = _mask_to_centerline_xy(white_mask)
        yellow_xs, yellow_ys = _mask_to_centerline_xy(yellow_mask)

        out = PoseArray()
        out.header = msg.header
        out.header.frame_id = LIDAR_FRAME
        for x, y in zip(yellow_xs, yellow_ys):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(CLS_YELLOW_MID)
            out.poses.append(p)
        for x, y in zip(white_xs, white_ys):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(CLS_WHITE_LANE)
            out.poses.append(p)
        self._pub_lane.publish(out)
        self._pub_quality.publish(Float32(data=float(quality)))

    @staticmethod
    def _image_to_rgb(msg):
        enc = msg.encoding.lower()
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(enc)
        if channels is None:
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.step > 0:
                rows = raw.reshape(msg.height, msg.step)
                arr = rows[:, :msg.width * channels].reshape(
                    msg.height, msg.width, channels)
            else:
                arr = raw.reshape(msg.height, msg.width, channels)
        except ValueError:
            return None

        if enc == "rgb8":
            return arr
        if enc == "bgr8":
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if enc == "rgba8":
            return arr[..., :3]
        if enc == "bgra8":
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        if enc == "mono8":
            return cv2.cvtColor(arr.squeeze(axis=2), cv2.COLOR_GRAY2RGB)
        return None


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
