#!/usr/bin/env python3
"""
풀스택 디버거 viewer — 4 카메라 + BEV 맵 + 라인 마스크 별창 + IMU + 제어.

구독 (raw):
  /usb_cam/image_raw/{front,left,right,behind}
  /usb_cam/camera_info/front
  /tf_static
  /scan
  /imu

구독 (planner / motion):
  /target, /center_path, /lane_left, /lane_right, /motion/status, /xycar_motor

처리 (viewer 자체):
  - BEV 호모그래피 동적 빌드 (CameraInfo + tf_static)
  - test_viewer와 같은 extract_lane_mask + lane_mask_to_lidar_xy
  - test_viewer와 같은 LiDAR obstacle clustering

키:
  M : ~/에 front/bev/mask png 덤프
  ESC : 종료
"""

import math
import os
import threading
import time

import cv2
import numpy as np

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
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from geometry_msgs.msg import PointStamped, PoseArray
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from xycar_msgs.msg import XycarMotor
import tf2_ros

import matplotlib
matplotlib.use("TkAgg")
for _k in ("save", "quit", "quit_all", "fullscreen", "home", "back",
           "forward", "pan", "zoom", "grid", "grid_minor", "yscale",
           "xscale", "copy", "help"):
    if f"keymap.{_k}" in matplotlib.rcParams:
        matplotlib.rcParams[f"keymap.{_k}"] = []
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from .common import (
    CAM_FRONT_FRAME,
    GROUND_Z_IN_LIDAR,
    LIDAR_FRAME,
    R_ROBOT_TO_OPTICAL,
    quat_to_R,
    quaternion_to_euler,
)


ANIM_INTERVAL_MS = 100

CAMERA_TOPICS = {
    "front":  "/usb_cam/image_raw/front",
    "left":   "/usb_cam/image_raw/left",
    "right":  "/usb_cam/image_raw/right",
    "behind": "/usb_cam/image_raw/behind",
}
CAM_INFO_FRONT_TOPIC = "/usb_cam/camera_info/front"
SCAN_TOPIC = "/scan"
IMU_TOPIC = "/imu"
TARGET_TOPIC = "/target"
STATUS_TOPIC = "/motion/status"
MOTOR_TOPIC = "/xycar_motor"
CENTER_PATH_TOPIC = "/center_path"
LANE_LEFT_TOPIC = "/lane_left"
LANE_RIGHT_TOPIC = "/lane_right"

# test_viewer와 동일한 표시용 BEV/마스크/장애물 파라미터.
BEV_RES = 0.05
BEV_X_MIN, BEV_X_MAX = -2.0, 10.0
BEV_Y_HALF = 4.0
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))
M_GRID = np.array([
    [0.0, -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES, 0.0, BEV_X_MAX / BEV_RES],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

LANE_WHITE_V_MIN = 200
LANE_WHITE_S_MAX = 50
LANE_YELLOW_H_MIN = 18
LANE_YELLOW_H_MAX = 38
LANE_YELLOW_S_MIN = 80
LANE_YELLOW_V_MIN = 130
LANE_PIX_SUBSAMPLE = 3
LANE_RUN_MIN_PX = 2
LANE_RUN_MAX_PX = 26
LANE_CC_MIN_AREA = 10
LANE_CC_MIN_H = 4

OBSTACLE_X_MIN, OBSTACLE_X_MAX = 0.1, 8.0
OBSTACLE_Y_HALF = 3.5
CLUSTER_GAP = 0.25
CLUSTER_MIN_PTS = 3
OBSTACLE_DRAW_PADDING = 0.05


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cam_front = None
        self.cam_front_recv_time = 0.0
        self.cam_left = None
        self.cam_right = None
        self.cam_behind = None
        self.scan_ranges = None
        self.scan_angles = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.scan_range_max = 100.0
        self.imu_rpy = (0.0, 0.0, 0.0)
        self.K = None
        self.H_pix2bev = None
        self.cam_height_m = float("nan")
        self.convention = "?"
        self.target = None
        self.status = ""
        self.motor_speed = 0.0
        self.motor_angle = 0.0
        self.center_path_xs = np.array([], dtype=np.float32)
        self.center_path_ys = np.array([], dtype=np.float32)
        self.lane_left_xs = np.array([], dtype=np.float32)
        self.lane_left_ys = np.array([], dtype=np.float32)
        self.lane_right_xs = np.array([], dtype=np.float32)
        self.lane_right_ys = np.array([], dtype=np.float32)


class ViewerSubscriber(Node):
    def __init__(self, state):
        super().__init__("fused_viewer_node")
        self.state = state

        for slot, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, slot=slot: self._on_image(slot, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(CameraInfo, CAM_INFO_FRONT_TOPIC,
                                 self._on_caminfo, qos_profile_sensor_data)
        self.create_subscription(LaserScan, SCAN_TOPIC,
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Imu, IMU_TOPIC,
                                 self._on_imu, qos_profile_sensor_data)
        self.create_subscription(PointStamped, TARGET_TOPIC,
                                 self._on_target, 10)
        self.create_subscription(String, STATUS_TOPIC,
                                 self._on_status, 10)
        self.create_subscription(XycarMotor, MOTOR_TOPIC,
                                 self._on_motor, 10)
        self.create_subscription(PoseArray, CENTER_PATH_TOPIC,
                                 self._on_center_path, 10)
        self.create_subscription(PoseArray, LANE_LEFT_TOPIC,
                                 self._on_lane_left, 10)
        self.create_subscription(PoseArray, LANE_RIGHT_TOPIC,
                                 self._on_lane_right, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        tf_static_qos = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(TFMessage, "/tf_static",
                                 self._on_tf_static, tf_static_qos)

        self._H_built = False
        self.create_timer(0.5, self._maybe_build_homography)

        self.get_logger().info("VIEWER")

    def _on_image(self, slot, msg):
        if msg.encoding != "rgb8":
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return
        recv_time = time.monotonic()
        with self.state.lock:
            setattr(self.state, f"cam_{slot}", arr)
            if slot == "front":
                self.state.cam_front_recv_time = recv_time

    def _on_caminfo(self, msg):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self.state.lock:
            self.state.K = K

    def _on_scan(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = (msg.angle_min
                  + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment)
        with self.state.lock:
            self.state.scan_ranges = ranges
            self.state.scan_angles = angles
            self.state.scan_angle_min = msg.angle_min
            self.state.scan_angle_inc = msg.angle_increment
            self.state.scan_range_max = msg.range_max

    def _on_imu(self, msg):
        q = msg.orientation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        with self.state.lock:
            self.state.imu_rpy = (roll, pitch, yaw)

    def _on_tf_static(self, msg):
        for tr in msg.transforms:
            self.tf_buffer.set_transform_static(tr, "fused_viewer_node")

    def _on_target(self, msg):
        with self.state.lock:
            self.state.target = (msg.point.x, msg.point.y)

    def _on_status(self, msg):
        with self.state.lock:
            self.state.status = msg.data

    def _on_motor(self, msg):
        with self.state.lock:
            self.state.motor_speed = float(msg.speed)
            self.state.motor_angle = float(msg.angle)

    def _on_center_path(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.center_path_xs = xs
            self.state.center_path_ys = ys

    def _on_lane_left(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.lane_left_xs = xs
            self.state.lane_left_ys = ys

    def _on_lane_right(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.lane_right_xs = xs
            self.state.lane_right_ys = ys

    def _maybe_build_homography(self):
        if self._H_built:
            return
        with self.state.lock:
            K = self.state.K
        if K is None:
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
            self.get_logger().warn("BEV Z")
            return
        plane_scale = GROUND_Z_IN_LIDAR - float(t[2])
        T_homog = np.array([
            [plane_scale, 0.0, t[0]],
            [0.0, plane_scale, t[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        u0 = K[0, 2]
        v0 = K[1, 2] + K[1, 1] * 0.30
        test_pix = np.array([u0, v0, 1.0])

        candidates = [("optical", R_raw),
                      ("robot", R_raw @ R_ROBOT_TO_OPTICAL)]
        best = None
        for label, R_eff in candidates:
            H_pl = T_homog @ R_eff @ K_inv
            p = H_pl @ test_pix
            if p[2] == 0:
                continue
            X_chk = p[0] / p[2]
            Y_chk = p[1] / p[2]
            score = X_chk - 2.0 * abs(Y_chk)
            self.get_logger().info("BEV TRY")
            if 0.5 < X_chk < 20 and abs(Y_chk) < 2.0:
                if best is None or score > best[0]:
                    best = (score, label, H_pl, X_chk, Y_chk)

        if best is None:
            self.get_logger().warn("BEV ERR")
            return

        _, label, H_pix2lidar, X_chk, Y_chk = best
        H_pix2bev = M_GRID @ H_pix2lidar

        with self.state.lock:
            self.state.H_pix2bev = H_pix2bev
            self.state.cam_height_m = cam_ground_height
            self.state.convention = label
        self._H_built = True
        self.get_logger().info("BEV OK")


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


class Viewer:
    def __init__(self, node, state):
        self.node = node
        self.state = state
        self._last_bev_rgb = None
        self._last_mask = None

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title("kookmin9_viewer")
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[1, 1, 0.8])

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
        self._lane_left_line, = self.ax_map.plot(
            [], [], "-", color="#88ddff", alpha=0.9, linewidth=1.4,
            zorder=6, label="lane bounds")
        self._lane_right_line, = self.ax_map.plot(
            [], [], "-", color="#88ddff", alpha=0.9, linewidth=1.4,
            zorder=6)
        self._center_path_line, = self.ax_map.plot(
            [], [], "-", color="magenta", alpha=0.95, linewidth=2.0,
            zorder=8, label="/center_path")
        self._center_path_pts, = self.ax_map.plot(
            [], [], "x", color="#00ffcc", markersize=6,
            zorder=9, label="path samples")
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

    def _on_key(self, event):
        k = event.key
        if k == "escape":
            plt.close(self.fig)
            return
        if k in ("m", "M"):
            self._dump_debug()
            return

    def _dump_debug(self):
        ts = time.strftime("%H%M%S")
        out_dir = os.path.expanduser("~")
        saved = []
        with self.state.lock:
            front = self.state.cam_front
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

    def _update(self, _frame):
        with self.state.lock:
            cams = {slot: getattr(self.state, f"cam_{slot}")
                    for slot in CAMERA_TOPICS}
            H = self.state.H_pix2bev
            front = self.state.cam_front
            convention = self.state.convention
            cam_height_m = self.state.cam_height_m
            K = self.state.K
            target = self.state.target
            status = self.state.status
            motor_speed = self.state.motor_speed
            motor_angle = self.state.motor_angle
            center_xs = self.state.center_path_xs.copy()
            center_ys = self.state.center_path_ys.copy()
            lane_left_xs = self.state.lane_left_xs.copy()
            lane_left_ys = self.state.lane_left_ys.copy()
            lane_right_xs = self.state.lane_right_xs.copy()
            lane_right_ys = self.state.lane_right_ys.copy()
            ranges = self.state.scan_ranges
            angles = self.state.scan_angles
            scan_angle_min = self.state.scan_angle_min
            scan_angle_inc = self.state.scan_angle_inc
            scan_range_max = self.state.scan_range_max
            roll, pitch, yaw = self.state.imu_rpy

        for slot, arr in cams.items():
            if arr is not None:
                self.cam_im[slot].set_data(arr)

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
                disp[mask > 0] = [255, 0, 255]
            self.bev_im.set_data(disp)
            cam_h_txt = (f"{cam_height_m:+.2f} m"
                         if math.isfinite(cam_height_m) else "n/a")
            self._bev_status.set_text(
                f"BEV ON [{convention}]  (cam_h={cam_h_txt})"
            )
            self._last_bev_rgb = bev_img
            self._last_mask = mask
            self.lane_mask_im.set_data(
                mask if mask is not None
                else np.zeros((BEV_H, BEV_W), dtype=np.uint8))
        else:
            missing = []
            if K is None: missing.append("CameraInfo")
            if K is not None and H is None: missing.append("/tf_static")
            if front is None: missing.append("/usb_cam/image_raw/front")
            self._bev_status.set_text(
                "BEV: waiting for " + ", ".join(missing) if missing else "")
            self.lane_mask_im.set_data(np.zeros((BEV_H, BEV_W), dtype=np.uint8))

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

        if lane_left_xs.size >= 2:
            self._lane_left_line.set_data(-lane_left_ys, lane_left_xs)
        else:
            self._lane_left_line.set_data([], [])
        if lane_right_xs.size >= 2:
            self._lane_right_line.set_data(-lane_right_ys, lane_right_xs)
        else:
            self._lane_right_line.set_data([], [])
        if center_xs.size >= 2:
            self._center_path_line.set_data(-center_ys, center_xs)
            self._center_path_pts.set_data(-center_ys, center_xs)
        else:
            self._center_path_line.set_data([], [])
            self._center_path_pts.set_data([], [])

        xs = ys = np.array([])
        if ranges is not None and ranges.size > 0:
            n = ranges.size
            if angles is None or angles.size != n:
                angles = scan_angle_min + np.arange(n) * scan_angle_inc
            valid = (np.isfinite(ranges)
                     & (ranges > 0.05)
                     & (ranges < scan_range_max - 1e-3))
            if valid.any():
                xs = ranges[valid] * np.cos(angles[valid])
                ys = ranges[valid] * np.sin(angles[valid])
                self.lidar_pts.set_data(-ys, xs)
            else:
                self.lidar_pts.set_data([], [])

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

        if target is not None:
            tx, ty = target
            sx, sy = -ty, tx
            self._target_marker.set_data([sx], [sy])
            self._target_line.set_data([0, sx], [0, sy])
        else:
            self._target_marker.set_data([], [])
            self._target_line.set_data([], [])

        mode_txt = f"AUTO [{status}]"
        self._det_status.set_text(
            f"{mode_txt}    lanes: {n_lane:4d} pts   "
            f"path: {center_xs.size:2d} samp   obstacles: {len(clusters):2d}"
        )

        for art in self._arrows:
            art.remove()
        self._arrows = []
        L = 0.9
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(roll), L * math.sin(roll),
            color="red", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(pitch + math.pi / 2), L * math.sin(pitch + math.pi / 2),
            color="green", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(yaw), L * math.sin(yaw),
            color="blue", width=0.02, head_width=0.06, length_includes_head=True))
        self._imu_text.set_text(
            f"roll  (red)   = {math.degrees(roll):+7.2f} deg\n"
            f"pitch (green) = {math.degrees(pitch):+7.2f} deg\n"
            f"yaw   (blue)  = {math.degrees(yaw):+7.2f} deg"
        )

        self._ctrl_text.set_text(
            f"speed = {motor_speed:+6.2f}    angle = {motor_angle:+6.2f}\n"
            f"AUTO  = ON\n"
            f"\n"
            f"  M       : dump BEV/mask/front to ~/\n"
            f"  ESC     : exit"
        )

    def show(self):
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    state = SharedState()
    node = ViewerSubscriber(state)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viewer = Viewer(node, state)
    try:
        viewer.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
