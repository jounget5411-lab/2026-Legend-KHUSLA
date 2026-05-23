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
for _k in ("save", "quit", "quit_all", "fullscreen", "home", "back", "forward",
           "pan", "zoom", "grid", "grid_minor", "yscale", "xscale", "copy",
           "help"):
    if f"keymap.{_k}" in matplotlib.rcParams:
        matplotlib.rcParams[f"keymap.{_k}"] = []
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# ---------- 설정 상수 -------------------------------------------------------

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

SPEED_MIN, SPEED_MAX = -50.0, 50.0
ANGLE_MIN, ANGLE_MAX = -100.0, 100.0
SPEED_STEP = 1.0
ANGLE_STEP = 5.0

ANIM_INTERVAL_MS = 100

# ---------- BEV 좌표계 ------------------------------------------------------
BEV_RES   = 0.05
BEV_X_MIN, BEV_X_MAX = -2.0, 10.0
BEV_Y_HALF           = 4.0
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))

M_GRID = np.array([
    [ 0.0,         -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,         BEV_X_MAX  / BEV_RES],
    [ 0.0,           0.0,          1.0],
], dtype=np.float64)

# ---------- 차선 HSV 임계값 — 스크린샷 색상 기준 + 여유 ----------------------
LANE_WHITE_V_MIN  = 200
LANE_WHITE_S_MAX  = 50
LANE_YELLOW_H_MIN = 18
LANE_YELLOW_H_MAX = 38
LANE_YELLOW_S_MIN = 80
LANE_YELLOW_V_MIN = 130
LANE_PIX_SUBSAMPLE = 3

# ---------- 장애물 클러스터링 -----------------------------------------------
OBSTACLE_X_MIN, OBSTACLE_X_MAX = 0.1, 8.0
OBSTACLE_Y_HALF = 3.5
CLUSTER_GAP = 0.25
CLUSTER_MIN_PTS = 3
OBSTACLE_DRAW_PADDING = 0.05

# robot 컨벤션(x=forward) → optical 컨벤션(z=forward) 회전
R_ROBOT_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


# ---------- 공유 상태 -------------------------------------------------------

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
    K: np.ndarray = None
    H_pix2bev: np.ndarray = None
    cam_height_m: float = float("nan")
    convention: str = "?"


# ---------- ROS 노드 --------------------------------------------------------

class TestViewerNode(Node):
    def __init__(self, latest: Latest):
        super().__init__("kookmin9_viewer")
        self.latest = latest
        self._lock = threading.Lock()

        for slot, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, slot=slot: self._on_image(slot, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(CameraInfo, CAM_INFO_FRONT, self._on_caminfo,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, IMU_TOPIC, self._on_imu,
                                 qos_profile_sensor_data)

        self._motor_pub = self.create_publisher(XycarMotor, XYCAR_TOPIC, 10)

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

        self.target_speed = 0.0
        self.target_angle = 0.0
        self.create_timer(0.1, self._publish_motor)

        self._H_built = False
        self.create_timer(0.5, self._maybe_build_homography)

        self.get_logger().info(
            "kookmin9_viewer 시작. W/A/S/D 조작, M 키로 디버그 덤프."
        )

    # --- callbacks ----------------------------------------------------------

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

    # --- BEV 호모그래피 (수정된 공식 + 자동 컨벤션 판별) ----------------------

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

        d = float(t[2])
        if abs(d) < 1e-3:
            self.get_logger().warn("카메라 광학중심 z≈0. BEV 비활성.")
            return

        # 픽셀→라이다 지면 호모그래피 (직접 유도)
        T_homog = np.array([
            [-d,  0.0, t[0]],
            [0.0, -d,  t[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        # 화면 중앙 + 30% 아래 픽셀 (차량 정면 도로 부근) 으로 컨벤션 판별
        u0 = K[0, 2]
        v0 = K[1, 2] + K[1, 1] * 0.30
        test_pix = np.array([u0, v0, 1.0])

        candidates = [
            ("optical", R_raw),
            ("robot",   R_raw @ R_ROBOT_TO_OPTICAL),
        ]
        best = None
        for label, R_eff in candidates:
            H_pl = T_homog @ R_eff @ K_inv
            p = H_pl @ test_pix
            if p[2] == 0:
                continue
            X_chk = p[0] / p[2]; Y_chk = p[1] / p[2]
            score = X_chk - 2.0 * abs(Y_chk)   # 전방 양수 + 중앙 가까울수록 좋음
            self.get_logger().info(
                f"BEV 시도 [{label}]: 화면 중앙-하단 → "
                f"(X={X_chk:+.2f}, Y={Y_chk:+.2f}) m   score={score:+.2f}"
            )
            if 0.5 < X_chk < 20 and abs(Y_chk) < 2.0:
                if best is None or score > best[0]:
                    best = (score, label, H_pl, X_chk, Y_chk)

        if best is None:
            self.get_logger().warn(
                "어떤 컨벤션으로도 합리적 BEV 가 안 나옴. TF 또는 t_z 부호 확인 필요."
            )
            return

        _, label, H_pix2lidar, X_chk, Y_chk = best
        H_pix2bev = M_GRID @ H_pix2lidar

        with self._lock:
            self.latest.H_pix2bev = H_pix2bev
            self.latest.cam_height_m = d
            self.latest.convention = label
        self._H_built = True
        self.get_logger().info(
            f"BEV 호모그래피 빌드 완료 [{label}]. "
            f"cam_h={d:+.3f} m, t_xy=({t[0]:+.3f}, {t[1]:+.3f}) m, "
            f"중앙-하단 픽셀 → (X={X_chk:+.2f}, Y={Y_chk:+.2f}) m"
        )

    # --- publish ------------------------------------------------------------

    def _publish_motor(self):
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self.target_speed)
        msg.angle = float(self.target_angle)
        self._motor_pub.publish(msg)

    def adjust_speed(self, delta):
        self.target_speed = float(np.clip(self.target_speed + delta, SPEED_MIN, SPEED_MAX))
    def adjust_angle(self, delta):
        self.target_angle = float(np.clip(self.target_angle + delta, ANGLE_MIN, ANGLE_MAX))
    def stop(self):
        self.target_speed = 0.0; self.target_angle = 0.0
    def speed_zero(self):
        self.target_speed = 0.0


# ---------- 유틸 ------------------------------------------------------------

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
    return mask


def lane_mask_to_lidar_xy(mask):
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    screen_x = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    screen_y = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return screen_x, screen_y


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


# ---------- matplotlib UI ---------------------------------------------------

class Viewer:
    def __init__(self, node, latest):
        self.node = node
        self.latest = latest
        self._last_bev_rgb = None
        self._last_mask = None

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title("kookmin9_viewer")
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[1, 1, 0.8])

        # 카메라 4개
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

        # BEV 통합 맵
        self.ax_map = self.fig.add_subplot(gs[0:2, 2:4])
        self.ax_map.set_title(
            "BEV map  —  camera warp + mask(magenta) + LiDAR(cyan) + obstacles(red)  "
            "(lidar_frame, meters)"
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

        # 거리 가이드 동심원 + 라벨
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

        self._bev_status = self.ax_map.text(
            0.02, 0.98, "BEV: waiting for CameraInfo + /tf_static …",
            transform=self.ax_map.transAxes, color="yellow", fontsize=9,
            verticalalignment="top", zorder=7,
        )
        self._det_status = self.ax_map.text(
            0.02, 0.02, "",
            transform=self.ax_map.transAxes, color="white", fontsize=9,
            verticalalignment="bottom", zorder=7,
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
        )
        self.ax_map.legend(loc="lower right", fontsize=8, framealpha=0.7)

        # IMU
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

        # 제어
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
        if   k in ("w", "up"):     self.node.adjust_speed(+SPEED_STEP)
        elif k in ("s", "down"):   self.node.adjust_speed(-SPEED_STEP)
        elif k in ("a", "left"):   self.node.adjust_angle(-ANGLE_STEP)
        elif k in ("d", "right"):  self.node.adjust_angle(+ANGLE_STEP)
        elif k == " ":             self.node.stop()
        elif k == "q":             self.node.speed_zero()
        elif k in ("m", "M"):      self._dump_debug()
        elif k == "escape":        plt.close(self.fig)

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

    def _update(self, _frame):
        # 카메라 원본
        for slot, im in self.cam_im.items():
            arr = getattr(self.latest, f"cam_{slot}")
            if arr is not None:
                im.set_data(arr)

        # BEV warp + 차선 마스크
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
                disp[mask > 0] = [255, 0, 255]  # 마젠타 강조
            self.bev_im.set_data(disp)
            self._bev_status.set_text(
                f"BEV ON [{self.latest.convention}]  "
                f"(cam_h={self.latest.cam_height_m:+.2f} m)"
            )
            self._last_bev_rgb = bev_img
            self._last_mask = mask
        else:
            missing = []
            if self.latest.K is None: missing.append("CameraInfo")
            if self.latest.K is not None and H is None: missing.append("/tf_static")
            if front is None: missing.append("/usb_cam/image_raw/front")
            self._bev_status.set_text(
                "BEV: waiting for " + ", ".join(missing) if missing else "")

        if mask is not None:
            lx, ly = lane_mask_to_lidar_xy(mask)
            self.lane_pts.set_data(lx, ly)
            n_lane = lx.size
        else:
            self.lane_pts.set_data([], [])
            n_lane = 0

        # LiDAR
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

        # 장애물
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

        self._det_status.set_text(
            f"lanes: {n_lane:4d} pts   obstacles: {len(clusters):2d}"
        )

        # IMU
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

        # 제어
        self._ctrl_text.set_text(
            f"speed = {self.node.target_speed:+6.2f}    angle = {self.node.target_angle:+6.2f}\n"
            f"\n"
            f"  W / S   : speed +/- {SPEED_STEP}\n"
            f"  A / D   : angle +/- {ANGLE_STEP}\n"
            f"  Space   : stop (speed=0, angle=0)\n"
            f"  Q       : speed=0 (keep angle)\n"
            f"  M       : dump BEV/mask/front to ~/\n"
            f"  ESC     : exit"
        )

    def show(self):
        plt.show()


# ---------- main ------------------------------------------------------------

def main():
    rclpy.init()
    latest = Latest()
    node = TestViewerNode(latest)

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