#!/usr/bin/env python3
"""
통합 파이프라인 검증 viewer — 세 소스를 BEV에 구분 표시.

회색: 라이다 원본 (/scan raw, 클러스터 전)
초록: 카메라 H변환 (/detect/lane, 검출 노드 출력)
마젠타: 통합 차선 (/fused/lane, 통합 노드 출력)
빨강 원: 통합 장애물 (/fused/obstacles, 중심+반경)

좌표계: lidar_frame (X=전방+, Y=좌+, 미터)
BEV 표시: 가로=Y(좌+→화면좌), 세로=X(전방+→화면위)
"""

import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray

import matplotlib
matplotlib.use("TkAgg")
for _k in ("save", "quit", "quit_all", "fullscreen", "home", "back",
           "forward", "pan", "zoom", "grid", "grid_minor", "yscale",
           "xscale", "copy", "help"):
    if f"keymap.{_k}" in matplotlib.rcParams:
        matplotlib.rcParams[f"keymap.{_k}"] = []
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================== BEV 표시 범위 ========================

BEV_X_MIN, BEV_X_MAX = -2.0, 12.0   # 전방 범위 (m)
BEV_Y_HALF = 5.0                      # 좌우 ±5m
LIDAR_RANGE_MAX = 12.0
ANIM_INTERVAL_MS = 150

# ======================== 공유 데이터 ========================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        # 라이다 원본
        self.scan_xs = np.array([])
        self.scan_ys = np.array([])
        # 검출 차선
        self.det_xs = np.array([])
        self.det_ys = np.array([])
        # 통합 차선
        self.fused_lane_xs = np.array([])
        self.fused_lane_ys = np.array([])
        # 통합 장애물 (cx, cy, r)
        self.fused_obs = []


# ======================== ROS 노드 ========================

class ViewerSubscriber(Node):
    def __init__(self, state: SharedState):
        super().__init__("fused_viewer_node")
        self.state = state

        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(
            PoseArray, "/detect/lane", self._on_detect, 10)
        self.create_subscription(
            PoseArray, "/fused/lane", self._on_fused_lane, 10)
        self.create_subscription(
            PoseArray, "/fused/obstacles", self._on_fused_obs, 10)

        self.get_logger().info("fused_viewer_node started — subscribing only")

    def _on_scan(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = ranges.size
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        valid = (np.isfinite(ranges)
                 & (ranges > 0.05)
                 & (ranges < min(LIDAR_RANGE_MAX, msg.range_max - 1e-3)))
        if valid.any():
            xs = ranges[valid] * np.cos(angles[valid])
            ys = ranges[valid] * np.sin(angles[valid])
        else:
            xs = ys = np.array([])
        with self.state.lock:
            self.state.scan_xs = xs
            self.state.scan_ys = ys

    def _on_detect(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.det_xs = xs
            self.state.det_ys = ys

    def _on_fused_lane(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.fused_lane_xs = xs
            self.state.fused_lane_ys = ys

    def _on_fused_obs(self, msg):
        obs = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        with self.state.lock:
            self.state.fused_obs = obs


# ======================== matplotlib viewer ========================

class BEVViewer:
    def __init__(self, state: SharedState):
        self.state = state

        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.fig.canvas.manager.set_window_title("fused_viewer")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-BEV_Y_HALF, BEV_Y_HALF)
        self.ax.set_ylim(BEV_X_MIN, BEV_X_MAX)
        self.ax.set_xlabel("Y lateral [m]  (← left  |  right →)")
        self.ax.set_ylabel("X forward [m]")
        self.ax.set_title("Pipeline BEV: scan(gray) / detect(green) / fused(magenta+red)")

        # 거리 가이드 원
        theta = np.linspace(0, 2 * math.pi, 128)
        for r in (1, 2, 3, 5, 8):
            self.ax.plot(r * np.sin(theta), r * np.cos(theta),
                         color="white", alpha=0.2, lw=0.7)
            self.ax.text(0.05, r + 0.08, f"{r}m",
                         color="white", alpha=0.5, fontsize=7)

        # 차량 삼각형
        veh = np.array([[0, 0.45], [-0.20, -0.25], [0.20, -0.25]])
        self.ax.add_patch(plt.Polygon(
            veh, closed=True, color="#4ec3ff", alpha=0.9, zorder=10))

        # 스캐터/라인 핸들 (BEV: 가로=Y, 세로=X → set_data(-y, x))
        self.scan_pts = self.ax.scatter([], [], s=1, c="#888888", alpha=0.15, zorder=1)
        self.det_pts = self.ax.scatter([], [], s=4, c="#33ff66", alpha=0.6, zorder=3,
                                        label="/detect/lane")
        self.fused_lane_pts = self.ax.scatter([], [], s=6, c="magenta", alpha=0.5,
                                               zorder=4, label="/fused/lane")
        self._obs_artists = []

        self._status = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            color="white", fontsize=9, va="top", family="monospace",
            bbox=dict(fc="black", alpha=0.6, ec="none", pad=3), zorder=20)

        self.ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
        self.ax.set_facecolor("#1a1a2e")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(
            self.fig, self._update, interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False)

    def _on_key(self, event):
        if event.key == "escape":
            plt.close(self.fig)

    def _update(self, _frame):
        with self.state.lock:
            scan_xs = self.state.scan_xs.copy()
            scan_ys = self.state.scan_ys.copy()
            det_xs = self.state.det_xs.copy()
            det_ys = self.state.det_ys.copy()
            fl_xs = self.state.fused_lane_xs.copy()
            fl_ys = self.state.fused_lane_ys.copy()
            obs = list(self.state.fused_obs)

        # 라이다 raw (회색) — BEV: plot(-y, x)
        if scan_xs.size > 0:
            self.scan_pts.set_offsets(np.c_[-scan_ys, scan_xs])
        else:
            self.scan_pts.set_offsets(np.empty((0, 2)))

        # 검출 차선 (초록)
        if det_xs.size > 0:
            self.det_pts.set_offsets(np.c_[-det_ys, det_xs])
        else:
            self.det_pts.set_offsets(np.empty((0, 2)))

        # 통합 차선 (마젠타)
        if fl_xs.size > 0:
            self.fused_lane_pts.set_offsets(np.c_[-fl_ys, fl_xs])
        else:
            self.fused_lane_pts.set_offsets(np.empty((0, 2)))

        # 통합 장애물 (빨강 원)
        for art in self._obs_artists:
            art.remove()
        self._obs_artists.clear()
        for cx, cy, r in obs:
            sx, sy = -cy, cx  # BEV 좌표
            circ = plt.Circle((sx, sy), r, fill=False, ec="red", lw=1.5, zorder=5)
            self.ax.add_patch(circ)
            self._obs_artists.append(circ)
            txt = self.ax.text(
                sx, sy + r + 0.12, f"{math.hypot(cx, cy):.1f}m",
                color="red", fontsize=7, ha="center", zorder=5)
            self._obs_artists.append(txt)

        # 상태 텍스트
        self._status.set_text(
            f"scan raw:  {scan_xs.size:5d} pts\n"
            f"det lane:  {det_xs.size:5d} pts\n"
            f"fused lane:{fl_xs.size:5d} pts\n"
            f"fused obs: {len(obs):5d}")

    def show(self):
        plt.show()


# ======================== main ========================

def main(args=None):
    rclpy.init(args=args)
    state = SharedState()
    node = ViewerSubscriber(state)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viewer = BEVViewer(state)
    try:
        viewer.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
