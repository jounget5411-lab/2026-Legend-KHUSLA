#!/usr/bin/env python3
"""
통합 파이프라인 검증 viewer — YOLO+라이다 결과를 BEV에 색 구분 표시.

옅은 회색: 라이다 원본 (/scan raw, 클러스터 전)
노랑: 중앙선 (/fused/lane 중 cls=MID 8) — centerline 압축돼서 한 행당 1점
흰색: 외곽선 (/fused/lane 중 cls=LANE 6, CHILD_LANE 2)
주황: 정지선 (/fused/stopline)
연두: 출발/결승선 (/fused/goal)
빨강 큰 점: 라이다 장애물 (/fused/obstacles)

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
from geometry_msgs.msg import PoseArray, PointStamped

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

BEV_X_MIN, BEV_X_MAX = -3.0, 18.0   # 전방 범위 (m) — ROI 최대
BEV_Y_HALF = 8.0                      # 좌우 ±8m
LIDAR_RANGE_MAX = 15.0
ANIM_INTERVAL_MS = 150

# YOLO 클래스 분류 (path_planner와 동일하게 유지)
YELLOW_CLS_IDS = {8}        # MID
WHITE_CLS_IDS = {6, 2}      # LANE, CHILD_LANE

# ======================== 공유 데이터 ========================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.scan_xs = np.array([])
        self.scan_ys = np.array([])
        self.fused_yellow_xs = np.array([])
        self.fused_yellow_ys = np.array([])
        self.fused_white_xs = np.array([])
        self.fused_white_ys = np.array([])
        self.fused_other_xs = np.array([])   # cls가 노랑/흰색 둘 다 아니면
        self.fused_other_ys = np.array([])
        self.fused_cls_hist = {}             # cls_id → count (디버그용)
        self.fused_stop_xs = np.array([])
        self.fused_stop_ys = np.array([])
        self.fused_goal_xs = np.array([])
        self.fused_goal_ys = np.array([])
        self.fused_obs = []  # (cx, cy, r)
        self.center_xs = np.array([])
        self.center_ys = np.array([])
        self.lane_left_xs = np.array([])
        self.lane_left_ys = np.array([])
        self.lane_right_xs = np.array([])
        self.lane_right_ys = np.array([])
        self.yellow_inlier_xs = np.array([])
        self.yellow_inlier_ys = np.array([])
        self.target_xy = None  # (x, y) or None


# ======================== ROS 노드 ========================

class ViewerSubscriber(Node):
    def __init__(self, state: SharedState):
        super().__init__("fused_viewer_node")
        self.state = state

        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(
            PoseArray, "/fused/lane", self._on_fused_lane, 10)
        self.create_subscription(
            PoseArray, "/fused/stopline", self._on_fused_stop, 10)
        self.create_subscription(
            PoseArray, "/fused/goal", self._on_fused_goal, 10)
        self.create_subscription(
            PoseArray, "/fused/obstacles", self._on_fused_obs, 10)
        self.create_subscription(
            PoseArray, "/center_path", self._on_center_path, 10)
        self.create_subscription(
            PoseArray, "/lane_left", self._on_lane_left, 10)
        self.create_subscription(
            PoseArray, "/lane_right", self._on_lane_right, 10)
        self.create_subscription(
            PoseArray, "/fit/yellow_inliers", self._on_yellow_inliers, 10)
        self.create_subscription(
            PointStamped, "/target", self._on_target, 10)

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

    def _on_fused_lane(self, msg):
        # position.z 에 YOLO cls_id가 들어있다 (integration_node에서 보존).
        n = len(msg.poses)
        if n == 0:
            empty = np.array([], dtype=np.float32)
            with self.state.lock:
                self.state.fused_yellow_xs = empty
                self.state.fused_yellow_ys = empty
                self.state.fused_white_xs = empty
                self.state.fused_white_ys = empty
                self.state.fused_other_xs = empty
                self.state.fused_other_ys = empty
                self.state.fused_cls_hist = {}
            return
        xs = np.fromiter((p.position.x for p in msg.poses), dtype=np.float32, count=n)
        ys = np.fromiter((p.position.y for p in msg.poses), dtype=np.float32, count=n)
        cls = np.fromiter((int(p.position.z) for p in msg.poses), dtype=np.int32, count=n)
        ym = np.isin(cls, list(YELLOW_CLS_IDS))
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        om = ~(ym | wm)
        # 디버그: 들어온 cls_id 히스토그램 — 모델 라벨이 우리 가정과 다르면
        # 여기서 unknown cls가 잡혀서 상태창에 표시됨.
        uniq, cnts = np.unique(cls, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(uniq, cnts)}
        with self.state.lock:
            self.state.fused_yellow_xs = xs[ym]
            self.state.fused_yellow_ys = ys[ym]
            self.state.fused_white_xs = xs[wm]
            self.state.fused_white_ys = ys[wm]
            self.state.fused_other_xs = xs[om]
            self.state.fused_other_ys = ys[om]
            self.state.fused_cls_hist = hist

    def _on_fused_stop(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.fused_stop_xs = xs
            self.state.fused_stop_ys = ys

    def _on_fused_goal(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.fused_goal_xs = xs
            self.state.fused_goal_ys = ys

    def _on_fused_obs(self, msg):
        obs = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        with self.state.lock:
            self.state.fused_obs = obs

    def _on_center_path(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.center_xs = xs
            self.state.center_ys = ys

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

    def _on_yellow_inliers(self, msg):
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float32)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float32)
        with self.state.lock:
            self.state.yellow_inlier_xs = xs
            self.state.yellow_inlier_ys = ys

    def _on_target(self, msg):
        with self.state.lock:
            self.state.target_xy = (msg.point.x, msg.point.y)


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
        self.ax.set_title("BEV: yellow=MID  white=LANE  lime=inliers  "
                          "orange=STOP  green=GOAL  red=obstacles")

        theta = np.linspace(0, 2 * math.pi, 128)
        for r in (1, 2, 3, 5, 8, 12, 16):
            self.ax.plot(r * np.sin(theta), r * np.cos(theta),
                         color="white", alpha=0.2, lw=0.7)
            self.ax.text(0.05, r + 0.08, f"{r}m",
                         color="white", alpha=0.5, fontsize=7)

        veh = np.array([[0, 0.45], [-0.20, -0.25], [0.20, -0.25]])
        self.ax.add_patch(plt.Polygon(
            veh, closed=True, color="#4ec3ff", alpha=0.9, zorder=10))

        self.scan_pts = self.ax.scatter([], [], s=1, c="#555555", alpha=0.06, zorder=1)
        # YOLO 클래스별 — centerline 압축으로 점 수가 적으니 크게 + 외곽선으로 또렷이.
        self.fused_yellow_pts = self.ax.scatter(
            [], [], s=24, c="#ffd400", alpha=0.95,
            edgecolors="#1a1a2e", linewidths=0.4,
            zorder=5, label="MID (yellow)")
        self.fused_white_pts = self.ax.scatter(
            [], [], s=18, c="white", alpha=0.9,
            edgecolors="#1a1a2e", linewidths=0.4,
            zorder=4, label="LANE (white)")
        # unknown cls — 모델 라벨이 우리 가정과 다른 경우 여기 잡힌다. 눈에 띄게.
        self.fused_other_pts = self.ax.scatter(
            [], [], s=20, c="magenta", alpha=0.9,
            edgecolors="#1a1a2e", linewidths=0.4,
            zorder=4, label="unknown cls")
        self.fused_stop_pts = self.ax.scatter(
            [], [], s=20, c="#ff8800", alpha=0.95,
            edgecolors="#1a1a2e", linewidths=0.4,
            zorder=4, label="STOP")
        self.fused_goal_pts = self.ax.scatter(
            [], [], s=20, c="#22ff88", alpha=0.95,
            edgecolors="#1a1a2e", linewidths=0.4,
            zorder=4, label="GOAL")
        # yellow fit inliers — centerline 압축으로 MID 점과 거의 동일. 디버그용 최소 표시.
        self.yellow_inlier_pts = self.ax.scatter(
            [], [], s=4, c="#00ff66", alpha=0.45, zorder=6,
            label="yellow inliers")
        self.obs_pts = self.ax.scatter([], [], s=80, c="red", alpha=0.9,
                                        zorder=5, label="/fused/obstacles")

        # 2차식 피팅 (좌/우)
        self.lane_left_line, = self.ax.plot(
            [], [], "-", color="#88ddff", lw=1.6, alpha=0.9,
            zorder=4, label="lane fit (L/R)")
        self.lane_right_line, = self.ax.plot(
            [], [], "-", color="#88ddff", lw=1.6, alpha=0.9, zorder=4)

        # 중심선 샘플 (조향 포인트들)
        self.center_line, = self.ax.plot(
            [], [], "-", color="magenta", lw=1.3, alpha=0.8,
            zorder=6, label="center path")
        self.center_pts = self.ax.scatter(
            [], [], s=28, marker="x", c="#00ffcc",
            linewidths=1.6, zorder=7, label="steer samples")

        # 단일 타겟
        self.target_marker, = self.ax.plot(
            [], [], "*", color="magenta", markersize=18,
            markeredgecolor="black", markeredgewidth=0.5,
            zorder=9, label="/target")

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
            fy_xs = self.state.fused_yellow_xs.copy()
            fy_ys = self.state.fused_yellow_ys.copy()
            fw_xs = self.state.fused_white_xs.copy()
            fw_ys = self.state.fused_white_ys.copy()
            fo_xs = self.state.fused_other_xs.copy()
            fo_ys = self.state.fused_other_ys.copy()
            fs_xs = self.state.fused_stop_xs.copy()
            fs_ys = self.state.fused_stop_ys.copy()
            fg_xs = self.state.fused_goal_xs.copy()
            fg_ys = self.state.fused_goal_ys.copy()
            obs = list(self.state.fused_obs)
            c_xs = self.state.center_xs.copy()
            c_ys = self.state.center_ys.copy()
            ll_xs = self.state.lane_left_xs.copy()
            ll_ys = self.state.lane_left_ys.copy()
            lr_xs = self.state.lane_right_xs.copy()
            lr_ys = self.state.lane_right_ys.copy()
            yi_xs = self.state.yellow_inlier_xs.copy()
            yi_ys = self.state.yellow_inlier_ys.copy()
            cls_hist = dict(self.state.fused_cls_hist)
            target = self.state.target_xy

        def _xy(xs, ys):
            return np.c_[-ys, xs] if xs.size > 0 else np.empty((0, 2))

        # 라이다 raw (옅은 회색)
        self.scan_pts.set_offsets(_xy(scan_xs, scan_ys))

        # 차선 클래스별 표시
        self.fused_yellow_pts.set_offsets(_xy(fy_xs, fy_ys))
        self.fused_white_pts.set_offsets(_xy(fw_xs, fw_ys))
        self.fused_other_pts.set_offsets(_xy(fo_xs, fo_ys))
        self.fused_stop_pts.set_offsets(_xy(fs_xs, fs_ys))
        self.fused_goal_pts.set_offsets(_xy(fg_xs, fg_ys))

        # yellow fit inliers (라임)
        self.yellow_inlier_pts.set_offsets(_xy(yi_xs, yi_ys))

        # 통합 장애물 (빨강 큰 점)
        if obs:
            ox = np.array([-cy for _, cy, _ in obs])
            oy = np.array([cx for cx, _, _ in obs])
            self.obs_pts.set_offsets(np.c_[ox, oy])
        else:
            self.obs_pts.set_offsets(np.empty((0, 2)))

        # 좌/우 2차식 라인 (BEV: 가로=-y, 세로=x)
        if ll_xs.size >= 2:
            self.lane_left_line.set_data(-ll_ys, ll_xs)
        else:
            self.lane_left_line.set_data([], [])
        if lr_xs.size >= 2:
            self.lane_right_line.set_data(-lr_ys, lr_xs)
        else:
            self.lane_right_line.set_data([], [])

        # 중심선 path + 샘플 점
        if c_xs.size >= 2:
            self.center_line.set_data(-c_ys, c_xs)
            self.center_pts.set_offsets(np.c_[-c_ys, c_xs])
        else:
            self.center_line.set_data([], [])
            self.center_pts.set_offsets(np.empty((0, 2)))

        # /target 별 표시
        if target is not None:
            tx, ty = target
            self.target_marker.set_data([-ty], [tx])
        else:
            self.target_marker.set_data([], [])

        cls_summary = (",".join(f"{k}:{v}" for k, v in sorted(cls_hist.items()))
                       if cls_hist else "(empty)")
        self._status.set_text(
            f"scan raw:   {scan_xs.size:5d} pts\n"
            f"cls hist:   {cls_summary}\n"
            f"yellow MID: {fy_xs.size:5d} pts\n"
            f"white LANE: {fw_xs.size:5d} pts\n"
            f"unknown:    {fo_xs.size:5d} pts\n"
            f"yel inlier: {yi_xs.size:5d} pts\n"
            f"stop/goal:  {fs_xs.size:3d}/{fg_xs.size:3d}\n"
            f"fused obs:  {len(obs):5d}\n"
            f"center:     {c_xs.size:5d} samp\n"
            f"target:     {'on' if target else 'off':>5s}")

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
