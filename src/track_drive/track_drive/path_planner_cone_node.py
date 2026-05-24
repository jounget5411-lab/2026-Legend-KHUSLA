#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 게이트(콘 쌍) 기반 중앙 경로 생성.

핵심: 비슷한 전방거리에서 트랙폭만큼 떨어진 두 콘을 쌍(게이트)으로 묶고,
게이트 중점 + 싱글 콘 오프셋을 합쳐 중앙선 피팅.
Y 부호 기반 좌우 분리 없음 → 곡선에서도 안정.

구독: /fused/obstacles, /auto_mode, /measure_width
발행: /center_path, /target, /lane_left, /lane_right

좌표계: lidar_frame (X=전방+, Y=좌+, 미터)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 파라미터 (튜닝 가능) ========================

PLAN_HZ = 10

# 콘 ROI
CONE_X_MIN = 0.3
CONE_X_MAX = 7.0

# 트랙 폭 (실측)
TRACK_WIDTH = 4.4
TRACK_HALF_WIDTH = TRACK_WIDTH / 2.0

# 게이트 매칭
GATE_X_TOL = 1.5           # 같은 게이트로 묶을 전방(x) 차이 최대 (m)
GATE_WIDTH_MIN = 3.0       # 게이트 폭 최소 (m) — 너무 가까운 쌍 제외
GATE_WIDTH_MAX = 6.0       # 게이트 폭 최대 (m) — 너무 먼 쌍 제외

# 피팅
FIT_MIN_PTS_QUAD = 4
FIT_MIN_PTS_LIN = 2

# 중앙선 샘플
SAMPLE_X_START = 0.5
SAMPLE_X_END = 6.0
SAMPLE_N = 25

# 목표점
TARGET_FORWARD_M = 1.5

# 이전 경로 유지
MEMORY_MAX_TICKS = 15

# 폭 측정 (C키)
WIDTH_MEASURE_XS = [1.0, 2.0, 3.0, 4.0, 5.0]
WIDTH_COLLECT_FRAMES = 15

# ======================== 게이트 매칭 ========================

def _find_gates_and_singles(xs, ys):
    """콘들을 게이트(쌍)와 싱글로 분류.

    Returns: gate_centers [(x, y_mid)], singles [(x, y)]
    """
    n = len(xs)
    if n == 0:
        return [], []

    # x로 정렬
    order = np.argsort(xs)
    sx, sy = xs[order], ys[order]

    used = [False] * n
    gates = []  # (x_mid, y_mid, y_a, y_b)

    # 모든 쌍 후보를 x 차이순으로 시도
    for i in range(n):
        if used[i]:
            continue
        best_j = -1
        best_width = 0.0
        for j in range(i + 1, n):
            if used[j]:
                continue
            dx = abs(sx[j] - sx[i])
            if dx > GATE_X_TOL:
                break  # x 정렬이라 이후는 더 멀어짐
            dy = abs(sy[j] - sy[i])
            if GATE_WIDTH_MIN <= dy <= GATE_WIDTH_MAX:
                if best_j < 0 or abs(dy - TRACK_WIDTH) < abs(best_width - TRACK_WIDTH):
                    best_j = j
                    best_width = dy
        if best_j >= 0:
            used[i] = True
            used[best_j] = True
            mx = (sx[i] + sx[best_j]) / 2.0
            my = (sy[i] + sy[best_j]) / 2.0
            ya, yb = sy[i], sy[best_j]
            if ya < yb:
                ya, yb = yb, ya  # ya = 좌측(y+), yb = 우측(y-)
            gates.append((mx, my, ya, yb))

    singles = [(sx[i], sy[i]) for i in range(n) if not used[i]]
    return gates, singles


# ======================== 헬퍼 ========================

def _fit_curve(xs, ys):
    n = len(xs)
    if n >= FIT_MIN_PTS_QUAD:
        try:
            return 2, np.polyfit(xs, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if n >= FIT_MIN_PTS_LIN:
        try:
            return 1, np.polyfit(xs, ys, 1)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return None


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


# ======================== ROS 노드 ========================

class PathPlannerConeNode(Node):
    def __init__(self):
        super().__init__("path_planner_cone_node")

        self._auto = False
        self._obstacles = []
        self._prev_center = None
        self._stale_ticks = 999
        self._width_collecting = 0
        self._width_samples = []
        self._log_counter = 0

        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(Bool, "/auto_mode", self._on_auto, 10)
        self.create_subscription(Empty, "/measure_width", self._on_measure, 10)

        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info(
            f"path_planner_cone (gate mode, width={TRACK_WIDTH:.1f}m, auto=OFF)")

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y, p.position.z)
                           for p in msg.poses]

    def _on_auto(self, msg: Bool):
        if self._auto != msg.data:
            self._auto = msg.data
            self.get_logger().info(f"auto_mode = {'ON' if self._auto else 'OFF'}")

    def _on_measure(self, msg):
        self._width_collecting = WIDTH_COLLECT_FRAMES
        self._width_samples = []
        self.get_logger().info("width measurement started — collecting frames...")

    def _tick(self):
        self._tick_width_measure()
        if not self._auto:
            return

        stamp = self.get_clock().now().to_msg()

        # ROI 필터
        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size > 0:
            roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
            all_x, all_y = all_x[roi], all_y[roi]

        if all_x.size == 0:
            self._stale_ticks += 1
            if self._prev_center is not None and self._stale_ticks < MEMORY_MAX_TICKS:
                sx, cy = self._prev_center
                self._pub_center.publish(_poses_from_xy(stamp, sx, cy))
                idx = int(np.argmin(np.abs(sx - TARGET_FORWARD_M)))
                self._publish_target(stamp, sx[idx], cy[idx])
            return

        # 게이트 매칭
        gates, singles = _find_gates_and_singles(all_x, all_y)
        n_gates = len(gates)

        # 중앙선 점 수집: 게이트 중점 + 싱글 오프셋
        center_pts_x = []
        center_pts_y = []
        left_pts_x = []
        left_pts_y = []
        right_pts_x = []
        right_pts_y = []

        for gx, gy_mid, ya, yb in gates:
            center_pts_x.append(gx)
            center_pts_y.append(gy_mid)
            left_pts_x.append(gx)
            left_pts_y.append(ya)
            right_pts_x.append(gx)
            right_pts_y.append(yb)

        # 싱글: 게이트가 있으면 중앙선 추세에서 어느 쪽인지 추정, 없으면 스킵
        if n_gates >= 1 and singles:
            gate_center_mean_y = np.mean(center_pts_y)
            for sx_s, sy_s in singles:
                if sy_s > gate_center_mean_y:
                    # 좌측 콘 → 중앙 = sy - half_width
                    center_pts_x.append(sx_s)
                    center_pts_y.append(sy_s - TRACK_HALF_WIDTH)
                    left_pts_x.append(sx_s)
                    left_pts_y.append(sy_s)
                else:
                    # 우측 콘 → 중앙 = sy + half_width
                    center_pts_x.append(sx_s)
                    center_pts_y.append(sy_s + TRACK_HALF_WIDTH)
                    right_pts_x.append(sx_s)
                    right_pts_y.append(sy_s)

        cx_arr = np.array(center_pts_x)
        cy_arr = np.array(center_pts_y)

        if len(cx_arr) == 0:
            self._stale_ticks += 1
            if self._prev_center is not None and self._stale_ticks < MEMORY_MAX_TICKS:
                sx, cy = self._prev_center
                self._pub_center.publish(_poses_from_xy(stamp, sx, cy))
                idx = int(np.argmin(np.abs(sx - TARGET_FORWARD_M)))
                self._publish_target(stamp, sx[idx], cy[idx])
            return

        # 피팅
        fit_result = _fit_curve(cx_arr, cy_arr)
        sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        if fit_result is not None:
            deg, coef = fit_result
            center_ys = np.polyval(coef, sample_xs)
        elif len(cx_arr) >= 1:
            center_ys = np.full(SAMPLE_N, float(np.mean(cy_arr)))
            deg = 0
        else:
            return

        # 좌/우 곡선 복원
        l_ys = center_ys + TRACK_HALF_WIDTH
        r_ys = center_ys - TRACK_HALF_WIDTH
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))

        self._prev_center = (sample_xs, center_ys)
        self._stale_ticks = 0

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        self._publish_target(stamp, sample_xs[idx], center_ys[idx])

        # 로그 (1초마다)
        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            n_singles = len(singles)
            self.get_logger().info(
                f"gates={n_gates} singles={n_singles} → "
                f"center {len(cx_arr)} pts, fit deg={deg}")

    def _publish_target(self, stamp, x, y):
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = "lidar_frame"
        t.point.x = float(x)
        t.point.y = float(y)
        self._pub_target.publish(t)

    def _tick_width_measure(self):
        if self._width_collecting <= 0:
            return
        obs = self._obstacles
        all_x = np.array([o[0] for o in obs])
        all_y = np.array([o[1] for o in obs])
        if all_x.size == 0:
            self._width_collecting -= 1
            return
        roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
        all_x, all_y = all_x[roi], all_y[roi]
        gates, _ = _find_gates_and_singles(all_x, all_y)
        if gates:
            widths = [abs(ya - yb) for _, _, ya, yb in gates]
            self._width_samples.append((float(np.mean(widths)), len(gates)))
        self._width_collecting -= 1
        if self._width_collecting == 0:
            if self._width_samples:
                ws = [s[0] for s in self._width_samples]
                avg_w = float(np.mean(ws))
                total_gates = sum(s[1] for s in self._width_samples)
                self.get_logger().info(
                    f"track width: {avg_w:.2f} m (half: {avg_w/2:.2f} m), "
                    f"{total_gates} gates over {len(ws)} frames")
            else:
                self.get_logger().warn(
                    "width measurement FAILED — no cone gates found.")


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerConeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
