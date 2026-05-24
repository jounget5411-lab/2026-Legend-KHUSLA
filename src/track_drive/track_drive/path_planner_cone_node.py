#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 게이트(콘 쌍) 기반 중앙 경로 생성.

좌/우 판단을 y부호가 아니라 "트랙 중앙선" 기준으로 한다:
- 게이트 있으면 → 게이트 중점이 중앙 기준
- 게이트 없고 싱글만 → 이전 프레임 중앙선이 기준
→ 곡선에서 우측 콘이 y>0에 와도 정확히 판단.

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

CONE_X_MIN = 0.3
CONE_X_MAX = 7.0

TRACK_WIDTH = 4.4
TRACK_HALF_WIDTH = TRACK_WIDTH / 2.0

GATE_X_TOL = 1.5
GATE_WIDTH_MIN = 3.0
GATE_WIDTH_MAX = 6.0

FIT_MIN_PTS_QUAD = 4
FIT_MIN_PTS_LIN = 2

SAMPLE_X_START = 0.5
SAMPLE_X_END = 6.0
SAMPLE_N = 25

TARGET_FORWARD_M = 1.5

MEMORY_MAX_TICKS = 15

WIDTH_MEASURE_XS = [1.0, 2.0, 3.0, 4.0, 5.0]
WIDTH_COLLECT_FRAMES = 15

# ======================== 게이트 매칭 ========================

def _find_gates_and_singles(xs, ys):
    n = len(xs)
    if n == 0:
        return [], []

    order = np.argsort(xs)
    sx, sy = xs[order], ys[order]

    used = [False] * n
    gates = []

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
                break
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
                ya, yb = yb, ya
            gates.append((mx, my, ya, yb))

    singles = [(sx[i], sy[i]) for i in range(n) if not used[i]]
    return gates, singles


def _center_y_at(coef, x):
    """이전 중앙선 계수로 특정 x에서의 중앙 y를 구함."""
    if coef is None:
        return 0.0
    return float(np.polyval(coef, x))


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
        self._prev_center = None        # (sample_xs, center_ys)
        self._prev_center_coef = None   # 이전 중앙선 피팅 계수 (좌우 판단 기준)
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

        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size > 0:
            roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
            all_x, all_y = all_x[roi], all_y[roi]

        if all_x.size == 0:
            self._do_memory(stamp)
            return

        gates, singles = _find_gates_and_singles(all_x, all_y)
        n_gates = len(gates)
        n_singles = len(singles)

        center_pts_x = []
        center_pts_y = []

        # --- 게이트 중점 ---
        for gx, gy_mid, ya, yb in gates:
            center_pts_x.append(gx)
            center_pts_y.append(gy_mid)

        # --- 싱글 콘: 중앙 기준으로 좌우 판단 ---
        if singles:
            if n_gates >= 1:
                # 게이트 중점 평균을 중앙 기준으로
                ref_coef = None
                ref_mean_y = np.mean(center_pts_y)
            elif self._prev_center_coef is not None:
                # 이전 프레임 중앙선을 기준으로
                ref_coef = self._prev_center_coef
                ref_mean_y = None
            else:
                # 기준 없음 — 싱글 스킵
                ref_coef = None
                ref_mean_y = None

            for sx_s, sy_s in singles:
                if ref_coef is not None:
                    center_y_here = _center_y_at(ref_coef, sx_s)
                elif ref_mean_y is not None:
                    center_y_here = ref_mean_y
                else:
                    continue  # 기준 없으면 이 싱글 무시

                if sy_s > center_y_here:
                    center_pts_x.append(sx_s)
                    center_pts_y.append(sy_s - TRACK_HALF_WIDTH)
                else:
                    center_pts_x.append(sx_s)
                    center_pts_y.append(sy_s + TRACK_HALF_WIDTH)

        cx_arr = np.array(center_pts_x)
        cy_arr = np.array(center_pts_y)

        if len(cx_arr) == 0:
            self._do_memory(stamp)
            return

        # 피팅
        fit_result = _fit_curve(cx_arr, cy_arr)
        sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        if fit_result is not None:
            deg, coef = fit_result
            center_ys = np.polyval(coef, sample_xs)
            self._prev_center_coef = coef
        elif len(cx_arr) >= 1:
            center_ys = np.full(SAMPLE_N, float(np.mean(cy_arr)))
            deg = 0
            self._prev_center_coef = np.array([float(np.mean(cy_arr))])
        else:
            return

        l_ys = center_ys + TRACK_HALF_WIDTH
        r_ys = center_ys - TRACK_HALF_WIDTH
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))

        self._prev_center = (sample_xs, center_ys)
        self._stale_ticks = 0

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        self._publish_target(stamp, sample_xs[idx], center_ys[idx])

        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            mode = "gate" if n_gates > 0 else "single+prev"
            self.get_logger().info(
                f"{mode}: gates={n_gates} singles={n_singles} → "
                f"center {len(cx_arr)} pts, fit deg={deg}")

    def _do_memory(self, stamp):
        self._stale_ticks += 1
        if self._prev_center is not None and self._stale_ticks < MEMORY_MAX_TICKS:
            sx, cy = self._prev_center
            self._pub_center.publish(_poses_from_xy(stamp, sx, cy))
            idx = int(np.argmin(np.abs(sx - TARGET_FORWARD_M)))
            self._publish_target(stamp, sx[idx], cy[idx])

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
