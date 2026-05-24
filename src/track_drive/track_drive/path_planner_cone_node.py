#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 게이트(콘 쌍) + 한쪽 콘 오프셋 + EMA 안정화.

- 게이트 있으면 → 중점 중앙선
- 게이트 없고 싱글만 → 콘 줄 피팅 + 2.2m 오프셋 (방향은 직전 상태 유지)
- 매 프레임 중앙선을 이전과 EMA 평균 → 촐랑댐 제거

구독: /fused/obstacles, /auto_mode, /measure_width
발행: /center_path, /target, /lane_left, /lane_right

좌표계: lidar_frame (X=전방+, Y=좌+, 미터)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 파라미터 ========================

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

# EMA 안정화: 이전 경로 비중 (0.7 = 이전 70% + 새 30%)
PATH_SMOOTH_ALPHA = 0.7

# 한쪽 콘 방향: +1 = 콘이 좌측(중앙 = 콘 - half), -1 = 콘이 우측(중앙 = 콘 + half)
# 게이트에서 자동 갱신, 싱글 구간에서는 고정 유지

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
        self._prev_center_ys = None     # EMA 평균용
        self._stale_ticks = 999
        self._single_side = +1          # +1=콘이 좌측, -1=콘이 우측. 게이트에서 갱신.
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

        self._sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info(
            f"path_planner_cone (width={TRACK_WIDTH:.1f}m, smooth={PATH_SMOOTH_ALPHA}, auto=OFF)")

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
        self.get_logger().info("width measurement started...")

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
        sample_xs = self._sample_xs

        raw_center_ys = None
        mode = ""

        if n_gates >= 1:
            # ---- 게이트 중점 + 싱글 합침 ----
            cx_list = [gx for gx, _, _, _ in gates]
            cy_list = [gy for _, gy, _, _ in gates]
            gate_center_mean = np.mean(cy_list)

            # 게이트에서 싱글 방향 갱신
            for _, _, ya, yb in gates:
                self._single_side = +1 if ya > yb else -1

            for sx_s, sy_s in singles:
                if sy_s > gate_center_mean:
                    cy_list.append(sy_s - TRACK_HALF_WIDTH)
                else:
                    cy_list.append(sy_s + TRACK_HALF_WIDTH)
                cx_list.append(sx_s)

            fit = _fit_curve(np.array(cx_list), np.array(cy_list))
            if fit is not None:
                deg, coef = fit
                raw_center_ys = np.polyval(coef, sample_xs)
                mode = f"gate({n_gates})+s({n_singles}) deg={deg}"
            else:
                raw_center_ys = np.full(SAMPLE_N, float(np.mean(cy_list)))
                mode = f"gate({n_gates}) mean"

        elif n_singles >= FIT_MIN_PTS_LIN:
            # ---- 싱글만: 콘 줄 피팅 + 방향 고정 오프셋 ----
            s_x = np.array([s[0] for s in singles])
            s_y = np.array([s[1] for s in singles])
            fit = _fit_curve(s_x, s_y)
            if fit is not None:
                deg, coef = fit
                cone_ys = np.polyval(coef, sample_xs)
            else:
                cone_ys = np.full(SAMPLE_N, float(np.mean(s_y)))
                deg = 0

            if self._single_side > 0:
                raw_center_ys = cone_ys - TRACK_HALF_WIDTH
            else:
                raw_center_ys = cone_ys + TRACK_HALF_WIDTH
            side_label = "L" if self._single_side > 0 else "R"
            mode = f"single({n_singles},{side_label}) deg={deg}"

        else:
            self._do_memory(stamp)
            return

        # ---- EMA 안정화 ----
        if self._prev_center_ys is not None and len(self._prev_center_ys) == SAMPLE_N:
            center_ys = PATH_SMOOTH_ALPHA * self._prev_center_ys + \
                        (1.0 - PATH_SMOOTH_ALPHA) * raw_center_ys
        else:
            center_ys = raw_center_ys

        self._prev_center_ys = center_ys
        self._stale_ticks = 0

        # ---- 발행 ----
        l_ys = center_ys + TRACK_HALF_WIDTH
        r_ys = center_ys - TRACK_HALF_WIDTH
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        self._publish_target(stamp, sample_xs[idx], center_ys[idx])

        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            self.get_logger().info(mode)

    def _do_memory(self, stamp):
        self._stale_ticks += 1
        if self._prev_center_ys is not None and self._stale_ticks < MEMORY_MAX_TICKS:
            sample_xs = self._sample_xs
            self._pub_center.publish(_poses_from_xy(stamp, sample_xs, self._prev_center_ys))
            idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
            self._publish_target(stamp, sample_xs[idx], self._prev_center_ys[idx])

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
        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
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
                self.get_logger().warn("width measurement FAILED — no gates found.")


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
