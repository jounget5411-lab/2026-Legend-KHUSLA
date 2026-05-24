#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 라인 추적(tracking) 기반 좌우 분류.

콘 좌우를 y부호가 아니라 "이전 좌/우 라인과의 거리"로 배정.
한 번 오른쪽으로 잡힌 라인은 곡선에서도 오른쪽 유지.

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
PATH_SMOOTH_ALPHA = 0.7

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


# ======================== 콘→좌/우 배정 (라인 추적) ========================

def _assign_cones_by_tracking(all_x, all_y, prev_left_coef, prev_right_coef):
    """이전 좌/우 라인 coef 기준으로 각 콘을 가까운 쪽에 배정.

    이전 라인 없으면 y부호로 초기 분류 (첫 프레임 전용).
    """
    lx, ly, rx, ry = [], [], [], []

    have_prev = prev_left_coef is not None and prev_right_coef is not None

    for x, y in zip(all_x, all_y):
        if have_prev:
            dl = abs(y - np.polyval(prev_left_coef, x))
            dr = abs(y - np.polyval(prev_right_coef, x))
            if dl < dr:
                lx.append(x); ly.append(y)
            else:
                rx.append(x); ry.append(y)
        else:
            if y > 0:
                lx.append(x); ly.append(y)
            else:
                rx.append(x); ry.append(y)

    return (np.array(lx), np.array(ly), np.array(rx), np.array(ry))


# ======================== ROS 노드 ========================

class PathPlannerConeNode(Node):
    def __init__(self):
        super().__init__("path_planner_cone_node")

        self._auto = False
        self._obstacles = []
        self._prev_center_ys = None
        self._prev_left_coef = None
        self._prev_right_coef = None
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

        self._sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info(
            f"path_planner_cone (tracking, width={TRACK_WIDTH:.1f}m, "
            f"smooth={PATH_SMOOTH_ALPHA}, auto=OFF)")

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
        sample_xs = self._sample_xs

        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size > 0:
            roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
            all_x, all_y = all_x[roi], all_y[roi]

        if all_x.size == 0:
            self._do_memory(stamp)
            return

        # ---- 콘 좌/우 배정 (라인 추적) ----
        lx, ly, rx, ry = _assign_cones_by_tracking(
            all_x, all_y, self._prev_left_coef, self._prev_right_coef)

        n_left, n_right = len(lx), len(rx)

        # ---- 좌/우 각각 피팅 ----
        l_fit = _fit_curve(lx, ly) if n_left >= FIT_MIN_PTS_LIN else None
        r_fit = _fit_curve(rx, ry) if n_right >= FIT_MIN_PTS_LIN else None

        have_left = l_fit is not None
        have_right = r_fit is not None

        if have_left and have_right:
            l_deg, l_coef = l_fit
            r_deg, r_coef = r_fit
            l_ys = np.polyval(l_coef, sample_xs)
            r_ys = np.polyval(r_coef, sample_xs)
            raw_center_ys = (l_ys + r_ys) / 2.0
            self._prev_left_coef = l_coef
            self._prev_right_coef = r_coef
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
            mode = f"both L{n_left}/R{n_right} deg={l_deg},{r_deg}"

        elif have_left:
            _, l_coef = l_fit
            l_ys = np.polyval(l_coef, sample_xs)
            raw_center_ys = l_ys - TRACK_HALF_WIDTH
            self._prev_left_coef = l_coef
            if self._prev_right_coef is None:
                self._prev_right_coef = np.array([c for c in l_coef])
                self._prev_right_coef[-1] -= TRACK_WIDTH
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
            r_ys = raw_center_ys - TRACK_HALF_WIDTH
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
            mode = f"left-only L{n_left}"

        elif have_right:
            _, r_coef = r_fit
            r_ys = np.polyval(r_coef, sample_xs)
            raw_center_ys = r_ys + TRACK_HALF_WIDTH
            self._prev_right_coef = r_coef
            if self._prev_left_coef is None:
                self._prev_left_coef = np.array([c for c in r_coef])
                self._prev_left_coef[-1] += TRACK_WIDTH
            l_ys = raw_center_ys + TRACK_HALF_WIDTH
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
            mode = f"right-only R{n_right}"

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

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))
        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        self._publish_target(stamp, sample_xs[idx], center_ys[idx])

        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            trk = "tracked" if self._prev_left_coef is not None else "init"
            self.get_logger().info(f"{mode} [{trk}]")

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
