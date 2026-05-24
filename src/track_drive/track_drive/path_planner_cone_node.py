#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 두 줄 콘 사이 중앙 경로 생성.

핵심: 좌/우 콘을 트랙폭만큼 평행이동해 합친 뒤 피팅 → 점 수 2배, 곡선 피팅 가능.

구독: /fused/obstacles (PoseArray), /auto_mode (Bool), /measure_width (Empty)
발행: /center_path, /target, /lane_left, /lane_right

좌표계: lidar_frame (X=전방+, Y=좌+, 미터)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 파라미터 (튜닝 가능) ========================

PLAN_HZ = 10              # 계획 주기 (Hz)

# 콘 ROI — 피팅에 쓸 전방 범위
CONE_X_MIN = 0.3           # 전방 최소 (m)
CONE_X_MAX = 7.0           # 전방 최대 (m)

# 피팅 최소 점수 (합친 후 기준)
FIT_MIN_PTS_QUAD = 4       # 2차 피팅
FIT_MIN_PTS_LIN = 2        # 1차 피팅

# 트랙 폭
TRACK_WIDTH = 4.4          # 좌 콘 줄 ~ 우 콘 줄 전체 폭 (m, 실측)
TRACK_HALF_WIDTH = TRACK_WIDTH / 2.0  # 2.2m

# 중앙선 샘플
SAMPLE_X_START = 0.5       # 샘플 시작 전방 (m)
SAMPLE_X_END = 6.0         # 샘플 끝 전방 (m)
SAMPLE_N = 25              # 샘플 점 개수

# 목표점
TARGET_FORWARD_M = 1.5     # 중앙선에서 목표점 전방 거리 (m)

# 이전 경로 유지 (콘 없을 때)
MEMORY_MAX_TICKS = 15      # 이전 경로를 유지할 최대 틱 (1.5초)

# 트랙 폭 측정 (C키 트리거)
WIDTH_MEASURE_XS = [1.0, 2.0, 3.0, 4.0, 5.0]
WIDTH_COLLECT_FRAMES = 15

# ======================== 헬퍼 ========================

def _fit_curve(xs, ys):
    """점들을 2차(충분하면) 또는 1차로 피팅. (차수, coef) 또는 None."""
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


def _merge_cones(lx, ly, rx, ry):
    """좌/우 콘을 중앙선 기준으로 합침.

    좌 콘: y를 -TRACK_HALF_WIDTH 해서 중앙 기준으로 이동
    우 콘: y를 +TRACK_HALF_WIDTH 해서 중앙 기준으로 이동
    합친 점들은 "중앙선"의 x, y를 나타냄.
    """
    merged_x = np.concatenate([lx, rx])
    merged_y = np.concatenate([ly - TRACK_HALF_WIDTH, ry + TRACK_HALF_WIDTH])
    return merged_x, merged_y


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
            f"path_planner_cone_node started (track_width={TRACK_WIDTH:.1f}m, auto=OFF)")

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
        obs = self._obstacles

        # ROI 필터
        all_x = np.array([o[0] for o in obs])
        all_y = np.array([o[1] for o in obs])
        if all_x.size > 0:
            roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
            all_x, all_y = all_x[roi], all_y[roi]

        # 좌/우 분리
        left_mask = all_y > 0.1
        right_mask = all_y < -0.1
        lx, ly = all_x[left_mask], all_y[left_mask]
        rx, ry = all_x[right_mask], all_y[right_mask]

        n_left, n_right = len(lx), len(rx)
        have_any = n_left >= 1 or n_right >= 1

        if not have_any:
            self._stale_ticks += 1
            if self._prev_center is not None and self._stale_ticks < MEMORY_MAX_TICKS:
                sample_xs, center_ys = self._prev_center
                self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))
                idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
                self._publish_target(stamp, sample_xs[idx], center_ys[idx])
            return

        # 좌/우 합쳐서 중앙선 피팅 (핵심)
        merged_x, merged_y = _merge_cones(lx, ly, rx, ry)
        fit_result = _fit_curve(merged_x, merged_y)

        sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        if fit_result is not None:
            deg, coef = fit_result
            center_ys = np.polyval(coef, sample_xs)

            # 좌/우 곡선 복원 (중앙 + half_width)
            l_ys = center_ys + TRACK_HALF_WIDTH
            r_ys = center_ys - TRACK_HALF_WIDTH
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
        elif n_left >= 1 and n_right < 1:
            center_ys = np.full(SAMPLE_N, float(np.mean(ly)) - TRACK_HALF_WIDTH)
            deg = 0
        elif n_right >= 1 and n_left < 1:
            center_ys = np.full(SAMPLE_N, float(np.mean(ry)) + TRACK_HALF_WIDTH)
            deg = 0
        else:
            return

        self._prev_center = (sample_xs, center_ys)
        self._stale_ticks = 0

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        self._publish_target(stamp, sample_xs[idx], center_ys[idx])

        # 주기 로그 (1초마다)
        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            self.get_logger().info(
                f"cone L{n_left}/R{n_right} → merged {len(merged_x)} pts, "
                f"fit deg={deg}")

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
        lx, ly = all_x[all_y > 0.1], all_y[all_y > 0.1]
        rx, ry = all_x[all_y < -0.1], all_y[all_y < -0.1]
        if len(lx) >= FIT_MIN_PTS_LIN and len(rx) >= FIT_MIN_PTS_LIN:
            try:
                lc = np.polyfit(lx, ly, 1)
                rc = np.polyfit(rx, ry, 1)
                widths = [np.polyval(lc, x) - np.polyval(rc, x)
                          for x in WIDTH_MEASURE_XS]
                self._width_samples.append(
                    (float(np.mean(widths)), len(lx), len(rx)))
            except (np.linalg.LinAlgError, ValueError):
                pass
        self._width_collecting -= 1
        if self._width_collecting == 0:
            if self._width_samples:
                ws = [s[0] for s in self._width_samples]
                avg_w = float(np.mean(ws))
                half = avg_w / 2.0
                nl = self._width_samples[-1][1]
                nr = self._width_samples[-1][2]
                self.get_logger().info(
                    f"track width: {avg_w:.2f} m (half: {half:.2f} m), "
                    f"L {nl} pts / R {nr} pts, {len(ws)} frames averaged")
            else:
                self.get_logger().warn(
                    "width measurement FAILED — not enough cones on both sides. "
                    "Align car in straight section with cones visible on both sides.")


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
