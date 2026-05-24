#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 두 줄 콘 사이 중앙 경로 생성.

구독: /fused/obstacles (PoseArray), /auto_mode (Bool)
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

# 피팅
FIT_MIN_PTS_QUAD = 4       # 2차 피팅 최소 점수
FIT_MIN_PTS_LIN = 2        # 1차 피팅 최소 점수

# 트랙 폭 (한쪽만 있을 때 반대쪽 추정)
TRACK_HALF_WIDTH = 1.2     # 중앙 → 콘 줄까지 거리 (m)

# 중앙선 샘플
SAMPLE_X_START = 0.5       # 샘플 시작 전방 (m)
SAMPLE_X_END = 6.0         # 샘플 끝 전방 (m)
SAMPLE_N = 25              # 샘플 점 개수

# 목표점
TARGET_FORWARD_M = 1.5     # 중앙선에서 목표점 전방 거리 (m)

# 이전 경로 유지 (콘 없을 때)
MEMORY_MAX_TICKS = 15      # 이전 경로를 유지할 최대 틱 (1.5초)

# 트랙 폭 측정 (C키 트리거)
WIDTH_MEASURE_XS = [1.0, 2.0, 3.0, 4.0, 5.0]  # 폭 측정할 전방 거리들 (m)
WIDTH_COLLECT_FRAMES = 15  # C키 후 수집할 프레임 수 (1.5초)

# ======================== 헬퍼 ========================

def _fit_curve(xs, ys):
    """점들을 2차(충분하면) 또는 1차로 피팅. 실패 시 None."""
    n = len(xs)
    if n >= FIT_MIN_PTS_QUAD:
        try:
            return np.polyfit(xs, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if n >= FIT_MIN_PTS_LIN:
        try:
            return np.polyfit(xs, ys, 1)
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
        self._prev_center = None   # (sample_xs, center_ys) 캐시
        self._stale_ticks = 999
        self._width_collecting = 0
        self._width_samples = []

        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(Bool, "/auto_mode", self._on_auto, 10)
        self.create_subscription(Empty, "/measure_width", self._on_measure, 10)

        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info("path_planner_cone_node started (auto=OFF)")

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

        # 콘 ROI 필터 + 좌/우 분리
        all_x = np.array([o[0] for o in obs])
        all_y = np.array([o[1] for o in obs])
        if all_x.size > 0:
            roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
            all_x, all_y = all_x[roi], all_y[roi]

        left_mask = all_y > 0.1
        right_mask = all_y < -0.1
        lx, ly = all_x[left_mask], all_y[left_mask]
        rx, ry = all_x[right_mask], all_y[right_mask]

        l_coef = _fit_curve(lx, ly)
        r_coef = _fit_curve(rx, ry)

        have_left = l_coef is not None
        have_right = r_coef is not None

        sample_xs = np.linspace(SAMPLE_X_START, SAMPLE_X_END, SAMPLE_N)

        if have_left and have_right:
            l_ys = np.polyval(l_coef, sample_xs)
            r_ys = np.polyval(r_coef, sample_xs)
            center_ys = (l_ys + r_ys) / 2.0
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
        elif have_left:
            l_ys = np.polyval(l_coef, sample_xs)
            center_ys = l_ys - TRACK_HALF_WIDTH
            self._pub_left.publish(_poses_from_xy(stamp, sample_xs, l_ys))
        elif have_right:
            r_ys = np.polyval(r_coef, sample_xs)
            center_ys = r_ys + TRACK_HALF_WIDTH
            self._pub_right.publish(_poses_from_xy(stamp, sample_xs, r_ys))
        else:
            # 콘 없음 — 이전 경로 유지
            self._stale_ticks += 1
            if self._prev_center is not None and self._stale_ticks < MEMORY_MAX_TICKS:
                sample_xs, center_ys = self._prev_center
            else:
                return  # 데이터 없으면 발행 안 함
            # 이전 경로 그대로 발행 (아래로)

        if have_left or have_right:
            self._prev_center = (sample_xs, center_ys)
            self._stale_ticks = 0

        # /center_path 발행
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        # /target: 전방 TARGET_FORWARD_M 에 가장 가까운 점
        idx = int(np.argmin(np.abs(sample_xs - TARGET_FORWARD_M)))
        t_msg = PointStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = "lidar_frame"
        t_msg.point.x = float(sample_xs[idx])
        t_msg.point.y = float(center_ys[idx])
        self._pub_target.publish(t_msg)


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
