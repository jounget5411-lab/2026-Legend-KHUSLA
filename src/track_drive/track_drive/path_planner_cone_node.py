#!/usr/bin/env python3
"""
라바콘 경로계획 노드 — 왼쪽 콘 줄 피팅 + 오프셋 방식.

친구 차선 로직과 동일 구조:
  왼쪽 콘 줄 = 노란 중앙선 역할 → 2차 피팅 → +TRACK_HALF_WIDTH 오프셋 = center_path
  EMA 스무딩으로 안정화. miss 카운터로 일시 누락 유지.

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

# 콘 ROI
CONE_X_MIN = 0.3
CONE_X_MAX = 7.0
CONE_Y_MIN = -6.0          # 곡선에서 양쪽 콘 다 잡히게 넓게
CONE_Y_MAX = 6.0

# 트랙 폭
TRACK_WIDTH = 4.4
TRACK_HALF_WIDTH = TRACK_WIDTH / 2.0   # 2.2m — 왼쪽 줄에서 중앙까지

# 피팅
FIT_MIN_POINTS = 2
FIT_MIN_X_SPAN = 0.8
FIT_CURVE_MAX = 1.5        # |a| 제한 (2차 계수)
FIT_SLOPE_MAX = 3.0        # |b| 제한

# 왼쪽 줄 판별: 이전 왼쪽 피팅에서 이 거리 이내면 왼쪽 콘으로 인정
LEFT_ACCEPT_DIST = 1.5     # (m) — 트랙폭 4.4m의 ~1/3, 오른쪽 콘은 2.2m+ 떨어져 걸러짐

# 왼쪽 콘끼리 x 간격이 이 이상이면 먼 콘 버림 (끊긴 줄 잇지 않음)
LEFT_MAX_X_GAP = 4.0       # (m)

# 스무딩
FIT_SMOOTH_ALPHA = 0.20    # 새 피팅 반영 비율
FIT_MAX_MISS = 8           # 피팅 실패 시 이전 유지 최대 프레임

# 중앙선 샘플
SAMPLE_X_START = 0.5
SAMPLE_X_END = 6.0
SAMPLE_N = 25

# 목표점
TARGET_X = 3.0             # 친구 motion과 맞춤
TARGET_Y_LIMIT = 4.0

# 폭 측정
WIDTH_MEASURE_XS = [1.0, 2.0, 3.0, 4.0, 5.0]
WIDTH_COLLECT_FRAMES = 15

# ======================== 헬퍼 ========================

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
        self._prev_left_fit = None     # 왼쪽 줄 EMA 피팅 계수
        self._miss_count = 0
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
            f"path_planner_cone (left-fit+offset, width={TRACK_WIDTH:.1f}m, auto=OFF)")

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

        # ROI 필터
        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size > 0:
            roi = ((all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
                   & (all_y >= CONE_Y_MIN) & (all_y <= CONE_Y_MAX))
            all_x, all_y = all_x[roi], all_y[roi]

        # 왼쪽 줄 추출
        left_fit = None
        if all_x.size >= FIT_MIN_POINTS:
            if self._prev_left_fit is not None:
                # 이전 왼쪽 줄에서 가까운 콘만 왼쪽으로 인정
                expected_y = np.polyval(self._prev_left_fit, all_x)
                left_mask = np.abs(all_y - expected_y) < LEFT_ACCEPT_DIST
            else:
                # 첫 프레임: y>0 을 왼쪽으로 (직선 출발)
                left_mask = all_y > 0

            lx, ly = all_x[left_mask], all_y[left_mask]

            # x 정렬 후, 인접 콘 간 gap > 4m 이면 먼 콘 버림
            if lx.size >= 2:
                order = np.argsort(lx)
                lx, ly = lx[order], ly[order]
                gaps = np.diff(lx)
                cut = np.where(gaps > LEFT_MAX_X_GAP)[0]
                if cut.size > 0:
                    keep = cut[0] + 1  # 첫 번째 큰 gap 앞까지만
                    lx, ly = lx[:keep], ly[:keep]

            if (lx.size >= FIT_MIN_POINTS
                    and float(np.max(lx) - np.min(lx)) >= FIT_MIN_X_SPAN):
                try:
                    w = 1.0 / (1.0 + lx * lx)
                    coef = np.polyfit(lx, ly, 2, w=w)
                    a, b, _ = coef
                    if abs(a) <= FIT_CURVE_MAX and abs(b) <= FIT_SLOPE_MAX:
                        left_fit = coef
                except (np.linalg.LinAlgError, ValueError):
                    pass

        # EMA 스무딩 + miss 유지 (친구 _fit_yellow_sliding 구조)
        if left_fit is None:
            self._miss_count += 1
            if self._prev_left_fit is not None and self._miss_count <= FIT_MAX_MISS:
                left_fit = self._prev_left_fit
            else:
                self._prev_left_fit = None
                return
        else:
            if self._prev_left_fit is None:
                self._prev_left_fit = np.asarray(left_fit, dtype=np.float64)
            else:
                self._prev_left_fit = (
                    (1.0 - FIT_SMOOTH_ALPHA) * self._prev_left_fit
                    + FIT_SMOOTH_ALPHA * np.asarray(left_fit, dtype=np.float64))
            left_fit = self._prev_left_fit
            self._miss_count = 0

        # 중앙선 = 왼쪽 줄 - TRACK_HALF_WIDTH
        center_coef = left_fit.copy()
        center_coef[2] -= TRACK_HALF_WIDTH

        left_ys = np.polyval(left_fit, sample_xs)
        center_ys = np.polyval(center_coef, sample_xs)
        right_ys = center_ys - TRACK_HALF_WIDTH

        center_ys = np.clip(center_ys, -TARGET_Y_LIMIT, TARGET_Y_LIMIT)

        # 발행
        self._pub_left.publish(_poses_from_xy(stamp, sample_xs, left_ys))
        self._pub_right.publish(_poses_from_xy(stamp, sample_xs, right_ys))
        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = "lidar_frame"
        t.point.x = float(TARGET_X)
        t.point.y = target_y
        self._pub_target.publish(t)

        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            n_cones = all_x.size
            self.get_logger().info(
                f"cones={n_cones} left_fit={'ok' if left_fit is not None else 'miss'} "
                f"target_y={target_y:+.2f}")

    def _tick_width_measure(self):
        if self._width_collecting <= 0:
            return
        all_x = np.array([o[0] for o in self._obstacles])
        all_y = np.array([o[1] for o in self._obstacles])
        if all_x.size < 4:
            self._width_collecting -= 1
            return
        roi = (all_x >= CONE_X_MIN) & (all_x <= CONE_X_MAX)
        all_x, all_y = all_x[roi], all_y[roi]
        if all_x.size >= 4:
            y_med = np.median(all_y)
            ly = all_y[all_y >= y_med]
            ry = all_y[all_y < y_med]
            if ly.size >= 1 and ry.size >= 1:
                w = float(np.mean(ly) - np.mean(ry))
                self._width_samples.append(w)
        self._width_collecting -= 1
        if self._width_collecting == 0:
            if self._width_samples:
                avg_w = float(np.mean(self._width_samples))
                self.get_logger().info(
                    f"track width: {avg_w:.2f} m (half: {avg_w/2:.2f} m), "
                    f"{len(self._width_samples)} frames")
            else:
                self.get_logger().warn("width measurement FAILED")


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
