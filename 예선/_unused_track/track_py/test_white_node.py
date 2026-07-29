#!/usr/bin/env python3
"""
실험용 — 오른쪽 흰 실선 기반 주행 + 차선 변경 테스트.

기본: 오른쪽 흰 실선 +1.5m = 2차선 중앙
/lane_change Bool=True → 노란선 +1.5m = 1차선으로 이동
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from std_msgs.msg import Bool

from .lane_planner import PLAN_HZ, WHITE_CLS_IDS, YELLOW_CLS_IDS, TARGET_X, TARGET_Y_LIMIT

LIDAR_FRAME = "lidar_frame"

RIGHT_LINE_OFFSET = 1.5     # 오른쪽 실선 → 2차선 중앙
YELLOW_LINE_OFFSET = 1.5    # 노란 중앙선 → 1차선 중앙 (왼쪽)
FIT_MIN_POINTS = 5
X_BIN_SIZE = 0.5
X_MIN = 0.5
X_MAX = 8.0
SAMPLE_N = 20
OUTLIER_THRESHOLD = 0.4
EMA_ALPHA = 0.15


def _poses_from_xy(stamp, xs, ys):
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = LIDAR_FRAME
    for x, y in zip(xs, ys):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        msg.poses.append(p)
    return msg


def _extract_right_line(xs, ys):
    """x 구간별 최소 Y(가장 오른쪽) 점을 모아 오른쪽 실선 추출."""
    bins = np.arange(X_MIN, X_MAX, X_BIN_SIZE)
    rx, ry = [], []
    for bx in bins:
        mask = (xs >= bx) & (xs < bx + X_BIN_SIZE)
        if not mask.any():
            continue
        idx = np.argmin(ys[mask])
        bin_xs = xs[mask]
        bin_ys = ys[mask]
        rx.append(float(bin_xs[idx]))
        ry.append(float(bin_ys[idx]))
    return np.array(rx), np.array(ry)


def _fit_with_outlier_rejection(xs, ys, prev_fit):
    """prev_fit 기준 아웃라이어 제거 후 2차 피팅."""
    if prev_fit is not None and xs.size >= 3:
        expected = np.polyval(prev_fit, xs)
        inlier = np.abs(ys - expected) < OUTLIER_THRESHOLD
        if inlier.sum() >= FIT_MIN_POINTS:
            xs, ys = xs[inlier], ys[inlier]

    if xs.size < FIT_MIN_POINTS:
        return None
    x_span = float(xs.max() - xs.min())
    if x_span < 1.0:
        return None
    try:
        deg = 2 if xs.size >= 6 and x_span >= 2.0 else 1
        coef = np.polyfit(xs, ys, deg)
        if deg == 1:
            coef = np.array([0.0, coef[0], coef[1]])
        return coef
    except (np.linalg.LinAlgError, ValueError):
        return None


def _ema_update(prev, new, alpha=EMA_ALPHA):
    if prev is not None:
        return (1.0 - alpha) * prev + alpha * new
    return new


class TestWhiteNode(Node):
    def __init__(self):
        super().__init__("test_white_node")

        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._yellow_xs = np.array([], dtype=np.float64)
        self._yellow_ys = np.array([], dtype=np.float64)
        self._has_lane = False

        self._right_fit = None    # 오른쪽 흰 실선 피팅
        self._yellow_fit = None   # 노란 중앙선 피팅
        self._lane = "2nd"        # "2nd" = 2차선(흰선 기반), "1st" = 1차선(노란선 기반)

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(Bool, "/lane_change", self._on_lane_change, 10)

        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_right = self.create_publisher(PoseArray, "/lane_right", 10)
        self._pub_left = self.create_publisher(PoseArray, "/lane_left", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self._log_counter = 0
        self.get_logger().info("test_white_node started (2nd lane, /lane_change to switch)")

    def _on_lane(self, msg: PoseArray):
        self._has_lane = True
        if not msg.poses:
            self._white_xs = np.array([], dtype=np.float64)
            self._white_ys = np.array([], dtype=np.float64)
            self._yellow_xs = np.array([], dtype=np.float64)
            self._yellow_ys = np.array([], dtype=np.float64)
            return
        xs = np.fromiter((p.position.x for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        ys = np.fromiter((p.position.y for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        cls = np.fromiter((int(p.position.z) for p in msg.poses),
                          dtype=np.int32, count=len(msg.poses))
        wm = np.isin(cls, list(WHITE_CLS_IDS))
        ym = np.isin(cls, list(YELLOW_CLS_IDS))
        self._white_xs = xs[wm]
        self._white_ys = ys[wm]
        self._yellow_xs = xs[ym]
        self._yellow_ys = ys[ym]

    def _on_lane_change(self, msg: Bool):
        if msg.data and self._lane == "2nd":
            self._lane = "1st"
            self.get_logger().info("LANE CHANGE → 1st lane (yellow + 1.5m)")
        elif not msg.data and self._lane == "1st":
            self._lane = "2nd"
            self.get_logger().info("LANE CHANGE → 2nd lane (right white + 1.5m)")

    def _tick(self):
        if not self._has_lane:
            return

        stamp = self.get_clock().now().to_msg()

        # 항상 양쪽 피팅 업데이트 (차선변경 시 바로 사용 가능)
        self._update_right_fit()
        self._update_yellow_fit()

        if self._lane == "2nd":
            fit = self._right_fit
            offset = RIGHT_LINE_OFFSET   # +1.5 (왼쪽으로)
            if fit is None:
                return
            center_coef = fit.copy()
            center_coef[2] += offset
        else:
            fit = self._yellow_fit
            offset = YELLOW_LINE_OFFSET  # +1.5 (왼쪽으로)
            if fit is None:
                return
            center_coef = fit.copy()
            center_coef[2] += offset

        sample_xs = np.linspace(X_MIN, X_MAX, SAMPLE_N)
        center_ys = np.clip(np.polyval(center_coef, sample_xs),
                            -TARGET_Y_LIMIT, TARGET_Y_LIMIT)

        self._pub_center.publish(_poses_from_xy(stamp, sample_xs, center_ys))

        # 시각화: 오른쪽 흰선 + 노란선
        if self._right_fit is not None:
            self._pub_right.publish(_poses_from_xy(
                stamp, sample_xs, np.polyval(self._right_fit, sample_xs)))
        if self._yellow_fit is not None:
            self._pub_left.publish(_poses_from_xy(
                stamp, sample_xs, np.polyval(self._yellow_fit, sample_xs)))

        target_y = float(np.clip(np.polyval(center_coef, TARGET_X),
                                 -TARGET_Y_LIMIT, TARGET_Y_LIMIT))
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = LIDAR_FRAME
        t.point.x = float(TARGET_X)
        t.point.y = target_y
        self._pub_target.publish(t)

        self._log_counter += 1
        if self._log_counter >= PLAN_HZ:
            self._log_counter = 0
            self.get_logger().info(
                f"lane={self._lane} target_y={target_y:.2f} "
                f"white={self._white_xs.size} yellow={self._yellow_xs.size}")

    def _update_right_fit(self):
        rx, ry = _extract_right_line(self._white_xs, self._white_ys)
        new_fit = _fit_with_outlier_rejection(rx, ry, self._right_fit)
        if new_fit is not None:
            self._right_fit = _ema_update(self._right_fit, new_fit)

    def _update_yellow_fit(self):
        xs, ys = self._yellow_xs, self._yellow_ys
        new_fit = _fit_with_outlier_rejection(xs, ys, self._yellow_fit)
        if new_fit is not None:
            self._yellow_fit = _ema_update(self._yellow_fit, new_fit)


def main(args=None):
    rclpy.init(args=args)
    node = TestWhiteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
