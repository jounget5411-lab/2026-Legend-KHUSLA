#!/usr/bin/env python3
"""
모터 제어 노드 — /center_path 다점 추종으로 /xycar_motor 발행.

구독:
  /center_path (PoseArray, lidar_frame) — 중심선 샘플들
  /target      (PointStamped) — 단일 타겟 (fallback / 활성신호)
발행:
  /xycar_motor (XycarMotor)

test_viewer.compute_steering_angle 이식: 다점 pursuit + heading 가중평균.
slope는 샘플 ys의 finite difference로 추정.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseArray
from xycar_msgs.msg import XycarMotor

# ======================== 제어 파라미터 ========================

CONTROL_HZ = 10

# 다점 추종 게인 — 조향 lookahead(STEER_FOCUS_X) 주변 점에 가우시안 가중
# 직선 와리가리를 잡기 위해 pursuit/heading 게인을 보수적으로.
STEERING_TARGET_GAIN = 1.30
STEERING_CURVE_GAIN = 1.20
STEER_FOCUS_X = 3.0      # 가중 peak 위치 (m)
STEER_FOCUS_SIGMA = 2.0  # 가중 폭 (m, σ) — 곡선 진입을 미리 보도록 약간 넓게

ANGLE_MIN = -80.0
ANGLE_MAX = 80.0
# 응답 빠르게 하려고 α↑ (한 프레임에 변화의 75% 반영).
ANGLE_SMOOTH_ALPHA = 0.75
ANGLE_MAX_STEP = 16.0
# 작은 path 떨림은 컷해서 직선 진동 줄임.
ANGLE_DEADBAND = 0.6

# 속도
SPEED_DEFAULT = 5.0
SPEED_MIN = 2.0
SPEED_STOP = 0.0
SPEED_SLOW_START_DEG = 12.0
SPEED_SLOW_FULL_DEG = 40.0

TARGET_TIMEOUT_S = 0.5
TARGET_Y_LIMIT = 4.0
TARGET_X = 3.0
LOOKAHEAD_X_MIN = 2.0
LOOKAHEAD_X_MAX = 6.0


# ======================== ROS 노드 ========================

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        self._path_xs = None
        self._path_ys = None
        self._path_stamp = None
        self._target_stamp = None

        self._prev_angle = 0.0

        self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
        self.create_subscription(PointStamped, "/target", self._on_target, 10)
        self._pub = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._tick)

        self.get_logger().info("motion_node started (centerline-pursuit)")

    def _on_path(self, msg: PoseArray):
        if not msg.poses:
            return
        self._path_xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
        self._path_ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)
        self._path_stamp = self.get_clock().now()

    def _on_target(self, msg: PointStamped):
        self._target_stamp = self.get_clock().now()

    def _tick(self):
        now = self.get_clock().now()

        # /center_path 또는 /target 둘 다 끊겼으면 정지
        stamps = [s for s in (self._path_stamp, self._target_stamp) if s is not None]
        if not stamps:
            self._publish_motor(SPEED_STOP, 0.0)
            return
        newest_age = min((now - s).nanoseconds * 1e-9 for s in stamps)
        if newest_age > TARGET_TIMEOUT_S:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        if self._path_xs is None or self._path_xs.size < 2:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        angle = self._compute_steering(self._path_xs, self._path_ys)
        angle = self._smooth_angle(angle)
        speed = self._speed_from_angle(angle)
        self._publish_motor(speed, angle)

    def _compute_steering(self, xs, ys):
        lookahead = (xs >= LOOKAHEAD_X_MIN) & (xs <= LOOKAHEAD_X_MAX)
        if int(np.count_nonzero(lookahead)) >= 2:
            xs = xs[lookahead]
            ys = ys[lookahead]
        else:
            near = xs <= LOOKAHEAD_X_MAX
            if int(np.count_nonzero(near)) >= 2:
                xs = xs[near]
                ys = ys[near]

        ys = np.clip(ys, -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        pursuit_angles = np.degrees(np.arctan2(-ys, np.maximum(xs, 1e-3)))

        if xs.size >= 2:
            slopes = np.gradient(ys, xs)
        else:
            slopes = np.zeros_like(xs)
        heading_angles = np.degrees(np.arctan(-slopes))

        sample_angles = (
            STEERING_TARGET_GAIN * pursuit_angles +
            STEERING_CURVE_GAIN * heading_angles
        )
        weights = np.exp(-((xs - STEER_FOCUS_X) / STEER_FOCUS_SIGMA) ** 2)
        angle = float(np.average(sample_angles, weights=weights))
        return float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    def _smooth_angle(self, angle):
        delta = float(angle - self._prev_angle)
        if abs(delta) < ANGLE_DEADBAND:
            angle = self._prev_angle
        else:
            angle = self._prev_angle + ANGLE_SMOOTH_ALPHA * delta
        step = float(np.clip(angle - self._prev_angle, -ANGLE_MAX_STEP, ANGLE_MAX_STEP))
        angle = float(np.clip(self._prev_angle + step, ANGLE_MIN, ANGLE_MAX))
        self._prev_angle = angle
        return angle

    def _speed_from_angle(self, angle):
        a = abs(angle)
        if a <= SPEED_SLOW_START_DEG:
            return SPEED_DEFAULT
        t = min(1.0, (a - SPEED_SLOW_START_DEG) / (SPEED_SLOW_FULL_DEG - SPEED_SLOW_START_DEG))
        return max(SPEED_MIN, SPEED_DEFAULT - (SPEED_DEFAULT - SPEED_MIN) * t)

    def _publish_motor(self, speed, angle):
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(speed)
        msg.angle = float(max(ANGLE_MIN, min(ANGLE_MAX, angle)))
        self._pub.publish(msg)


# ======================== main ========================

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
