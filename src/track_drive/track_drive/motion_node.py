#!/usr/bin/env python3
"""
모터 제어 노드 — /target 받아 /xycar_motor 발행.

구독: /target (PointStamped, lidar_frame: x=전방, y=좌)
발행: /xycar_motor (XycarMotor: angle, speed)

control() 함수에 제어 로직을 채우면 됨.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from xycar_msgs.msg import XycarMotor

# ======================== 제어 파라미터 ========================

CONTROL_HZ = 10         # 제어 루프 주기 (Hz)

# 여기에 제어 게인 — 나중에 PID로 교체
STEER_P = 50.0          # 비례 조향 게인 (deg per rad)

SPEED_DEFAULT = 5.0     # 기본 속도
SPEED_STOP = 0.0
ANGLE_MIN = -50.0
ANGLE_MAX = 50.0

TARGET_TIMEOUT_S = 0.5  # 이 시간 이상 목표 없으면 정지

# ======================== ROS 노드 ========================

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        self._target_x = None
        self._target_y = None
        self._target_stamp = None

        self.create_subscription(PointStamped, "/target", self._on_target, 10)
        self._pub = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._tick)

        self.get_logger().info("motion_node started (P-control placeholder)")

    def _on_target(self, msg: PointStamped):
        self._target_x = msg.point.x
        self._target_y = msg.point.y
        self._target_stamp = self.get_clock().now()

    def _tick(self):
        now = self.get_clock().now()

        if self._target_stamp is None:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        age = (now - self._target_stamp).nanoseconds * 1e-9
        if age > TARGET_TIMEOUT_S:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        angle, speed = control(self._target_x, self._target_y)
        self._publish_motor(speed, angle)

    def _publish_motor(self, speed, angle):
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(speed)
        msg.angle = float(max(ANGLE_MIN, min(ANGLE_MAX, angle)))
        self._pub.publish(msg)


# ======================== 제어 로직 ========================

def control(target_x, target_y):
    """목표점 → (angle, speed) 변환.

    Parameters
    ----------
    target_x : float — 목표 전방 거리 (m)
    target_y : float — 목표 좌우 거리 (m, 좌+)

    Returns
    -------
    (angle, speed)  — 조향각 (deg, 좌+), 속도

    ──────────────────────────────────────────────
    여기에 제어 로직을 채운다 — 나중에 PID로 교체.
    지금은 단순 P 제어: 목표 방향 각도에 비례해 조향.
    ──────────────────────────────────────────────
    """
    heading_rad = math.atan2(-target_y, max(target_x, 0.3))
    angle = STEER_P * heading_rad
    speed = SPEED_DEFAULT
    return angle, speed


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
