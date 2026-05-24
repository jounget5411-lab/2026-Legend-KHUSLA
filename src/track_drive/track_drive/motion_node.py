#!/usr/bin/env python3
"""
모터 제어 노드 — /target 받아 /xycar_motor 발행.

구독: /target (PointStamped, lidar_frame: x=전방, y=좌)
발행: /xycar_motor (XycarMotor: angle, speed)

파라미터 실행 중 변경: ros2 param set /motion_node steer_gain 300.0
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from xycar_msgs.msg import XycarMotor

# ======================== 기본값 ========================

CONTROL_HZ = 10
TARGET_TIMEOUT_S = 0.5

# ======================== ROS 노드 ========================

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        self.declare_parameter("steer_gain", 250.0)
        self.declare_parameter("speed_default", 5.0)
        self.declare_parameter("angle_max", 100.0)

        self._target_x = None
        self._target_y = None
        self._target_stamp = None

        self.create_subscription(PointStamped, "/target", self._on_target, 10)
        self._pub = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._tick)

        g = self.get_parameter("steer_gain").value
        s = self.get_parameter("speed_default").value
        a = self.get_parameter("angle_max").value
        self.get_logger().info(
            f"motion_node started (gain={g}, speed={s}, angle_max={a})")

    def _on_target(self, msg: PointStamped):
        self._target_x = msg.point.x
        self._target_y = msg.point.y
        self._target_stamp = self.get_clock().now()

    def _tick(self):
        now = self.get_clock().now()

        if self._target_stamp is None:
            self._publish_motor(0.0, 0.0)
            return

        age = (now - self._target_stamp).nanoseconds * 1e-9
        if age > TARGET_TIMEOUT_S:
            self._publish_motor(0.0, 0.0)
            return

        gain = self.get_parameter("steer_gain").value
        speed = self.get_parameter("speed_default").value
        a_max = self.get_parameter("angle_max").value

        heading_rad = math.atan2(-self._target_y, max(self._target_x, 0.3))
        angle = gain * heading_rad
        angle = max(-a_max, min(a_max, angle))

        self._publish_motor(speed, angle)

    def _publish_motor(self, speed, angle):
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(speed)
        msg.angle = float(angle)
        self._pub.publish(msg)


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
