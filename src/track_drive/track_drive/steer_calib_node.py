#!/usr/bin/env python3
"""
조향 곡률 캘리브레이션 — 실제 회전반경 측정 + 참조 호 비교.

원리: 정상 선회 중 회전반경 R = v / ω (속도 / 각속도).
키보드로 최대 좌회전 하면서 측정된 R을 참조 호와 비교 → 최대조향 곡률 찾기.

발행:
  /steer_arc (PoseArray) — 측정된 실제 회전 호 (뷰어 자홍)
  /ref_arc   (PoseArray) — 참조 호 (param ref_radius, 뷰어 흰색)

파라미터:
  ref_radius : 비교용 참조 회전반경 (m, 기본 2.0) — 실시간 조절하며 맞추기
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from xycar_msgs.msg import XycarMotor

LIDAR_FRAME = "lidar_frame"
ARC_N = 30
ARC_FWD_MAX = 8.0
V_MIN = 0.2
W_MIN = 0.03


def _arc(radius, left, n=ARC_N):
    """반경 radius 원의 전방 호만 (x = 0~전방, y = R - sqrt(R²-x²)). 안 말림."""
    if radius <= 0.01:
        return np.array([]), np.array([])
    x_max = min(ARC_FWD_MAX, 0.95 * radius)   # x는 R 못 넘음
    x = np.linspace(0.0, x_max, n)
    y = radius - np.sqrt(np.maximum(radius * radius - x * x, 0.0))
    if not left:
        y = -y
    return x, y


class SteerCalibNode(Node):
    def __init__(self):
        super().__init__("steer_calib_node")
        self.declare_parameter("ref_radius", 4.0)
        self.declare_parameter("drive", False)        # True면 노드가 직접 원 주행 (키보드 불필요)
        self.declare_parameter("drive_angle", -100.0)  # 자체 주행 조향 (좌=-100)
        self.declare_parameter("drive_speed", 5.0)     # 자체 주행 속도

        self._v = 0.0
        self._w_odom = 0.0
        self._w_imu = 0.0
        self._R_ema = None

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
        self._pub_meas = self.create_publisher(PoseArray, "/steer_arc", 10)
        self._pub_ref = self.create_publisher(PoseArray, "/ref_arc", 10)
        self._pub_motor = self.create_publisher(XycarMotor, "/xycar_motor", 10)

        self.create_timer(0.1, self._tick)
        self.get_logger().info("steer_calib_node started (R = v/omega)")

    def _on_odom(self, msg):
        self._v = float(msg.twist.twist.linear.x)
        self._w_odom = float(msg.twist.twist.angular.z)

    def _on_imu(self, msg):
        self._w_imu = float(msg.angular_velocity.z)

    def _poses(self, xs, ys):
        m = PoseArray()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = LIDAR_FRAME
        for x, y in zip(xs, ys):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            m.poses.append(p)
        return m

    def _tick(self):
        ref = float(self.get_parameter("ref_radius").value)
        rx, ry = _arc(ref, left=True)
        self._pub_ref.publish(self._poses(rx, ry))

        # 자체 주행: 일정 조향+속도로 깔끔한 원 (키보드 떨림 제거)
        if bool(self.get_parameter("drive").value):
            m = XycarMotor()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "base_link"
            m.angle = float(self.get_parameter("drive_angle").value)
            m.speed = float(self.get_parameter("drive_speed").value)
            self._pub_motor.publish(m)

        w = self._w_odom if abs(self._w_odom) > 1e-4 else self._w_imu
        v = abs(self._v)

        if v < V_MIN or abs(w) < W_MIN:
            self._pub_meas.publish(self._poses([], []))
            return

        R = v / abs(w)
        # 무거운 평균 (정상 선회 정착값만 신뢰) — 떨림 제거
        self._R_ema = R if self._R_ema is None else 0.92 * self._R_ema + 0.08 * R
        Rs = self._R_ema
        left = w > 0
        mx, my = _arc(Rs, left=left)
        self._pub_meas.publish(self._poses(mx, my))

        cmp_txt = "측정이 더 급함(곡률 큼)" if Rs < ref else "측정이 더 완만(곡률 작음)"
        self.get_logger().info(
            f"measured R={Rs:.2f}m k={1.0/Rs:.3f} | ref R={ref:.2f}m -> {cmp_txt} "
            f"(v={v:.2f} w={w:+.2f} {'L' if left else 'R'})")


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
