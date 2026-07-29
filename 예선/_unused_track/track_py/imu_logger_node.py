#!/usr/bin/env python3
"""IMU heading 로거 — yaw(deg)를 txt에 기록 (S자 구간 진입/끝 heading 파악용)."""
import os
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ImuLogger(Node):
    def __init__(self):
        super().__init__("imu_logger")
        self._yaw = 0.0
        self._have = False
        self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
        path = os.path.join(os.path.expanduser("~"), "xycar_ws", "imu_log.txt")
        self._f = open(path, "a")
        self._f.write("# --- new session ---\n# t_rel\tyaw_deg\n")
        self._f.flush()
        self._t0 = None
        self.create_timer(0.05, self._tick)
        self.get_logger().info("imu_logger started -> " + path)

    def _on_imu(self, msg):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.degrees(math.atan2(siny, cosy))
        self._have = True

    def _tick(self):
        if not self._have:
            return
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self._f.write("%.2f\t%+.2f\n" % (now - self._t0, self._yaw))
        self._f.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ImuLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._f.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
