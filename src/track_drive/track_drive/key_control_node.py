#!/usr/bin/env python3
"""
키보드 제어 노드 — 자동/수동 토글 + 수동 모터 제어 + 조향 측정.

수동 모드:
  W/S: 속도 ±    A/D: 조향 ±5    Space: 정지    E: speed=0
  1~6: angle 고정 (20/40/60/80/100/0) — 조향 측정용
  M: 기록 ON/OFF — odom+cmd를 txt에 저장
  C: 트랙 폭 측정    Q: auto 토글
"""

import sys
import os
import time
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from xycar_msgs.msg import XycarMotor

# ======================== 파라미터 ========================

SPEED_STEP = 1.0
ANGLE_STEP = 5.0
SPEED_MAX = 50.0
ANGLE_MAX = 100.0
MOTOR_HZ = 10
RECORD_HZ = 10

ANGLE_PRESETS = {'1': 20.0, '2': 40.0, '3': 60.0, '4': 80.0, '5': 100.0, '6': 0.0}

HELP_TEXT = """
========================================
  키보드 제어 노드
========================================
  Q       : 자동/수동 토글
  W / ↑   : 속도 +
  S / ↓   : 속도 -
  A / ←   : 좌회전 +5
  D / →   : 우회전 -5
  Space   : 정지
  E       : speed=0 (조향 유지)
  1~5     : angle 고정 (20/40/60/80/100)
  6       : angle = 0 (직진)
  M       : 기록 ON/OFF (조향 측정)
  C       : 트랙 폭 측정
  Ctrl+C  : 종료
========================================
"""

# ======================== 키 읽기 ========================

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    try:
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(ch3, '')
        return ch
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


# ======================== ROS 노드 ========================

class KeyControlNode(Node):
    def __init__(self):
        super().__init__("key_control_node")

        self._auto_mode = False
        self._speed = 0.0
        self._angle = 0.0

        self._recording = False
        self._record_file = None
        self._odom_v = 0.0
        self._odom_omega = 0.0
        self._imu_yaw_rate = 0.0
        self._imu_ax = 0.0
        self._imu_ay = 0.0

        self._pub_auto = self.create_publisher(Bool, "/auto_mode", 10)
        self._pub_motor = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self._pub_measure = self.create_publisher(Empty, "/measure_width", 10)

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Imu, "/imu", self._on_imu, 10)

        self.create_timer(1.0 / MOTOR_HZ, self._publish_motor)
        self.create_timer(1.0 / RECORD_HZ, self._record_tick)

        self._pub_auto_msg(False)
        self.get_logger().info("key_control_node started")

    def _pub_auto_msg(self, val):
        msg = Bool()
        msg.data = val
        self._pub_auto.publish(msg)

    def _on_odom(self, msg: Odometry):
        self._odom_v = msg.twist.twist.linear.x
        self._odom_omega = msg.twist.twist.angular.z

    def _on_imu(self, msg: Imu):
        self._imu_yaw_rate = msg.angular_velocity.z
        self._imu_ax = msg.linear_acceleration.x
        self._imu_ay = msg.linear_acceleration.y

    def _publish_motor(self):
        if self._auto_mode:
            return
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self._speed)
        msg.angle = float(self._angle)
        self._pub_motor.publish(msg)

    def _record_tick(self):
        if not self._recording or self._record_file is None:
            return
        v = self._odom_v
        w = self._odom_omega
        iw = self._imu_yaw_rate
        R_odom = v / w if abs(w) > 0.001 else float('inf')
        R_imu = v / iw if abs(iw) > 0.001 else float('inf')
        line = (f"{time.time():.3f}\t"
                f"{self._angle:+7.1f}\t{self._speed:+6.1f}\t"
                f"{v:+8.4f}\t{w:+8.4f}\t{R_odom:+10.3f}\t"
                f"{iw:+8.4f}\t{R_imu:+10.3f}\t"
                f"{self._imu_ax:+8.4f}\t{self._imu_ay:+8.4f}\n")
        self._record_file.write(line)
        self._record_file.flush()

    def _toggle_record(self):
        if self._recording:
            self._recording = False
            if self._record_file:
                path = self._record_file.name
                self._record_file.close()
                self._record_file = None
                print(f"\r  >>> RECORD OFF — saved: {path}          ")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/xycar_ws/steer_log_{ts}.txt")
            self._record_file = open(path, 'w')
            self._record_file.write(
                "# time\tcmd_angle\tcmd_speed\todom_v\todom_w\tR_odom\timu_yaw_rate\tR_imu\timu_ax\timu_ay\n")
            self._recording = True
            print(f"\r  >>> RECORD ON — {path}          ")

    def handle_key(self, key):
        if key in ('q', 'Q'):
            self._auto_mode = not self._auto_mode
            self._pub_auto_msg(self._auto_mode)
            label = "AUTO ON" if self._auto_mode else "MANUAL"
            print(f"\r  >>> {label}          ")
            if not self._auto_mode:
                self._speed = 0.0
                self._angle = 0.0
            return

        if key in ('c', 'C'):
            self._pub_measure.publish(Empty())
            print("\r  >>> MEASURING TRACK WIDTH...          ")
            return

        if key in ('m', 'M'):
            self._toggle_record()
            return

        if key in ANGLE_PRESETS:
            self._angle = ANGLE_PRESETS[key]
            print(f"\r  >>> angle FIXED = {self._angle:+.0f}          ")
            self._show_state()
            return

        if self._auto_mode:
            return

        if key in ('w', 'W', 'UP'):
            self._speed = min(self._speed + SPEED_STEP, SPEED_MAX)
        elif key in ('s', 'S', 'DOWN'):
            self._speed = max(self._speed - SPEED_STEP, -SPEED_MAX)
        elif key in ('a', 'A', 'LEFT'):
            self._angle = min(self._angle + ANGLE_STEP, ANGLE_MAX)
        elif key in ('d', 'D', 'RIGHT'):
            self._angle = max(self._angle - ANGLE_STEP, -ANGLE_MAX)
        elif key == ' ':
            self._speed = 0.0
            self._angle = 0.0
        elif key in ('e', 'E'):
            self._speed = 0.0

        self._show_state()

    def _show_state(self):
        rec = " [REC]" if self._recording else ""
        print(f"\r  speed={self._speed:+6.1f}  angle={self._angle:+6.1f}{rec}    ",
              end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = KeyControlNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    settings = termios.tcgetattr(sys.stdin)
    print(HELP_TEXT)

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key == '\x03':
                break
            if key:
                node.handle_key(key)
    except Exception:
        pass
    finally:
        if node._recording and node._record_file:
            node._record_file.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
