#!/usr/bin/env python3
"""
키보드 제어 노드 — 자동/수동 토글 + 수동 모터 제어.

Q: auto_mode 토글 (ON↔OFF) → /auto_mode (Bool) 발행
수동(OFF)일 때:
  W/↑: 속도 +    S/↓: 속도 -
  A/←: 좌회전    D/→: 우회전
  Space: 정지 (speed=0, angle=0)
  E: speed만 0 (조향 유지)

/xycar_motor 직접 발행 (수동 시).
"""

import sys
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from xycar_msgs.msg import XycarMotor

# ======================== 수동 조작 파라미터 ========================

SPEED_STEP = 1.0
ANGLE_STEP = 5.0
SPEED_MAX = 50.0
ANGLE_MAX = 50.0
MOTOR_HZ = 10

HELP_TEXT = """
========================================
  키보드 제어 노드
========================================
  Q       : 자동/수동 토글
  W / ↑   : 속도 +
  S / ↓   : 속도 -
  A / ←   : 좌회전
  D / →   : 우회전
  Space   : 정지
  E       : speed=0 (조향 유지)
  C       : 트랙 폭 측정 (정지 상태에서)
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

        self._pub_auto = self.create_publisher(Bool, "/auto_mode", 10)
        self._pub_motor = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self._pub_measure = self.create_publisher(Empty, "/measure_width", 10)

        self.create_timer(1.0 / MOTOR_HZ, self._publish_motor)

        self._pub_auto_msg(False)
        self.get_logger().info("key_control_node started — press Q to toggle auto")

    def _pub_auto_msg(self, val):
        msg = Bool()
        msg.data = val
        self._pub_auto.publish(msg)

    def _publish_motor(self):
        if self._auto_mode:
            return
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self._speed)
        msg.angle = float(self._angle)
        self._pub_motor.publish(msg)

    def handle_key(self, key):
        if key in ('q', 'Q'):
            self._auto_mode = not self._auto_mode
            self._pub_auto_msg(self._auto_mode)
            label = "AUTO ON" if self._auto_mode else "MANUAL"
            self.get_logger().info(label)
            print(f"\r  >>> {label}          ")
            if not self._auto_mode:
                self._speed = 0.0
                self._angle = 0.0
            return

        if key in ('c', 'C'):
            self._pub_measure.publish(Empty())
            print("\r  >>> MEASURING TRACK WIDTH...          ")
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

        print(f"\r  speed={self._speed:+6.1f}  angle={self._angle:+6.1f}    ", end="", flush=True)


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
            if key == '\x03':  # Ctrl+C
                break
            if key:
                node.handle_key(key)
    except Exception:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
