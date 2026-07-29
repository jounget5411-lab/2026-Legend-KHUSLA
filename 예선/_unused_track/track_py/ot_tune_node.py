#!/usr/bin/env python3
"""
추월 차선이동(pursuit) 튜닝 키보드 노드.

/manual_offset:    노란 중앙선 기준 횡오프셋 (path_planner가 LANE에서 적용)
/manual_lookahead: 조향 lookahead 거리 override (motion이 STEER_FOCUS_X 대신 사용)

직선에서 우1.5 <-> 좌1.5 이동시키며 lookahead 바꿔 수렴 튜닝.

키:
  r : 우 1.5m (시작 위치)
  l : 좌 1.5m (이동)
  k : lookahead +0.2m (더 멀리)
  j : lookahead -0.2m (더 가까이)
  Ctrl+C : 종료
둘 다 0.5초 끊기면 자동 해제(중앙/기본 lookahead).
"""

import sys
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

RIGHT = -1.5
LEFT = 1.5
LA_STEP = 0.2
PUB_HZ = 20

HELP = """
========================================
  추월 pursuit 튜닝 (ot_tune)
========================================
  r : 우 1.5m (시작)
  l : 좌 1.5m (이동)
  k : lookahead +0.2
  j : lookahead -0.2
  Ctrl+C : 종료
========================================
"""


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    try:
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


class OtTuneNode(Node):
    def __init__(self):
        super().__init__("ot_tune_node")
        self._offset = 0.0
        self._lookahead = 2.5
        self._pub_off = self.create_publisher(Float32, "/manual_offset", 10)
        self._pub_la = self.create_publisher(Float32, "/manual_lookahead", 10)
        self.create_timer(1.0 / PUB_HZ, self._tick)
        self.get_logger().info("ot_tune_node started")

    def _tick(self):
        self._pub_off.publish(Float32(data=float(self._offset)))
        self._pub_la.publish(Float32(data=float(self._lookahead)))

    def handle_key(self, key):
        if key in ('r', 'R'):
            self._offset = RIGHT
        elif key in ('l', 'L'):
            self._offset = LEFT
        elif key in ('k', 'K'):
            self._lookahead += LA_STEP
        elif key in ('j', 'J'):
            self._lookahead = max(0.5, self._lookahead - LA_STEP)
        else:
            return
        print(f"\r  offset={self._offset:+.2f}m  lookahead={self._lookahead:.1f}m    ",
              end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = OtTuneNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    settings = termios.tcgetattr(sys.stdin)
    print(HELP)
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
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
