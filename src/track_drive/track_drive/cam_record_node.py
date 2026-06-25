#!/usr/bin/env python3
"""
전방 카메라 녹화 + 키보드 주행 노드 — 라벨링용.

이 터미널에 포커스 두고 운전하면서 동시에 녹화. (motion/path_planner는 끄고 실행)

키보드:
  ↑ / ↓   : 전진 / 후진 (설정속도)
  ← / →   : 좌 / 우 조향 최대 (떼면 자동 직진 복귀)
  W / S   : 설정속도 +/-
  X       : 조향만 0
  Space   : 정지 (속도+조향 0)
  R       : 녹화 ON/OFF
  P       : 스냅샷 1장
  Q/Ctrl+C: 종료

저장: ~/xycar_ws/recordings/session_<시각>/frame_000123.jpg (5fps 기본)
"""

import os
import sys
import time
import tty
import termios
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor

# 주행 파라미터
SPEED_STEP = 1.0
ANGLE_MAX = 100.0
DRIVE_SPEED_DEFAULT = 10.0
STEER_HOLD_TICKS = 20       # 조향키 뗀 후 직진 복귀까지 (키 auto-repeat 초기지연 보완, 1초)
MOTOR_HZ = 20

HELP = """
========================================
  카메라 녹화 + 키보드 주행 (라벨링용)
========================================
  ↑/↓ : 전진/후진     ←/→ : 조향(자동복귀)
  W/S : 속도 +/-       X : 조향0   Space : 정지
  R   : 녹화 ON/OFF    P : 스냅샷   Q : 종료
========================================
"""


def get_key(settings):
    """1글자 + 방향키(이스케이프 시퀀스) 처리."""
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


def _to_bgr(msg):
    enc = msg.encoding.lower()
    try:
        if enc == "rgb8":
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        if enc == "bgr8":
            return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        if enc == "rgba8":
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 4)
            return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
        if enc == "mono8":
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    except ValueError:
        return None
    return None


class CamRecordNode(Node):
    def __init__(self):
        super().__init__("cam_record_node")

        self.declare_parameter("save_fps", 5.0)
        self.declare_parameter("save_dir",
                               os.path.join(os.path.expanduser("~"), "xycar_ws", "recordings"))
        self.declare_parameter("jpg_quality", 92)

        self._save_interval = 1.0 / max(0.5, float(self.get_parameter("save_fps").value))
        self._root = str(self.get_parameter("save_dir").value)
        self._jpgq = int(self.get_parameter("jpg_quality").value)
        os.makedirs(self._root, exist_ok=True)

        # 녹화 상태
        self._latest = None
        self._recording = False
        self._session_dir = None
        self._count = 0
        self._last_save = 0.0

        # 주행 상태
        self._speed = 0.0
        self._angle = 0.0
        self._drive_speed = DRIVE_SPEED_DEFAULT
        self._steer_hold = 0

        self.create_subscription(
            Image, "/usb_cam/image_raw/front", self._on_image, qos_profile_sensor_data)
        self._pub_motor = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self.create_timer(1.0 / MOTOR_HZ, self._publish_motor)

        self.get_logger().info(f"cam_record_node started. save_dir={self._root}")

    # ---- 녹화 ----
    def _on_image(self, msg):
        self._latest = msg
        if not self._recording:
            return
        now = time.time()
        if now - self._last_save < self._save_interval:
            return
        self._last_save = now
        self._save_frame(msg)

    def _save_frame(self, msg):
        bgr = _to_bgr(msg)
        if bgr is None:
            return
        if self._session_dir is None:
            base = os.path.join(self._root, "session_" + time.strftime("%H%M%S"))
            d = base
            k = 1
            while os.path.exists(d):      # 같은 초 중복 방지
                d = f"{base}_{k}"
                k += 1
            os.makedirs(d, exist_ok=True)
            self._session_dir = d
        path = os.path.join(self._session_dir, f"frame_{self._count:06d}.jpg")
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpgq])
        self._count += 1

    def toggle_record(self):
        if self._recording:
            self._recording = False
            print(f"\r  >>> REC OFF — {self._count} frames @ {self._session_dir}        ")
        else:
            self._session_dir = None
            self._count = 0
            self._last_save = 0.0
            self._recording = True
            print(f"\r  >>> REC ON ({1.0/self._save_interval:.0f} fps)        ")

    def snapshot(self):
        if self._latest is None:
            print("\r  >>> no frame yet        ")
            return
        snap_dir = os.path.join(self._root, "snaps")
        os.makedirs(snap_dir, exist_ok=True)
        bgr = _to_bgr(self._latest)
        if bgr is None:
            return
        ts = time.strftime("%H%M%S_") + f"{int(time.time()*1000) % 1000:03d}"
        path = os.path.join(snap_dir, f"snap_{ts}.jpg")
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpgq])
        print(f"\r  >>> SNAP saved: {path}        ")

    # ---- 주행 ----
    def _publish_motor(self):
        if self._steer_hold > 0:
            self._steer_hold -= 1
            if self._steer_hold == 0:
                self._angle = 0.0
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self._speed)
        msg.angle = float(self._angle)
        self._pub_motor.publish(msg)

    def drive_key(self, k):
        if k == 'UP':
            self._speed = self._drive_speed
        elif k == 'DOWN':
            self._speed = -self._drive_speed
        elif k == 'LEFT':
            self._angle = -ANGLE_MAX
            self._steer_hold = STEER_HOLD_TICKS
        elif k == 'RIGHT':
            self._angle = ANGLE_MAX
            self._steer_hold = STEER_HOLD_TICKS
        elif k in ('w', 'W'):
            self._drive_speed = min(self._drive_speed + SPEED_STEP, 50.0)
            if self._speed > 0:
                self._speed = self._drive_speed
        elif k in ('s', 'S'):
            self._drive_speed = max(self._drive_speed - SPEED_STEP, 0.0)
            if self._speed > 0:
                self._speed = self._drive_speed
        elif k in ('x', 'X'):
            self._angle = 0.0
            self._steer_hold = 0
        elif k == ' ':
            self._speed = 0.0
            self._angle = 0.0
            self._steer_hold = 0
        else:
            return
        rec = " [REC]" if self._recording else ""
        print(f"\r  set={self._drive_speed:4.1f} speed={self._speed:+5.1f} "
              f"angle={self._angle:+6.1f}{rec}      ", end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = CamRecordNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    settings = termios.tcgetattr(sys.stdin)
    print(HELP)
    try:
        while rclpy.ok():
            k = get_key(settings)
            if k == '\x03' or k in ('q', 'Q'):
                break
            if k in ('r', 'R'):
                node.toggle_record()
            elif k in ('p', 'P'):
                node.snapshot()
            else:
                node.drive_key(k)
    except Exception:
        pass
    finally:
        if node._recording:
            print(f"\r  >>> stopped. {node._count} frames @ {node._session_dir}")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
