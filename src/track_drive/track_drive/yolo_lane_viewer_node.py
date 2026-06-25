#!/usr/bin/env python3
"""
YOLO seg 차선 전용 BEV 뷰어 — OpenCV 해석 없이 /fused/lane(seg 점) 그대로 표시.

표시:
  /fused/lane   점 — z=cls로 색 구분 (8=노란중앙, 6=흰외곽, 16=노란점선, 2=child)
  /center_path  최종 경로 (초록)
  /lane_left    순수 큐빅 중앙선 (파랑, lane_curve가 발행)
  /lane_right   주행선 스무딩 전 (주황, lane_curve가 발행)
  /target       타겟점 (빨강 원)

전방 12m, 좌우 ±5m (실제 검출 범위 전체).
"""

import math

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, PointStamped
from sensor_msgs.msg import LaserScan

IMG_W = 600
IMG_H = 840
X_RANGE = (0.0, 12.0)    # 전방 (m)
Y_RANGE = (-5.0, 5.0)    # 좌우 (m)
SCALE_X = IMG_H / (X_RANGE[1] - X_RANGE[0])
SCALE_Y = IMG_W / (Y_RANGE[1] - Y_RANGE[0])

# cls_id → (BGR 색, 이름)
CLS_STYLE = {
    8:  ((0, 220, 220), "MID(8)"),     # 노란 중앙선
    16: ((0, 140, 255), "DASH(16)"),   # 노란 점선
    6:  ((255, 255, 255), "LANE(6)"),  # 흰 외곽선
    2:  ((160, 160, 160), "CHILD(2)"),  # child lane
}


def _to_px(x, y):
    px_x = int((Y_RANGE[1] - y) * SCALE_Y)
    px_y = int((X_RANGE[1] - x) * SCALE_X)
    return px_x, px_y


class YoloLaneViewer(Node):
    def __init__(self):
        super().__init__("yolo_lane_viewer")
        self._lane = []       # (x, y, cls)
        self._center = []
        self._left = []
        self._right = []
        self._target = None
        self._scan = []        # (x,y) 라이다 점 (lidar_frame)

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseArray, "/center_path", self._on_center, 10)
        self.create_subscription(PoseArray, "/lane_left", self._on_left, 10)
        self.create_subscription(PoseArray, "/lane_right", self._on_right, 10)
        self.create_subscription(PointStamped, "/target", self._on_target, 10)

        self.create_timer(0.05, self._draw)   # 20Hz
        self.get_logger().info("yolo_lane_viewer started (seg lane data)")

    def _on_lane(self, msg):
        self._lane = [(p.position.x, p.position.y, int(p.position.z)) for p in msg.poses]

    def _on_center(self, msg):
        self._center = [(p.position.x, p.position.y) for p in msg.poses]

    def _on_left(self, msg):
        self._left = [(p.position.x, p.position.y) for p in msg.poses]

    def _on_right(self, msg):
        self._right = [(p.position.x, p.position.y) for p in msg.poses]

    def _on_target(self, msg):
        self._target = (msg.point.x, msg.point.y)

    def _on_scan(self, msg):
        pts = []
        ang = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and 0.05 < r < 15.0:
                pts.append((r * math.cos(ang), r * math.sin(ang)))
            ang += msg.angle_increment
        self._scan = pts

    def _poly(self, img, pts, color, thick=2):
        for i in range(len(pts) - 1):
            cv2.line(img, _to_px(*pts[i]), _to_px(*pts[i + 1]), color, thick)

    def _draw(self):
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # 그리드 (2m 간격)
        for x in range(0, 13, 2):
            _, py = _to_px(x, 0)
            cv2.line(img, (0, py), (IMG_W, py), (35, 35, 35), 1)
            cv2.putText(img, f"{x}m", (4, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        for y in range(-4, 5, 2):
            px, _ = _to_px(0, y)
            cv2.line(img, (px, 0), (px, IMG_H), (35, 35, 35), 1)

        # 차량
        cx, cy = _to_px(0, 0)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        # 라이다 raw 점 (회색)
        for x, y in self._scan:
            px, py = _to_px(x, y)
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                cv2.circle(img, (px, py), 1, (90, 90, 90), -1)

        # 추월 왼쪽 감지 영역 (path_planner OT_LEFT_LIDAR_*: 90도±8.5도, 0.5~3.0m)
        for ang in (math.pi / 2 - 0.15, math.pi / 2 + 0.15):
            p0 = _to_px(0.5 * math.cos(ang), 0.5 * math.sin(ang))
            p1 = _to_px(3.0 * math.cos(ang), 3.0 * math.sin(ang))
            cv2.line(img, p0, p1, (0, 165, 255), 1)
        cv2.putText(img, "L-detect", _to_px(2.6, 2.2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # seg 차선 점 (cls 색)
        counts = {}
        for x, y, c in self._lane:
            color = CLS_STYLE.get(c, ((120, 120, 120), str(c)))[0]
            px, py = _to_px(x, y)
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                cv2.circle(img, (px, py), 3, color, -1)
            counts[c] = counts.get(c, 0) + 1

        # 큐빅 중앙선(파랑) / 주행선(주황) / 최종경로(초록)
        self._poly(img, self._left, (255, 120, 0), 1)
        self._poly(img, self._right, (0, 140, 255), 1)
        self._poly(img, self._center, (0, 255, 0), 2)        # 레퍼런스(초록)

        # 타겟
        if self._target is not None:
            cv2.circle(img, _to_px(*self._target), 5, (0, 0, 255), 2)

        # 범례 + 개수
        y0 = 18
        for c, (color, name) in CLS_STYLE.items():
            n = counts.get(c, 0)
            cv2.putText(img, f"{name}: {n}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y0 += 18
        cv2.putText(img, "center_path(ref)", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.imshow("yolo_lane_viewer", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloLaneViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
