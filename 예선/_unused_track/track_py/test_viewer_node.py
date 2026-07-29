#!/usr/bin/env python3
"""
실험용 BEV 뷰어 — 흰색 점 + 오른쪽 실선 피팅 + center path 시각화.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

from .lane_planner import WHITE_CLS_IDS, YELLOW_CLS_IDS

# BEV 이미지 파라미터
IMG_W = 600
IMG_H = 800
X_RANGE = (0.0, 10.0)   # 전방 (m)
Y_RANGE = (-5.0, 5.0)   # 좌우 (m)
SCALE_X = IMG_H / (X_RANGE[1] - X_RANGE[0])
SCALE_Y = IMG_W / (Y_RANGE[1] - Y_RANGE[0])


def _to_px(x, y):
    """lidar_frame (x=전방, y=좌) → 이미지 좌표 (px_x=좌우, px_y=위아래)."""
    px_x = int((Y_RANGE[1] - y) * SCALE_Y)
    px_y = int((X_RANGE[1] - x) * SCALE_X)
    return px_x, px_y


class TestViewerNode(Node):
    def __init__(self):
        super().__init__("test_viewer_node")

        self._white_pts = []
        self._yellow_pts = []
        self._right_pts = []
        self._center_pts = []

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/lane_right", self._on_right, 10)
        self.create_subscription(PoseArray, "/center_path", self._on_center, 10)

        self.create_timer(0.1, self._draw)  # 10Hz
        self.get_logger().info("test_viewer_node started")

    def _on_lane(self, msg: PoseArray):
        white, yellow = [], []
        for p in msg.poses:
            cls = int(p.position.z)
            pt = (p.position.x, p.position.y)
            if cls in WHITE_CLS_IDS:
                white.append(pt)
            elif cls in YELLOW_CLS_IDS:
                yellow.append(pt)
        self._white_pts = white
        self._yellow_pts = yellow

    def _on_right(self, msg: PoseArray):
        self._right_pts = [(p.position.x, p.position.y) for p in msg.poses]

    def _on_center(self, msg: PoseArray):
        self._center_pts = [(p.position.x, p.position.y) for p in msg.poses]

    def _draw(self):
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # 그리드
        for x in range(0, 11, 2):
            _, py = _to_px(x, 0)
            cv2.line(img, (0, py), (IMG_W, py), (30, 30, 30), 1)
            cv2.putText(img, f"{x}m", (5, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
        for y in range(-4, 5, 2):
            px, _ = _to_px(0, y)
            cv2.line(img, (px, 0), (px, IMG_H), (30, 30, 30), 1)

        # 차량 위치
        cx, cy = _to_px(0, 0)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(img, "CAR", (cx + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 노란색 점 (어두운 노랑)
        for x, y in self._yellow_pts:
            px, py = _to_px(x, y)
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                cv2.circle(img, (px, py), 2, (0, 150, 200), -1)

        # 흰색 점 (흰)
        for x, y in self._white_pts:
            px, py = _to_px(x, y)
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                cv2.circle(img, (px, py), 2, (255, 255, 255), -1)

        # 오른쪽 실선 피팅 (파랑)
        for i in range(len(self._right_pts) - 1):
            p1 = _to_px(*self._right_pts[i])
            p2 = _to_px(*self._right_pts[i + 1])
            cv2.line(img, p1, p2, (255, 100, 0), 2)

        # center path (초록)
        for i in range(len(self._center_pts) - 1):
            p1 = _to_px(*self._center_pts[i])
            p2 = _to_px(*self._center_pts[i + 1])
            cv2.line(img, p1, p2, (0, 255, 0), 2)

        # 범례
        cv2.putText(img, "white pts", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "yellow pts", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 200), 1)
        cv2.putText(img, "right fit", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.putText(img, "center path", (10, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 점 개수 표시
        cv2.putText(img, f"W:{len(self._white_pts)} Y:{len(self._yellow_pts)}",
                    (IMG_W - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("test_viewer", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TestViewerNode()
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
