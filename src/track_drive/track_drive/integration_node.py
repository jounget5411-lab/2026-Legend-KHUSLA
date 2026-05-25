#!/usr/bin/env python3
"""
통합 노드 — 차선 검출 결과 + 라이다 장애물을 같은 좌표계로 통합 발행.

구독:
  /detect/lane (PoseArray) — 차선 점 (lidar_frame 미터)
  /detect/lane_quality (Float32) — 차선 검출 품질 (0~1)
  /scan (LaserScan) — 라이다 원본

발행:
  /fused/lane (PoseArray) — 차선 점 통과 (lidar_frame 미터, x=전방 y=좌)
  /fused/lane_quality (Float32) — 차선 검출 품질 패스스루
  /fused/obstacles (PoseArray) — 라이다 장애물 클러스터 (position.z = 반경)

좌표계: 전부 lidar_frame (ROS 표준, X=전방+, Y=좌+, 미터).
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32

from .common import LIDAR_FRAME

# ======================== 라이다 파라미터 ========================

LIDAR_RANGE_MIN = 0.1   # 최소 유효 거리 (m) — 차체 반사 제거
LIDAR_RANGE_MAX = 8.0   # 최대 유효 거리 (m)

# 장애물 ROI (lidar_frame 미터) — 트랙 밖 노이즈 차단용, 튜닝 가능
OBS_X_MIN = 0.3         # 전방 최소 (m)
OBS_X_MAX = 8.0         # 전방 최대 (m)
OBS_Y_HALF = 6.0        # 좌우 ± (m) — 라바콘 진입 구간까지 포함

# 클러스터링 — 튜닝 가능
CLUSTER_GAP = 0.35      # 인접 점 간 gap > 이 값이면 새 클러스터 (m)
CLUSTER_MIN_PTS = 1     # 클러스터 최소 점 수 (먼 라바콘은 점 1개만 맞아도 잡음)
CLUSTER_PADDING = 0.05  # 반경에 추가할 패딩 (m)

# ======================== 장애물 클러스터링 (sumin.py 원본) ========================

def cluster_scan_obstacles(xs, ys):
    """연속된 라이다 점을 gap 기준으로 클러스터링. (cx, cy, radius) 리스트 반환."""
    if xs.size == 0:
        return []
    pts = np.column_stack([xs, ys])
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    splits = np.where(d > CLUSTER_GAP)[0] + 1
    groups = np.split(pts, splits)
    out = []
    for g in groups:
        if len(g) < CLUSTER_MIN_PTS:
            continue
        cx, cy = g.mean(axis=0)
        r = float(np.linalg.norm(g - [cx, cy], axis=1).max()) + CLUSTER_PADDING
        out.append((float(cx), float(cy), r))
    return out


# ======================== ROS 노드 ========================

class IntegrationNode(Node):
    def __init__(self):
        super().__init__("integration_node")

        self._latest_lane = None

        self.create_subscription(
            PoseArray, "/detect/lane",
            self._on_lane, 10)
        self.create_subscription(
            Float32, "/detect/lane_quality",
            self._on_lane_quality, 10)

        self.create_subscription(
            LaserScan, "/scan",
            self._on_scan, qos_profile_sensor_data)

        self._pub_lane = self.create_publisher(PoseArray, "/fused/lane", 10)
        self._pub_lane_quality = self.create_publisher(Float32, "/fused/lane_quality", 10)
        self._pub_obs = self.create_publisher(PoseArray, "/fused/obstacles", 10)

        self.get_logger().info("integration_node started (with quality passthrough)")

    def _on_lane(self, msg: PoseArray):
        self._latest_lane = msg
        fused = PoseArray()
        fused.header = msg.header
        fused.header.frame_id = LIDAR_FRAME
        fused.poses = list(msg.poses)
        self._pub_lane.publish(fused)

    def _on_lane_quality(self, msg: Float32):
        self._pub_lane_quality.publish(Float32(data=float(msg.data)))

    def _on_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = ranges.size
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        valid = (np.isfinite(ranges)
                 & (ranges > LIDAR_RANGE_MIN)
                 & (ranges < min(LIDAR_RANGE_MAX, msg.range_max - 1e-3)))

        if not valid.any():
            self._publish_empty_obstacles(msg.header.stamp)
            return

        xs = ranges[valid] * np.cos(angles[valid])  # X = 전방
        ys = ranges[valid] * np.sin(angles[valid])  # Y = 좌

        roi = ((xs >= OBS_X_MIN) & (xs <= OBS_X_MAX)
               & (np.abs(ys) <= OBS_Y_HALF))

        clusters = []
        if roi.any():
            clusters = cluster_scan_obstacles(xs[roi], ys[roi])

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = LIDAR_FRAME
        for cx, cy, r in clusters:
            p = Pose()
            p.position.x = float(cx)  # 전방 (m)
            p.position.y = float(cy)  # 좌 (m)
            p.position.z = float(r)   # 클러스터 반경 (m)
            out.poses.append(p)

        self._pub_obs.publish(out)

    def _publish_empty_obstacles(self, stamp):
        out = PoseArray()
        out.header.stamp = stamp
        out.header.frame_id = LIDAR_FRAME
        self._pub_obs.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = IntegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
