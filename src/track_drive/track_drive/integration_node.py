#!/usr/bin/env python3
"""
통합 노드 — YOLO 픽셀 + 라이다 → 미터 좌표 통합 발행.

구독:
  /detect/road_pixels (PoseArray) — YOLO 원본 픽셀 (x=u, y=v, z=cls_id)
  /scan (LaserScan) — 라이다 원본

발행:
  /fused/lane (PoseArray) — 차선 미터 (LANE 6, MID 8, CHILD_LANE 2). z=cls_id 유지
  /fused/stopline (PoseArray) — 정지선 미터 (STOP 10)
  /fused/goal (PoseArray) — 출발/결승선 미터 (GOAL 4)
  /fused/obstacles (PoseArray) — 라이다 장애물 클러스터 (z=반경)

좌표계: 전부 lidar_frame (X=전방+, Y=좌+, 미터).
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray

# ======================== Ground H: 원본 픽셀(640×480) → lidar_frame 미터 ========================
# config/homography_ground.npz 의 H. x=전방, y=좌.
# p = H @ [u, v, 1];  x_fwd = p[0]/p[2],  y_left = p[1]/p[2]

GROUND_H = np.array([
    [-4.18097842466549e-03, -1.87301533010043e-04,  1.37819064484720e+00],
    [ 0.00000000000000e+00,  4.63055528548851e-03, -2.73689017508590e+00],
    [ 0.00000000000000e+00, -2.40130177039888e-03,  5.16378846754945e-01],
], dtype=np.float64)

# ======================== 라이다 파라미터 ========================

LIDAR_RANGE_MIN = 0.1   # 최소 유효 거리 (m) — 차체 반사 제거
LIDAR_RANGE_MAX = 10.0  # 최대 유효 거리 (m)

# 장애물 ROI (lidar_frame 미터) — 튜닝 가능
OBS_X_MIN = 0.1         # 전방 최소 (m)
OBS_X_MAX = 8.0          # 전방 최대 (m)
OBS_Y_HALF = 3.0         # 좌우 ± (m)

# 클러스터링 — 튜닝 가능
CLUSTER_GAP = 0.35      # 인접 점 간 gap > 이 값이면 새 클러스터 (m)
CLUSTER_MIN_PTS = 1     # 클러스터 최소 점 수 (먼 라바콘은 점 1개만 맞아도 잡음)
CLUSTER_PADDING = 0.05  # 반경에 추가할 패딩 (m)

# YOLO 클래스 분류
CLS_LANE_IDS = {2, 6, 8}   # CHILD_LANE, LANE, MID → /fused/lane
CLS_STOP = 10               # STOP → /fused/stopline
CLS_GOAL = 4                # GOAL → /fused/goal

# ======================== H 변환 함수 ========================

def pixels_to_meters(us, vs):
    """원본 카메라 픽셀 배열 → lidar_frame 미터 (벡터화).

    Returns (xs_fwd, ys_left) numpy 배열.
    """
    n = len(us)
    if n == 0:
        return np.array([]), np.array([])
    pts = np.vstack([us, vs, np.ones(n)])  # (3, N)
    proj = GROUND_H @ pts                   # (3, N)
    w = proj[2]
    w[w == 0] = 1e-12
    return proj[0] / w, proj[1] / w


# ======================== 장애물 클러스터링 ========================

def cluster_scan_obstacles(xs, ys):
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


# ======================== PoseArray 헬퍼 ========================

def _make_pose_array(stamp, xs, ys, zs=None):
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = "lidar_frame"
    for i in range(len(xs)):
        p = Pose()
        p.position.x = float(xs[i])
        p.position.y = float(ys[i])
        if zs is not None:
            p.position.z = float(zs[i])
        msg.poses.append(p)
    return msg


# ======================== ROS 노드 ========================

class IntegrationNode(Node):
    def __init__(self):
        super().__init__("integration_node")

        self.create_subscription(
            PoseArray, "/detect/road_pixels", self._on_road_pixels, 10)
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)

        self._pub_lane = self.create_publisher(PoseArray, "/fused/lane", 10)
        self._pub_stop = self.create_publisher(PoseArray, "/fused/stopline", 10)
        self._pub_goal = self.create_publisher(PoseArray, "/fused/goal", 10)
        self._pub_obs = self.create_publisher(PoseArray, "/fused/obstacles", 10)

        self.get_logger().info("integration_node started (YOLO pixel H-transform)")

    def _on_road_pixels(self, msg: PoseArray):
        if not msg.poses:
            self._pub_lane.publish(_make_pose_array(msg.header.stamp, [], []))
            return

        us = np.array([p.position.x for p in msg.poses])
        vs = np.array([p.position.y for p in msg.poses])
        cls = np.array([int(p.position.z) for p in msg.poses])

        # H 변환: 픽셀 → 미터 (전체 한 번에)
        xs_fwd, ys_left = pixels_to_meters(us, vs)

        stamp = msg.header.stamp

        # 차선류 (LANE, MID, CHILD_LANE)
        lane_mask = np.isin(cls, list(CLS_LANE_IDS))
        if lane_mask.any():
            self._pub_lane.publish(_make_pose_array(
                stamp, xs_fwd[lane_mask], ys_left[lane_mask], cls[lane_mask]))
        else:
            self._pub_lane.publish(_make_pose_array(stamp, [], []))

        # 정지선 (STOP)
        stop_mask = cls == CLS_STOP
        if stop_mask.any():
            self._pub_stop.publish(_make_pose_array(
                stamp, xs_fwd[stop_mask], ys_left[stop_mask]))

        # 출발/결승선 (GOAL)
        goal_mask = cls == CLS_GOAL
        if goal_mask.any():
            self._pub_goal.publish(_make_pose_array(
                stamp, xs_fwd[goal_mask], ys_left[goal_mask]))

    def _on_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = ranges.size
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        valid = (np.isfinite(ranges)
                 & (ranges > LIDAR_RANGE_MIN)
                 & (ranges < min(LIDAR_RANGE_MAX, msg.range_max - 1e-3)))

        if not valid.any():
            self._pub_obs.publish(_make_pose_array(msg.header.stamp, [], []))
            return

        xs = ranges[valid] * np.cos(angles[valid])
        ys = ranges[valid] * np.sin(angles[valid])

        roi = ((xs >= OBS_X_MIN) & (xs <= OBS_X_MAX)
               & (np.abs(ys) <= OBS_Y_HALF))

        clusters = []
        if roi.any():
            clusters = cluster_scan_obstacles(xs[roi], ys[roi])

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "lidar_frame"
        for cx, cy, r in clusters:
            p = Pose()
            p.position.x = float(cx)
            p.position.y = float(cy)
            p.position.z = float(r)
            out.poses.append(p)
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
