#!/usr/bin/env python3
"""
경로계획 노드 — /fused/* 받아 목표점 /target 발행.

구독: /fused/lane (PoseArray), /fused/obstacles (PoseArray)
발행: /target (PointStamped, lidar_frame 미터: x=전방, y=좌)

plan() 함수에 경로계획 로직을 채우면 됨.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PointStamped

# ======================== 파라미터 ========================

PLAN_HZ = 10  # plan() 호출 주기 (Hz)

# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self._lane_xs = np.array([])
        self._lane_ys = np.array([])
        self._obstacles = []  # [(x, y, r), ...]
        self._has_lane = False
        self._has_obs = False

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)

        self._pub = self.create_publisher(PointStamped, "/target", 10)
        self.create_timer(1.0 / PLAN_HZ, self._tick)

        self.get_logger().info("path_planner_node started (placeholder)")

    # ---- 토픽 → numpy 변환 ----

    def _on_lane(self, msg: PoseArray):
        self._lane_xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
        self._lane_ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)
        self._has_lane = True

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [
            (p.position.x, p.position.y, p.position.z)
            for p in msg.poses
        ]
        self._has_obs = True

    # ---- 주기 실행 ----

    def _tick(self):
        if not self._has_lane and not self._has_obs:
            return

        result = plan(self._lane_xs, self._lane_ys, self._obstacles)
        if result is None:
            return

        target_x, target_y = result

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_frame"
        msg.point.x = float(target_x)  # 전방 (m)
        msg.point.y = float(target_y)  # 좌 (m)
        self._pub.publish(msg)


# ======================== 경로계획 로직 ========================

def plan(lane_xs, lane_ys, obstacles):
    """경로계획: 차선+장애물 → 목표점 (target_x, target_y).

    Parameters
    ----------
    lane_xs : np.ndarray  — 차선 점 전방 좌표 (m), shape (N,)
    lane_ys : np.ndarray  — 차선 점 좌우 좌표 (m), shape (N,)
    obstacles : list of (x, y, r) — 장애물 중심+반경 (m)

    Returns
    -------
    (target_x, target_y) or None  — 목표점 (lidar_frame m)

    ──────────────────────────────────────────────
    여기에 경로계획(차선추종 / 장애물 회피) 로직을 채운다.
    sumin.py의 compute_drive_command 참고.
    입력 형식이 sumin과 동일: lane_xs/ys = numpy, obstacles = [(x,y,r),...].
    ──────────────────────────────────────────────
    """
    # placeholder: 전방 2m 직진
    return (2.0, 0.0)


# ======================== main ========================

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
