#!/usr/bin/env python3
"""
경로계획 노드 — 상태 머신 기반.

구독: /fused/obstacles, /fused/lane, /auto_mode
발행: /target (PointStamped), /planned_path (PoseArray)

상태: CONE → LANE → PEDESTRIAN → OVERTAKE → TURN → DONE
현재 CONE만 구현. 나머지 placeholder.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseArray, PointStamped

# ======================== 파라미터 ========================

PLAN_HZ = 10                # plan 주기 (Hz)

# CONE 상태 파라미터
TARGET_FORWARD_M = 1.5      # 목표점을 중앙선의 전방 몇 m 지점에서 잡을지
FIT_MIN_PTS = 2             # 한쪽 줄 피팅에 필요한 최소 점 개수
FIT_X_MIN = 0.3             # 피팅에 쓸 전방 최소 거리 (m)
FIT_X_MAX = 7.0             # 피팅에 쓸 전방 최대 거리 (m)
CENTER_SAMPLE_N = 30        # 중앙선 샘플 점 개수
ONE_SIDE_OFFSET = 1.5       # 한쪽만 있을 때 반대쪽 추정용 오프셋 (m)
DEFAULT_TARGET = (2.0, 0.0) # 데이터 부족 시 직진 목표

# ======================== CONE 로직 ========================

def plan_cone(obstacles):
    """라바콘 두 줄 사이 중앙 경로 생성.

    Parameters
    ----------
    obstacles : list of (x, y, r) — lidar_frame 미터

    Returns
    -------
    target : (x, y) or None
    path   : list of (x, y) — 중앙 경로 전체 (viewer용)
    """
    if not obstacles:
        return DEFAULT_TARGET, []

    xs = np.array([o[0] for o in obstacles])
    ys = np.array([o[1] for o in obstacles])

    roi = (xs >= FIT_X_MIN) & (xs <= FIT_X_MAX)
    xs, ys = xs[roi], ys[roi]

    left_mask = ys > 0
    right_mask = ys < 0
    lx, ly = xs[left_mask], ys[left_mask]
    rx, ry = xs[right_mask], ys[right_mask]

    have_left = len(lx) >= FIT_MIN_PTS
    have_right = len(rx) >= FIT_MIN_PTS

    if not have_left and not have_right:
        return DEFAULT_TARGET, []

    x_min = max(FIT_X_MIN, min(xs))
    x_max = min(FIT_X_MAX, max(xs))
    sample_xs = np.linspace(x_min, x_max, CENTER_SAMPLE_N)

    if have_left and have_right:
        lcoef = np.polyfit(lx, ly, 2)
        rcoef = np.polyfit(rx, ry, 2)
        left_curve = np.polyval(lcoef, sample_xs)
        right_curve = np.polyval(rcoef, sample_xs)
        center_ys = (left_curve + right_curve) / 2.0
    elif have_left:
        lcoef = np.polyfit(lx, ly, 2)
        left_curve = np.polyval(lcoef, sample_xs)
        center_ys = left_curve - ONE_SIDE_OFFSET
    else:
        rcoef = np.polyfit(rx, ry, 2)
        right_curve = np.polyval(rcoef, sample_xs)
        center_ys = right_curve + ONE_SIDE_OFFSET

    path = list(zip(sample_xs.tolist(), center_ys.tolist()))

    # 목표점: 전방 TARGET_FORWARD_M 에 가장 가까운 중앙선 점
    dists = np.abs(sample_xs - TARGET_FORWARD_M)
    idx = int(np.argmin(dists))
    target = (float(sample_xs[idx]), float(center_ys[idx]))

    return target, path


# ======================== placeholder 상태 함수 ========================

def plan_lane(lane_xs, lane_ys, obstacles):
    # 여기 나중에 차선 추종 구현
    return DEFAULT_TARGET, []

def plan_pedestrian(lane_xs, lane_ys, obstacles):
    # 여기 나중에 보행자 대응 구현
    return None, []

def plan_overtake(lane_xs, lane_ys, obstacles):
    # 여기 나중에 추월 구현
    return DEFAULT_TARGET, []

def plan_turn(lane_xs, lane_ys, obstacles):
    # 여기 나중에 회전 구현
    return DEFAULT_TARGET, []

def plan_done():
    return None, []


# ======================== ROS 노드 ========================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self.state = "CONE"
        self._auto_mode = False

        self._lane_xs = np.array([])
        self._lane_ys = np.array([])
        self._obstacles = []

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)
        self.create_subscription(PoseArray, "/fused/obstacles", self._on_obs, 10)
        self.create_subscription(Bool, "/auto_mode", self._on_auto, 10)

        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_path = self.create_publisher(PoseArray, "/planned_path", 10)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self.get_logger().info(f"path_planner_node started — state={self.state}, auto=OFF")

    def _on_lane(self, msg: PoseArray):
        self._lane_xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
        self._lane_ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)

    def _on_obs(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]

    def _on_auto(self, msg: Bool):
        prev = self._auto_mode
        self._auto_mode = msg.data
        if prev != self._auto_mode:
            self.get_logger().info(f"auto_mode = {'ON' if self._auto_mode else 'OFF'}")

    def _tick(self):
        if not self._auto_mode:
            return

        # 상태 머신 디스패치
        if self.state == "CONE":
            target, path = plan_cone(self._obstacles)
        elif self.state == "LANE":
            target, path = plan_lane(self._lane_xs, self._lane_ys, self._obstacles)
        elif self.state == "PEDESTRIAN":
            target, path = plan_pedestrian(self._lane_xs, self._lane_ys, self._obstacles)
        elif self.state == "OVERTAKE":
            target, path = plan_overtake(self._lane_xs, self._lane_ys, self._obstacles)
        elif self.state == "TURN":
            target, path = plan_turn(self._lane_xs, self._lane_ys, self._obstacles)
        elif self.state == "DONE":
            target, path = plan_done()
        else:
            target, path = DEFAULT_TARGET, []

        # 여기 나중에 상태 전환 조건 (CONE→LANE 등)

        if target is not None:
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "lidar_frame"
            msg.point.x = float(target[0])
            msg.point.y = float(target[1])
            self._pub_target.publish(msg)

        if path:
            msg = PoseArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "lidar_frame"
            for x, y in path:
                p = Pose()
                p.position.x = float(x)
                p.position.y = float(y)
                msg.poses.append(p)
            self._pub_path.publish(msg)


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
