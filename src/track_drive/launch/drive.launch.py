#!/usr/bin/env python3
"""track_drive 주행 일괄 실행 (통신/센서 제외 — ros_tcp_endpoint는 따로 실행).
우리 노드 6개 + BEV 뷰어를 한 번에 띄움.
실행:  ros2 launch track_drive drive.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = "track_drive"

    def node(exe):
        return Node(package=pkg, executable=exe, output="screen")

    return LaunchDescription([
        node("yolo_lane"),         # 차선 seg → /detect/lane
        node("yolo_detect"),       # 표지/객체 → /detect/events_raw, road_pixels
        node("yolo_light"),        # 신호등 → /detect/traffic_light
        node("integration"),       # 라이다+차선 융합 → /fused/lane, /fused/obstacles
        node("path_planner"),      # 상태머신
        node("motion"),            # 조향/속도 → /xycar_motor
        node("yolo_lane_viewer"),  # BEV 디버그 뷰어
    ])
