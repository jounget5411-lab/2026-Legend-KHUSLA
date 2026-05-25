import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_viewer",
            default_value="true",
            description="Start fused_viewer_node GUI when true.",
        ),

        # Unity 시뮬 브릿지
        Node(
            package="ros_tcp_endpoint",
            executable="default_server_endpoint",
            name="tcp_endpoint",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="lane_detect",
            name="lane_detect_node",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="yolo_detect",
            name="yolo_detect_node",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="integration",
            name="integration_node",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="path_planner",
            name="path_planner_node",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="motion",
            name="motion_node",
            output="screen",
        ),
        Node(
            package="track_drive",
            executable="fused_viewer",
            name="fused_viewer_node",
            output="screen",
            condition=IfCondition(use_viewer),
        ),
    ])
