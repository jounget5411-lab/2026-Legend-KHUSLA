import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")
    model_path = LaunchConfiguration("model_path")
    default_model_path = os.path.join(
        get_package_share_directory("motion_node"), "config", "best.pt")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_viewer",
            default_value="false",
            description="Start fused_viewer_node GUI when true.",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=default_model_path,
            description="YOLO model path for yolo_detect_node.",
        ),

        Node(
            package="motion_node",
            executable="lane_detect_node",
            name="lane_detect_node",
            output="screen",
        ),
        Node(
            package="motion_node",
            executable="yolo_detect_node",
            name="yolo_detect_node",
            output="screen",
            parameters=[{"model_path": model_path}],
        ),
        Node(
            package="motion_node",
            executable="integration_node",
            name="integration_node",
            output="screen",
        ),
        Node(
            package="motion_node",
            executable="path_planner_node",
            name="path_planner_node",
            output="screen",
        ),
        Node(
            package="motion_node",
            executable="track_drive",
            name="track_drive",
            output="screen",
        ),
        Node(
            package="motion_node",
            executable="fused_viewer_node",
            name="fused_viewer_node",
            output="screen",
            condition=IfCondition(use_viewer),
        ),
    ])
