#!/usr/bin/env python3
"""센서 브링업 — usb_cam(respawn) + 차에만 있는 xycar 라이다/IMU launch include.

역할: 카메라/라이다/IMU 를 한 번에 기동. 실행: ros2 launch track_drive sensors.launch.py
입력: 없음. 출력: /image_raw(rgb8 640x480 ~30fps), /scan(BEST_EFFORT), /imu.
예선 대비 변경점: 신규 파일 (예선은 시뮬이라 센서 브링업 없음).
  - usb_cam 은 첫 실행이 flaky(실측) → respawn=True 로 죽으면 자동 재기동.
  - xycar_lidar / xycar_imu 패키지는 실차에만 설치돼 있음 → 개발 PC 에서도
    이 launch 가 죽지 않도록 존재할 때만 include (없으면 경고 로그만).
"""

import os
from glob import glob

from ament_index_python.packages import (get_package_share_directory,
                                         PackageNotFoundError)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _include_if_present(pkg):
    """pkg 의 launch/*.launch.py 를 찾아 include. 없으면 경고 LogInfo 반환."""
    try:
        share = get_package_share_directory(pkg)
    except PackageNotFoundError:
        return LogInfo(msg="[sensors.launch] %s 패키지 없음 — 실차에서만 기동됨 (스킵)" % pkg)
    candidates = sorted(glob(os.path.join(share, "launch", "*.launch.py")))
    if not candidates:
        return LogInfo(msg="[sensors.launch] %s launch 파일 없음 (스킵)" % pkg)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(candidates[0]))


def generate_launch_description():
    camera = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
        respawn=True,          # 첫 실행 flaky 실측 대응 — 죽으면 자동 재기동
        respawn_delay=2.0,
        parameters=[{
            "video_device": "/dev/video0",   # 실차에서 다르면 여기만 수정 (예: /dev/videoCAM)
            "pixel_format": "yuyv2rgb",      # → /image_raw 인코딩 rgb8 (실측)
            "image_width": 640,
            "image_height": 480,
            "framerate": 30.0,               # 실측 ~29.8Hz
        }],
    )

    return LaunchDescription([
        camera,                            # → /image_raw
        _include_if_present("xycar_lidar"),  # → /scan (500빔, BEST_EFFORT) — 실차 전용
        _include_if_present("xycar_imu"),    # → /imu (35Hz) — 실차 전용
    ])
