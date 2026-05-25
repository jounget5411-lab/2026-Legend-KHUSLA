from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'track_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 메인 파이프라인
            'lane_detect = track_drive.lane_detect_node:main',
            'yolo_detect = track_drive.yolo_detect_node:main',
            'integration = track_drive.integration_node:main',
            'path_planner = track_drive.path_planner_node:main',
            'motion = track_drive.motion_node:main',
            'fused_viewer = track_drive.fused_viewer_node:main',
            'key_control = track_drive.key_control_node:main',
        ],
    },
)
