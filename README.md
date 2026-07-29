# 2026 국민대 자율주행 경진대회 — Legend KHUSLA

## 구조
- **`예선/`** — 예선(시뮬레이터) 최종 제출본. ROS2 Humble, track_drive 패키지 + 기록/스크립트.
  (빌드 제외용 COLCON_IGNORE 포함 — 참고/보존용)
- **`본선/track_drive/`** — 본선 실차(Xycar Y) 코드. 예선 코어(Pure Pursuit·polyfit·phase 머신)를 보존하고
  실차 인터페이스(Float32MultiArray 모터, 어안 카메라, 500빔 라이다, VESC 텔레메트리)로 수술한 v3.1.

## 본선 실행
```bash
# 워크스페이스 루트에서 (예선/은 COLCON_IGNORE로 빌드 제외됨)
colcon build --symlink-install --packages-select track_drive
source install/setup.bash
ros2 launch track_drive sensors.launch.py   # 센서 (차에서)
ros2 launch track_drive drive.launch.py     # 자율주행 (규정 형식)
ros2 launch track_drive viewer.launch.py    # 디버그 뷰어 (선택)
ros2 run track_drive teleop                 # 수동 조종 (자율 중 개입 가능)
```

## 브랜치/태그
- `main` — 이 구조 (예선/ + 본선/)
- `feature/docs` — 예선 원본 레이아웃 보존, 태그: submit-final, submit-safe, jeongtaek-*, gosumin-*
- `feature/realcar` — 본선 개발 히스토리 (예선 대비 diff 추적용)
