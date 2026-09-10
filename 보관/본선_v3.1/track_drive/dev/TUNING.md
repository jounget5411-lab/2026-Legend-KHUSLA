# 왜곡보정(K/D) + 지면 H 수동 튜닝 — 노트북 작업 환경

차 없이 노트북에서 저장된 사진으로 손수 맞춘다. 다 맞춘 뒤 `camera.yaml` 만 차로 보낸다.

## 1. 환경 (이미 세팅됨 — 확인용)

| 항목 | 값 |
|---|---|
| 위치 | WSL `Ubuntu-22.04` : `~/xycar_ws/본선/track_drive` |
| 창 표시 | WSLg (`cv2.imshow` 동작 확인됨) |
| 파이썬 | 3.10.12 / numpy 1.26.4 / opencv 4.13.0 / PyYAML 5.4.1 / scipy 1.8.0 / Pillow 9.0.1 |
| 한글 폰트 | `/mnt/c/Windows/Fonts/malgun.ttf` (윈도우 맑은 고딕을 WSL 에서 직접 읽음) |
| 사진 | `dev/shots/tape4_raw.jpg` ← 조명 켠 상태 20장 중앙값, 기본으로 열림 |
| 캘리브 | `config/camera.yaml` — 차에 올라가 있는 값과 동일본 |

`scipy` 가 NumPy 버전 경고를 뱉지만 동작에 문제 없다. `tune.sh` 가 그 경고와 Qt 폰트 경고를 걸러준다.

## 2. 실행

```bash
wsl -d Ubuntu-22.04            # 윈도우 터미널에서
~/tune.sh                      # 튜너 실행 (dev/shots/tape4_raw.jpg 로 열림)
```

다른 사진으로 열거나 여러 장 비교:

```bash
~/tune.sh --image dev/shots/tape4_raw.jpg dev/shots/tape3_raw.jpg   # n 키로 순환
~/tune.sh --enhance                                                 # 대비강화 켠 상태로 시작
```

`~/tune.sh` 는 `cd ~/xycar_ws/본선/track_drive && python3 tools/tuner.py "$@"` 이다.
경로에 한글이 있어서 직접 타이핑하면 귀찮으니 이 스크립트를 쓰는 게 편하다.

## 3. 화면

```
┌────────────────────────┬──────────────────┐
│ 보정영상                │ BEV              │
│  연두 격자 = 현재 H 로   │ 노랑 점선 = 타일  │
│  지면격자를 되쏜 예측선  │ 초록 실선 = 0.5m │
│  파란 굵은선 = y=0      │                  │
├────────────────────────┴──────────────────┤
│ 파라미터 목록 (선택된 줄 = 노랑)            │
│ 잰 선: #1 최대 x.xx / RMS x.xx px          │
└───────────────────────────────────────────┘
```

- **보정영상 판정**: 연두 가로선이 실제 줄눈/테이프 위에 얹히는가. 파란 선이 정면 중심 줄눈 위인가.
- **BEV 판정**: 타일이 정사각형인가, 선이 곧은가.
- 격자 원점은 `x0 = -h_lidar_offset_x = -0.10 m`. H 원점이 라이다인데 줄눈은 A'(카메라 지면점)
  기준으로 그었기 때문에 그만큼 뒤로 민 값이다. 시작할 때 터미널에 찍힌다.
  A' 를 줄눈에 정확히 안 맞췄으면 `--grid-x0 <값>` 으로 직접 준다.

## 4. 조작

| 키 | 동작 |
|---|---|
| `TAB` | 그룹 전환 — 왜곡 K/D ↔ 지면 H |
| `↑` `↓` (또는 `w` `x`) | 파라미터 선택 |
| `←` `→` (또는 `a` `d`) | 값 감소 / 증가 |
| `[` `]` | 스텝 배율 ÷10 / ×10 |
| `0` | 선택한 항목만 초기값으로 |
| `R` | 전부 초기값으로 |
| `m` | H 모드 전환 — 물리 6개 ↔ 행렬 원소 9개 |
| `L` → 클릭 2번 | 자로 재기. 보정영상에서 선 양끝을 찍으면 직선에서 벗어난 px 가 계속 갱신됨 |
| `C` | 잰 선 전부 지움 |
| `g` | 격자 on/off (왜곡 맞출 때는 끄는 게 편하다) |
| `e` | 대비강화 on/off (어두운 사진에서 줄눈 볼 때) |
| `N` | new_K 재계산 — **H 무효화됨**, 검은 여백 생겼을 때만 |
| `n` | 다음 사진 |
| `s` | `camera.yaml` 저장 (기존 파일은 `.bak` 으로 백업) |
| `q` / `ESC` | 종료 |

화살표가 안 먹으면 `w` `x` `a` `d` 를 쓴다. 못 알아먹은 키는 화면에 코드가 찍히니
그 숫자를 알려주면 매핑을 추가할 수 있다.

## 5. 순서 (이 순서를 지켜야 한다)

**왜곡을 먼저 확정하고 그다음 H.** `k1` 을 바꾸면 보정영상의 픽셀 위치가 통째로 움직이므로
H 를 먼저 맞춰도 무효가 된다.

1. `g` 로 연두격자 끄기 (왜곡만 보게)
2. `L` 누르고 **가장 긴 테이프 선의 양 끝**을 클릭 → 하단 숫자 확인.
   서로 다른 위치의 선 2~3개를 재두면 한쪽만 좋아지는 걸 잡을 수 있다.
3. `TAB` → 왜곡 그룹. `k1` 을 좌우로 돌려 숫자 최소화 → `k2` → 좌우가 다르면 `cx`
4. 만족하면 `g` 로 격자 켜고 `TAB` → H 그룹
5. `h02`/`h12`(원점) → `h01`/`h10`(축척) → `h21`(먼 곳) → `h11`/`h00`(기울어짐) → `h20`
   (또는 `m` 눌러 물리 6개 모드에서 pitch/height/yaw 로)
6. `s` 저장

## 6. 결과 검증

```bash
cd ~/xycar_ws/본선/track_drive
python3 tools/bev_check.py --image dev/shots/tape4_raw.jpg --camera config/camera.yaml \
    --enhance --save /tmp/bev_after.jpg
```

노란 점선이 사진 속 줄눈 위에 겹치면 성공. 실선이 아니라 점선인 이유가 이것이다 —
실선은 판정 대상인 줄눈을 덮어버린다.

## 7. 차로 보내기

```bash
~/cam_push.sh
```

차 쪽 `camera.yaml` 을 날짜 붙여 `dev/calib/` 에 백업한 뒤 덮어쓰고,
차에서 `load_camera_config` + `check_new_K_match` 를 돌려 런타임이 받아들이는지 확인한다.

## 8. 사진 새로 찍기 (차 연결됐을 때)

```bash
ssh xytron@10.42.0.1
bash ~/xycar_ws/src/track_drive/dev/scripts/start_cam.sh      # 카메라 기동 + 발행 확인
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=7
python3 ~/xycar_ws/src/track_drive/dev/scripts/_shot.py tape5 20
```

`dev/shots/tape5_raw.jpg` (+ 현재 K/D 로 보정한 `tape5_und.jpg`) 로 저장된다.
20장 중앙값이라 압축잡음/노이즈가 줄어 선 검출이 안정적이다. 조명은 켜고 찍을 것.
