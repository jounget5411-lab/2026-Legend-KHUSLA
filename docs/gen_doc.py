#!/usr/bin/env python3
"""국민대 자율주행 경진대회 - track_drive 개발 과정 문서 생성"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# ─── 스타일 ───
def h1(t):
    h = doc.add_heading(t, level=1)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1f, 0x3a, 0x5f)
def h2(t):
    h = doc.add_heading(t, level=2)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x2c, 0x5a, 0x8a)
def h3(t):
    doc.add_heading(t, level=3)
def p(t, bold=False):
    para = doc.add_paragraph()
    r = para.add_run(t)
    if bold: r.bold = True
    r.font.size = Pt(10)
def code(t):
    para = doc.add_paragraph()
    r = para.add_run(t)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
def bullet(t):
    para = doc.add_paragraph(style="List Bullet")
    para.add_run(t).font.size = Pt(10)
def table(headers, rows):
    tbl = doc.add_table(rows=1+len(rows), cols=len(headers))
    tbl.style = "Light Grid Accent 1"
    for i, h_ in enumerate(headers):
        c = tbl.rows[0].cells[i]
        c.text = h_
        for r in c.paragraphs[0].runs:
            r.bold = True
    for r, row in enumerate(rows, 1):
        for i, v in enumerate(row):
            tbl.rows[r].cells[i].text = str(v)

# ═══════════════════════════════════════════════════════════
# 표지
# ═══════════════════════════════════════════════════════════
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = title.add_run("2026 국민대 자율주행 경진대회\nKHUSLA Legend - track_drive 개발 문서")
tr.bold = True
tr.font.size = Pt(20)
tr.font.color.rgb = RGBColor(0x1f, 0x3a, 0x5f)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("\n\n개발자: 정은택 (센서 융합), 고수민 (차선 주행)\n").font.size = Pt(12)
sub.add_run("환경: ROS2 Humble + WSL Ubuntu 22.04 + Unity Sim\n").font.size = Pt(11)
sub.add_run("기간: 2026년 5월 - 6월\n").font.size = Pt(11)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 1. 프로젝트 개요
# ═══════════════════════════════════════════════════════════
h1("1. 프로젝트 개요")

h2("1.1 목표")
p("국민대 자율주행 경진대회용 자율주행 시스템 개발. Unity 시뮬레이터 환경에서 신호등, 정지선, "
  "라바콘, 어린이보호구역, 좌회전 지름길, 차량 추월, 보행자 정지 등 복합 상황을 처리하는 "
  "통합 상태머신을 구현.")

h2("1.2 환경")
table(["항목", "내용"], [
    ["OS", "Windows 11 + WSL Ubuntu 22.04"],
    ["ROS", "ROS2 Humble"],
    ["시뮬레이터", "Unity + ros_tcp_endpoint (ROS_DOMAIN_ID=7)"],
    ["언어", "Python 3.10"],
    ["주요 라이브러리", "OpenCV, numpy, ultralytics(YOLOv8n), rclpy"],
    ["GPU", "NVIDIA RTX 4070 Laptop (CUDA)"],
    ["좌표계", "lidar_frame: X=전방+, Y=좌+ (ROS 표준)"],
])

h2("1.3 역할 분담")
bullet("정은택: 카메라-라이다 좌표 통합, 라바콘 주행, 상태머신 통합, "
       "추월/좌회전/어린이보호/신호 로직, IMU 융합")
bullet("고수민(친구 sm): 차선 주행 plan() 함수, 경로 이탈 복구, 콘 모드 동특성")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 2. 시스템 아키텍처
# ═══════════════════════════════════════════════════════════
h1("2. 시스템 아키텍처")

h2("2.1 노드 구조")
p("ROS2 패키지 'track_drive' 안에 7개 노드가 파이프라인으로 동작.")

table(["노드", "역할", "발행 토픽"], [
    ["lane_detect_node", "OpenCV BEV 변환 + HSV 차선 검출", "/detect/lane (z=cls_id)"],
    ["yolo_detect_node", "YOLOv8n 객체 검출 (16 클래스)",
     "/detect/road_pixels, /detect/events_raw"],
    ["integration_node", "라이다 클러스터링 + 차선 통과", "/fused/lane, /fused/obstacles"],
    ["path_planner_node", "상태머신 + 경로 생성",
     "/center_path, /target, /left_turn1,2, /child_zone, /cone_mode, /slow_merge"],
    ["motion_node", "다점 추종 제어 → 모터 명령", "/xycar_motor"],
    ["fused_viewer_node", "BEV 디버그 뷰", "(GUI)"],
    ["key_control_node", "수동 키보드 제어 (RC 모드)", "/xycar_motor (수동)"],
])

h2("2.2 데이터 흐름")
code("Unity Sim → /usb_cam/image_raw, /scan, /imu\n"
     "    ↓\n"
     "lane_detect → /detect/lane (cls_id=6 흰, 8 노란)\n"
     "yolo_detect → /detect/road_pixels (BBOX), /detect/events_raw (플래그)\n"
     "    ↓\n"
     "integration → /fused/lane (차선 패스스루), /fused/obstacles (LiDAR 클러스터)\n"
     "    ↓\n"
     "path_planner → /center_path (PoseArray, 다점 경로)\n"
     "             + 토픽들 (/emergency_stop, /left_turn1,2, /child_zone, /cone_mode)\n"
     "    ↓\n"
     "motion → /xycar_motor (angle, speed)\n"
     "    ↓\n"
     "Unity Sim (차량 제어)")

h2("2.3 좌표계")
p("ROS 표준 lidar_frame 사용:")
bullet("X = 전방(+)")
bullet("Y = 좌측(+) / 우측(-)")
bullet("각도: 0° = 전방, +90° = 좌(9시), -90° = 우(3시)")

p("카메라 픽셀(u, v):")
bullet("u = 가로 (0~640)")
bullet("v = 세로 (0~480), v↑ = 화면 아래 = 가까움")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 3. 개발 과정 (시간순)
# ═══════════════════════════════════════════════════════════
h1("3. 개발 과정 (시간순)")

h2("3.1 카메라-라이다 캘리브레이션")
p("초기에 카메라 픽셀과 라이다 좌표계를 통합하기 위해 호모그래피(H) 매트릭스 산출.")
bullet("Ground H 채택: tf_static 기반, lidar_frame 좌표로 픽셀 → 실측 미터 변환")
bullet("Cone H는 백업으로 유지 (config/homography_ground.npz)")
bullet("좌표 행 정렬: row0=전방(X), row1=좌(Y)")

h3("발생한 버그")
p("호모그래피 좌표 행 순서가 (X=우, Y=전방)으로 저장돼 있어서 통합 시 잘못 매핑됨. "
  "→ lidar_frame 기준으로 row 순서 재정렬한 npz로 교체.")

h2("3.2 노드 분리")
p("초기 단일 노드(track_drive.py)에서 책임 분리:")
bullet("lane_detect_node: OpenCV BEV warp + HSV 흰/노란 필터")
bullet("integration_node: LiDAR 클러스터링, ROI 필터")
bullet("path_planner_node: 상태머신 + 경로 생성")
bullet("motion_node: 다점 추종 제어")
bullet("fused_viewer_node: BEV 시각화 (디버그)")

h2("3.3 차선 검출 (LANE)")
p("OpenCV 기반 BEV 변환 후 HSV 색상 필터로 흰선/노란선 분리.")

h3("HSV 임계값 튜닝 (보수적 → 매우 보수적)")
table(["파라미터", "초기", "최종"], [
    ["LANE_WHITE_V_MIN", "200", "255 (순백만)"],
    ["LANE_WHITE_S_MAX", "50", "0"],
    ["LANE_YELLOW_H_MIN/MAX", "18/38", "18/38"],
    ["LANE_YELLOW_S_MIN", "80", "80"],
    ["LANE_YELLOW_V_MIN", "130", "130"],
])
p("흰선 마스크에 '노란 H 범위 제외' 조건 추가 — 노란선이 흰선으로 오분류되는 문제 해결.")

h3("친구(sm) plan() 함수")
p("lane_planner.py의 plan() 함수가 LANE 모드의 핵심 — 노란 중앙선(cls_id=8) 2차 피팅 후 "
  "center path 생성. 친구 코드 원본 그대로 사용, 절대 수정 금지.")

h2("3.4 라바콘 주행 (CONE)")
p("LiDAR 클러스터에서 왼쪽 콘 줄을 2차 피팅한 후 일정 오프셋만큼 우측으로 이동 = 중앙선 경로.")

h3("핵심 파라미터")
table(["파라미터", "값", "설명"], [
    ["CONE_LEFT_ACCEPT_DIST", "1.5m", "이전 피팅 기준 이 거리 안만 왼쪽 콘으로 인정"],
    ["CONE_OFFSET (TRACK_HALF_WIDTH)", "2.2m", "왼쪽 콘 → 중앙선 오프셋"],
    ["CONE_FIT_MAX_MISS", "39", "이전 피팅 유지 한계 (콘 안 보일 때)"],
    ["CONE_TO_LANE_MISS", "40", "이후 LANE 전환 (2초)"],
    ["CONE_FIT_SMOOTH_ALPHA", "0.20", "EMA 스무딩"],
    ["CONE_GRACE_TICKS", "20", "초기 2초간 피팅 실패 허용"],
])

h3("CONE → LANE 전환")
p("콘이 안 보이는 상태가 지속되면 자동 전환:")
bullet("miss 1~39: 이전 피팅 유지 (잠깐 안 보여도 버팀, prev[0] 점진 감쇠로 직선화)")
bullet("miss 40 이상: LANE 전환, 즉시 _tick_lane() 호출로 경로 끊김 방지")
p("정지 없이 부드럽게 전환됨.")

h3("왼쪽 콘 1개일 때 보정")
p("마지막 구간에서 왼쪽 콘이 1개만 잡힐 때, 경로 1차 계수에 -0.176 보정 (15도 시계방향) → "
  "콘 구간 빠져나갈 때 살짝 우측으로 빠짐. 콘 2개 이상이면 보정 안 함.")

h2("3.5 사람 감지 (PEDESTRIAN)")
p("LANE 주행 중 도로 안에 작은 클러스터가 나타나면 정지.")

h3("감지 조건 (모두 만족)")
table(["조건", "값", "의미"], [
    ["전방 거리 x", "1.5 ~ 8.0 m", "1.5m 미만은 노이즈, 8m 초과는 무관"],
    ["클러스터 반경 r", "0.1 ~ 0.4 m", "0.1 미만 노이즈, 0.4 이상은 차"],
    ["중앙선 기준 dist", "< 1.2 m", "PED_ROAD_HALF_WIDTH 안 = 도로 위"],
    ["클러스터 개수", "1~2개", "3개 이상이면 차"],
    ["밀집도", "낮음", "차는 점이 빽빽함"],
])

h3("해제 조건")
p("사람이 중앙선을 왼쪽으로 넘었으면(oy > center_y) 출발, 또는 사람이 사라지면 출발. "
  "20초 쿨다운으로 재감지 차단.")

h3("최소 정지 시간")
p("PED_MIN_STOP_TICKS = 60 (3초). 즉각 출발 방지.")

h3("발생한 버그")
bullet("초기엔 x 최솟값 0.3m이라 차 코앞 노이즈 잘못 잡힘 → 1.5m로 상향")
bullet("road_half 1.75m라 도로 경계 노이즈 잘못 잡힘 → 1.2m로 축소")
bullet("PEDESTRIAN 정지 시 motion이 PATH_MEMORY_TIMEOUT(1.5초) 대기 후에야 멈춤 → "
       "/emergency_stop Bool 토픽 추가, motion이 estop=True 받으면 즉시 speed=0 발행")

doc.add_page_break()

h2("3.6 좌회전 지름길 (SHORTCUT)")
p("정지선 도착 후 LEFT 신호 감지하면 강제 좌회전 호 발행 → 지름길 차선 추종 → "
  "두 번째 좌회전 → LANE 복귀. 친구(dh)가 작성한 로직.")

h3("Sub-state 흐름")
code("LANE (트리거: stopline + GREEN + no POLICE + heading ±10°)\n"
     "  ↓\n"
     "WAITING: estop + LEFT 신호 대기 (2틱 디바운싱)\n"
     "  ↓ LEFT 확인\n"
     "TURNING_1: motion_node에 /left_turn1 발행 → angle=-100, speed=10 (48틱 = 2.4초)\n"
     "  ↓ 48틱 완료\n"
     "FOLLOW: _tick_lane() 차선 추종, CROSSROAD_OUT v ≥ 278 대기\n"
     "  ↓ CROSSROAD 감지\n"
     "TURNING_2: /left_turn2 발행 → 50틱 (2.5초)\n"
     "  ↓\n"
     "LANE 복귀: slow 3초 + CHILD_ZONE 강제 진입 + 10초 쿨다운")

h3("STOP/CROSSROAD v 임계값")
table(["파라미터", "값", "의미"], [
    ["STOP_V_THRESHOLD", "320", "정지선 bbox bottom v ≥ 320 = 가까이 옴"],
    ["CROSS_V_THRESHOLD", "278", "CROSSROAD_OUT v ≥ 278 = 두 번째 좌회전 시작"],
    ["STOP_HEADING_TARGET", "0°", "정지선이 있는 직선 구간 heading"],
    ["STOP_HEADING_TOL", "10°", "±10° 범위에서만 정지선 인식"],
])

h3("핵심: motion 충돌 회피")
p("초기엔 path_planner가 직접 /xycar_motor에 발행했더니 motion_node와 충돌해서 좌회전이 "
  "이상하게 동작. → /left_turn1, /left_turn2 Bool 토픽 추가, motion이 그 신호 받으면 "
  "하드코딩 angle/speed를 우선 발행.")

h3("RED_STOP 통합")
p("stopline + no GREEN이면 정지. GREEN 뜨면 분기:")
bullet("GREEN + POLICE 0.5초 이내: 그냥 직진 (POLICE 교차로)")
bullet("GREEN + no POLICE: 바로 SHORTCUT.WAITING 진입 (좌회전 대기)")
p("일단 정지하면 stopline이 깜빡여도 GREEN이 떠야만 해제 (깜빡임 무시).")

h2("3.7 어린이 보호구역 (CHILD_ZONE)")
p("CHILD_START 표지 감지 → 속도 6m/s로 제한. CHILD_END 또는 timeout으로 해제.")

h3("동작 방식")
bullet("CHILD_START bbox v ≥ 200: 진입, /child_zone Bool=True 발행")
bullet("3초 쿨다운(CHILD_ENTER): 진입 직후 CHILD_END 오인식 방지")
bullet("CHILD_END bbox v ≥ 400: 3초 지연 후 해제")
bullet("10초 쿨다운(CHILD_EXIT): 해제 후 재진입 방지")
bullet("Hard timeout 17초: CHILD_END 못 잡아도 강제 해제")
bullet("SHORTCUT 종료 직후: 자동 진입 (커브 회전 중이라 표지 못 봐도 강제 활성)")

h3("motion 측 동작")
p("motion_node가 /child_zone=True 받으면 _publish_motor에서 speed > 6.0이면 6.0으로 cap.")

h2("3.8 추월 (OVERTAKE)")
p("전방 차량(BLACK_CAR/GREEN_CAR) 감지 시 2차선으로 이동 → 추월 → 1차선 복귀.")

h3("진입 조건")
bullet("BLACK_CAR/GREEN_CAR bbox bottom v ≥ 250 (가까이 옴)")
bullet("heading 50~170° (추월 허용 구간)")
bullet("POLICE 안 보임")
bullet("20초 쿨다운 종료")

h3("Sub-state 흐름")
table(["단계", "로그명", "기준선/오프셋", "지속"], [
    ["LANE_TO_2ND", "2차선1", "노란선 오른쪽 1.5m", "5초 (100틱)"],
    ["PASSING", "2차선2", "흰선 왼쪽 0.8m", "왼쪽 라이다 감지까지"],
    ["CAR_BESIDE", "추월", "흰선 왼쪽 5.5m", "왼쪽 사라질 때까지"],
    ["RETURN_TO_1ST", "1차선", "노란선 왼쪽 2.0m", "3초 (60틱)"],
    ["MERGE", "복귀", "노란선 오른쪽 0.5m", "2초 (40틱)"],
])

h3("라이다 감지 범위")
bullet("왼쪽 (PASSING/CAR_BESIDE): +90° ±8°, 0.5~3.0m → 옆 차 감지")
bullet("오른쪽 (RETURN_TO_1ST 중): -88° ±2°, 0.01~3.0m → 1차선 합류 신호")

h3("흰선 피팅 (오른쪽 실선)")
p("PASSING/CAR_BESIDE 단계에서 노란선이 아닌 흰선 기준으로 주행해야 함.")
bullet("x 구간(0.5m)별로 Y가 가장 작은(가장 오른쪽) 점 추출")
bullet("이전 피팅 기준 ±0.4m 안 점만 인라이어로 사용 (아웃라이어 제거)")
bullet("2차 피팅 + EMA 스무딩(α=0.15) → 안정적인 경로")
bullet("커브에서도 자동 대응 (피팅이 곡선이면 곡선, 직선이면 직선)")

doc.add_page_break()

h2("3.9 IMU 통합")
p("자율주행 중 heading 정보로 STOP 오인식 방지 및 OVERTAKE 구간 한정.")

h3("Heading 추출")
p("IMU orientation 쿼터니언 → yaw 각도 변환:")
code("siny = 2 * (q.w*q.z + q.x*q.y)\n"
     "cosy = 1 - 2 * (q.y² + q.z²)\n"
     "yaw_deg = degrees(atan2(siny, cosy))")

h3("사용처")
bullet("STOP 필터: heading ±10° 범위에서만 stopline 인식 (커브에서 차선이 stopline으로 오인 방지)")
bullet("OVERTAKE 필터: heading 50~170°에서만 추월 허용")

h3("기록 도구")
p("key_control에서 'I' 키 눌러 IMU heading을 imu_heading.txt에 10Hz로 기록. "
  "트랙 한 바퀴 돌며 stopline/커브 위치의 heading 분포 측정 가능.")

h2("3.10 모델 학습 협업")
h3("초기 모델 (best.pt)")
p("YOLOv8n detection, 16개 클래스. 멀리서는 잘 잡지만 정지선 근처에서 RED/LEFT를 "
  "못 잡는 문제 발생. 학습 데이터가 주행 중 시점으로만 구성됨.")

h3("재학습 (dh2/best.pt)")
p("친구(dh)가 정지선 멈춰서 신호등 바뀌는 모습을 녹화해 추가 학습. 적용 후 RED/LEFT "
  "인식률 개선됨. 클래스 매핑 동일.")

h3("클래스 매핑")
table(["ID", "이름", "용도"], [
    ["0", "BLACK_CAR", "추월 대상"],
    ["1", "CHILD_END", "어린이보호 종료"],
    ["3", "CHILD_START", "어린이보호 시작"],
    ["4", "CROSSROAD_OUT", "교차로 출구 (SHORTCUT용)"],
    ["6", "GREEN", "초록불"],
    ["7", "GREEN_CAR", "추월 대상"],
    ["8", "HUMAN", "보행자"],
    ["10", "LEFT", "좌회전 화살표"],
    ["12", "POLICE", "경찰차 (좌회전 금지)"],
    ["13", "RED", "빨간불"],
    ["14", "STOP", "정지선 마킹"],
    ["15", "YELLOW", "노란불"],
])

h2("3.11 친구(sm2) 코드 통합")
p("친구(고수민)가 콘/어린이/경로이탈 등 버그픽스를 적용한 sm2 노드 세트를 전달. "
  "내(정은택) 상태머신 위에 얹은 형태였고, phase 종류 동일.")

h3("적용 전후")
bullet("적용 전 = '5월 27일 정은택 ver' (git 태그: jeongtaek-0527)")
bullet("적용 후 = '5월 27일 고수민 ver' (git 태그: gosumin-0527)")
bullet("백업: src/track_drive/friend/jeongtaek_0527_backup/")

h3("주요 추가 사항")
bullet("/cone_mode Bool 토픽: CONE 구간에서 motion이 고정속도 12 + 조향게인 1.12")
bullet("경로 이탈 시 직전 경로로 복구 로직")
bullet("CHILD_ZONE 동작 정교화")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 4. 메인 상태머신 흐름
# ═══════════════════════════════════════════════════════════
h1("4. 메인 상태머신 흐름")

code("[IDLE]\n"
     "  ↓ 시뮬 연결 (장애물 or 차선 검출)\n"
     "[WAIT]\n"
     "  - estop + 차선 시각화\n"
     "  ↓ GREEN 신호 감지 후 3.6초 대기\n"
     "[CONE]\n"
     "  - 왼쪽 콘 줄 2차 피팅 + 2.2m 우측 오프셋\n"
     "  - 20틱 초기 grace (peek into cones)\n"
     "  ↓ miss 40회(2초) 연속 → 즉시 LANE 전환\n"
     "[LANE] ← 메인 주행 상태\n"
     "  - lane_planner.plan() 노란선 기반\n"
     "  - 동시 체크: PEDESTRIAN / RED_STOP / SHORTCUT / OVERTAKE\n"
     "  - CHILD_ZONE 활성 시 속도 6 cap (별도 phase 아님)\n"
     "  ↓ 분기 후 모두 LANE으로 복귀\n"
     "[PEDESTRIAN] 사람 감지 → 정지 → 사라지면 LANE\n"
     "[SHORTCUT] WAITING → TURNING_1 → FOLLOW → TURNING_2 → LANE\n"
     "[OVERTAKE] 2차선1 → 2차선2 → 추월 → 1차선 → 복귀 → LANE")

h2("4.1 우선순위")
p("LANE 틱에서 분기 체크 순서:")
code("1. _tick_child_zone()      # 어린이보호 상태 갱신 (병행)\n"
     "2. _check_pedestrian()     # 사람 감지\n"
     "3. _check_red_stop()       # stopline + 신호 (True 반환 시 즉시 종료)\n"
     "4. _check_shortcut()       # 좌회전 지름길\n"
     "5. _check_overtake()       # 추월\n"
     "6. _tick_lane()            # 기본 차선 주행")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 5. 주요 토픽 정리
# ═══════════════════════════════════════════════════════════
h1("5. 주요 토픽 정리")

h2("5.1 센서 입력 (Unity → ROS)")
table(["토픽", "타입", "용도"], [
    ["/usb_cam/image_raw/front", "Image", "전방 카메라"],
    ["/scan", "LaserScan", "360° LiDAR"],
    ["/imu", "Imu", "IMU (heading/angular_velocity)"],
])

h2("5.2 처리 파이프라인")
table(["토픽", "타입", "발행자 → 구독자"], [
    ["/detect/lane", "PoseArray(z=cls)", "lane_detect → integration"],
    ["/detect/road_pixels", "PoseArray(BBOX)", "yolo_detect → path_planner"],
    ["/detect/events_raw", "PoseArray(cls만)", "yolo_detect → path_planner"],
    ["/fused/lane", "PoseArray", "integration → path_planner"],
    ["/fused/obstacles", "PoseArray", "integration → path_planner"],
    ["/center_path", "PoseArray", "path_planner → motion"],
    ["/target", "PointStamped", "path_planner → motion"],
])

h2("5.3 제어 토픽")
table(["토픽", "타입", "용도"], [
    ["/emergency_stop", "Bool", "True 시 motion이 즉시 speed=0"],
    ["/left_turn1", "Bool", "TURNING_1 트리거 (motion 하드코딩)"],
    ["/left_turn2", "Bool", "TURNING_2 트리거"],
    ["/child_zone", "Bool", "어린이보호구역 속도 cap"],
    ["/cone_mode", "Bool", "CONE 구간 고정속도+조향게인 (sm2)"],
    ["/slow_after_turn", "Bool", "좌회전 후 3초 감속"],
    ["/slow_merge", "Bool", "추월 MERGE 시 감속 (현재 미사용)"],
    ["/xycar_motor", "XycarMotor", "최종 모터 명령 (motion → Sim)"],
])

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 6. 발생한 주요 버그/이슈와 해결
# ═══════════════════════════════════════════════════════════
h1("6. 발생한 주요 버그/이슈와 해결")

issues = [
    ("호모그래피 좌표 행 순서 버그",
     "homography_ground.npz가 (X=우, Y=전방)으로 저장돼 통합 시 잘못 매핑.",
     "lidar_frame 기준 (row0=전방, row1=좌)로 재정렬한 npz로 교체."),
    ("LANE 모드 plan 함수 혼동",
     "처음에 common.py의 plan_drive_target()(오른쪽 실선 기반)을 사용 → 친구 환경(노란선 기반)과 동작 다름.",
     "lane_planner.py의 plan()으로 변경. 친구 원본 코드 절대 수정 금지."),
    ("PLAN_HZ 불일치",
     "내 path_planner는 10Hz, 친구 plan()은 20Hz 가정 → 반응 느림.",
     "20Hz로 통일."),
    ("PEDESTRIAN 오인식",
     "x 최솟값 0.3m이라 차 코앞 노이즈 오인식. road_half 1.75m라 도로 경계까지 잡힘.",
     "x ≥ 1.5m, road_half = 1.2m로 조정."),
    ("PEDESTRIAN 정지 지연",
     "center_path 미발행 시 motion이 PATH_MEMORY_TIMEOUT(1.5초) 대기 후 정지.",
     "/emergency_stop Bool 토픽 추가, motion이 즉시 speed=0. PATH_MEMORY_TIMEOUT=0.5s로 단축."),
    ("Integration ROI 너무 좁음",
     "LIDAR_RANGE_MAX=7, OBS_X_MAX=7 → 8~10m 콘 못 잡음.",
     "10m로 확장."),
    ("CONE 시작 실패",
     "CONE_FIT_MIN_POINTS=3인데 시작 시 콘 1~2개만 보임.",
     "min_points=2로 완화, 20틱 grace 추가."),
    ("좌회전 호 경로가 충분히 안 꺾임",
     "motion_node의 다점 추종이 STEER_FOCUS_X=3.8m 지점 가중 → 호의 굽은 부분 약하게 본 결과 ~54° 정도만 조향.",
     "/left_turn1, /left_turn2 Bool 토픽으로 motion에 하드코딩 명령 전달 (angle=-100, speed=10, 50틱)."),
    ("STOP 라인 깜빡임 버그",
     "stopline 가까이서 YOLO가 매 프레임 잡았다 놓침 → STOP → GO → STOP 반복.",
     "_red_stopping 플래그 추가, 한번 정지하면 GREEN 뜰 때만 해제. 깜빡임 무시."),
    ("POLICE 교차로 직진 안 됨",
     "POLICE 봤어도 SHORTCUT 진입 후 LEFT 안 떠서 영원히 대기.",
     "_check_shortcut에서 POLICE 보이면 4초 쿨다운 + 직진. RED_STOP에서 'GREEN + POLICE 0.5초 이내 → GO' 분기."),
    ("OVERTAKE에서 motion 충돌",
     "path_planner가 /xycar_motor 직접 발행 → motion도 발행 중이라 덮어쓰기됨.",
     "path_planner는 /center_path만 발행, 하드코딩 동작은 motion에 Bool 토픽으로 위임."),
    ("OVERTAKE 경로 점프",
     "RETURN_TO_1ST에서 yellow_fit이 None일 때 _tick_lane() 폴백 → 추월 경로와 기본 차선 경로 사이 점프.",
     "yellow_fit 없으면 발행 안 함. motion이 마지막 경로 0.5초 기억하므로 부드럽게 유지."),
    ("흰선/노란선 오분류",
     "HSV 임계값이 느슨해서 노란선이 흰선으로도 잡힘.",
     "흰선 마스크에 '노란 H 범위 제외' 조건 추가. V=255, S=0으로 매우 보수적으로 잡음."),
    ("CONE → LANE 전환 직진 끊김",
     "miss 14 초과 시 prev_fit 버리고 경로 발행 안 함 → motion 0.5초 후 정지 → 자연스럽지 않음.",
     "CONE_FIT_MAX_MISS = CONE_TO_LANE_MISS-1로 설정해서 전환 직전까지 경로 유지."),
    ("YOLO GPU 미사용",
     "초기엔 NVIDIA 드라이버 호환 문제로 device='cpu' 사용 → 추론 느림.",
     "드라이버 업데이트 후 device='cuda'로 변경. RTX 4070 활용으로 추론 속도 대폭 개선."),
]
for title, problem, fix in issues:
    h3(title)
    p("문제: " + problem)
    p("해결: " + fix)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 7. 실행 방법
# ═══════════════════════════════════════════════════════════
h1("7. 실행 방법")

h2("7.1 빌드")
code("cd ~/xycar_ws\n"
     "source /opt/ros/humble/setup.bash\n"
     "colcon build --packages-select track_drive --symlink-install")

h2("7.2 시뮬 연결")
code("# 터미널 1: TCP endpoint\n"
     "ROS_DOMAIN_ID=7 ros2 run ros_tcp_endpoint default_server_endpoint \\\n"
     "  --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000\n"
     "→ Unity에서 Play 누르기")

h2("7.3 자동 주행 (launch 사용)")
code("# 터미널 2: 전체 노드 한방에\n"
     "ROS_DOMAIN_ID=7 ros2 launch track_drive track_drive.launch.py\n\n"
     "# 뷰어 끄려면\n"
     "ROS_DOMAIN_ID=7 ros2 launch track_drive track_drive.launch.py use_viewer:=false")

h2("7.4 노드 개별 실행 (디버그)")
code("ROS_DOMAIN_ID=7 ros2 run track_drive lane_detect\n"
     "ROS_DOMAIN_ID=7 ros2 run track_drive yolo_detect\n"
     "ROS_DOMAIN_ID=7 ros2 run track_drive integration\n"
     "ROS_DOMAIN_ID=7 ros2 run track_drive path_planner\n"
     "ROS_DOMAIN_ID=7 ros2 run track_drive motion\n"
     "ROS_DOMAIN_ID=7 ros2 run track_drive fused_viewer")

h2("7.5 수동 키보드 제어 (RC 모드)")
code("ROS_DOMAIN_ID=7 ros2 run track_drive key_control")
p("주의: motion 노드 끄고 실행 (둘 다 /xycar_motor 발행하면 충돌). 키 조작:")
table(["키", "동작"], [
    ["↑", "전진 (설정속도로)"],
    ["↓", "후진"],
    ["←", "좌조향 최대 (자동복귀)"],
    ["→", "우조향 최대 (자동복귀)"],
    ["W", "설정속도 +1"],
    ["S", "설정속도 -1"],
    ["X", "조향만 0"],
    ["Space", "완전 정지"],
    ["E", "속도만 0"],
    ["I", "IMU heading 기록 토글"],
    ["M", "Odom+IMU 기록 토글"],
])

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 8. 파일 구조
# ═══════════════════════════════════════════════════════════
h1("8. 파일 구조")

code("~/xycar_ws/\n"
     "├── src/track_drive/\n"
     "│   ├── setup.py                  # 패키지 entry points\n"
     "│   ├── package.xml\n"
     "│   ├── launch/\n"
     "│   │   └── track_drive.launch.py # 전체 노드 launch\n"
     "│   ├── config/\n"
     "│   │   ├── best.pt               # YOLO 모델 (현재: dh2 재학습 ver)\n"
     "│   │   ├── best_old.pt           # 백업\n"
     "│   │   └── homography_ground.npz # 카메라-라이다 H\n"
     "│   ├── friend/\n"
     "│   │   ├── sm2/                  # 친구 sm 적용본 (현재 active)\n"
     "│   │   ├── dh/, dh2/             # 친구 dh 모델/로직\n"
     "│   │   └── jeongtaek_0527_backup/# 적용 전 백업\n"
     "│   └── track_drive/              # ★ 실제 실행 모듈\n"
     "│       ├── common.py             # HSV 임계값, lane plan 헬퍼\n"
     "│       ├── lane_detect_node.py   # OpenCV BEV + HSV\n"
     "│       ├── lane_planner.py       # 친구 plan() 원본\n"
     "│       ├── yolo_detect_node.py   # YOLOv8n\n"
     "│       ├── integration_node.py   # LiDAR 클러스터링\n"
     "│       ├── path_planner_node.py  # ★ 상태머신 (이 프로젝트 핵심)\n"
     "│       ├── motion_node.py        # 다점 추종 제어\n"
     "│       ├── fused_viewer_node.py  # BEV 디버그 뷰\n"
     "│       ├── key_control_node.py   # 수동 RC 제어\n"
     "│       ├── test_white_node.py    # 흰선 추종 실험용\n"
     "│       └── test_viewer_node.py   # 흰선 점 디버그 뷰\n"
     "└── docs/                         # 다이어그램 + 이 문서")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 9. 현재 파라미터 전체 (튜닝 최종값)
# ═══════════════════════════════════════════════════════════
h1("9. 현재 파라미터 (튜닝 최종값)")

h2("9.1 LANE 검출 (lane_detect_node, common.py)")
table(["파라미터", "값"], [
    ["LANE_WHITE_V_MIN", "255"],
    ["LANE_WHITE_S_MAX", "0"],
    ["LANE_YELLOW_H_MIN/MAX", "18 / 38"],
    ["LANE_YELLOW_S_MIN", "80"],
    ["LANE_YELLOW_V_MIN", "130"],
])

h2("9.2 CONE 주행")
table(["파라미터", "값"], [
    ["TRACK_HALF_WIDTH (CONE_OFFSET)", "2.2 m"],
    ["CONE_FIT_MIN_POINTS", "2"],
    ["CONE_LEFT_ACCEPT_DIST", "1.5 m"],
    ["CONE_FIT_SMOOTH_ALPHA", "0.20"],
    ["CONE_FIT_MAX_MISS", "39"],
    ["CONE_TO_LANE_MISS", "40"],
    ["CONE_GRACE_TICKS", "20"],
])

h2("9.3 SHORTCUT")
table(["파라미터", "값"], [
    ["STOP_V_THRESHOLD", "320"],
    ["CROSS_V_THRESHOLD", "278"],
    ["STOP_HEADING_TOL", "±10°"],
    ["LEFT_SIGNAL_DEBOUNCE_TICKS", "2"],
    ["SC_BYPASS_COOLDOWN (POLICE)", "80 (4초)"],
    ["SC_DONE_COOLDOWN", "200 (10초)"],
    ["LEFT_TURN1_TICKS / motion", "48 (2.4초)"],
    ["LEFT_TURN2_TICKS / motion", "50 (2.5초)"],
])

h2("9.4 CHILD_ZONE")
table(["파라미터", "값"], [
    ["CHILD_START_V_THR", "200"],
    ["CHILD_END_V_THR", "400"],
    ["CHILD_EXIT_DELAY", "60 (3초)"],
    ["CHILD_ENTER_COOLDOWN", "60 (3초)"],
    ["CHILD_EXIT_COOLDOWN", "200 (10초)"],
    ["CHILD_HARD_TIMEOUT", "340 (17초)"],
    ["CHILD_ZONE_SPEED / motion", "6.0"],
])

h2("9.5 OVERTAKE")
table(["파라미터", "값"], [
    ["OT_CAR_V_THRESHOLD", "250"],
    ["OT_HEADING_MIN/MAX", "50° / 170°"],
    ["OT_YELLOW_RIGHT_OFFSET (2차선1)", "-1.5 m"],
    ["OT_WHITE_LEFT_OFFSET (2차선2)", "+0.8 m"],
    ["OT_WHITE_LEFT_PASSING (추월)", "+5.5 m"],
    ["OT_YELLOW_LEFT_OFFSET_1 (1차선)", "+2.0 m"],
    ["OT_YELLOW_RIGHT_MERGE (복귀)", "-0.5 m"],
    ["OT_LANE_TO_2ND_TICKS", "100 (5초)"],
    ["OT_RETURN_TICKS", "60 (3초)"],
    ["OT_MERGE_TICKS", "40 (2초)"],
    ["왼쪽 라이다", "+90° ±8°, 0.5~3m"],
    ["오른쪽 라이다", "-88° ±2°, 0.01~3m"],
])

h2("9.6 PEDESTRIAN")
table(["파라미터", "값"], [
    ["PED_X_MIN/MAX", "1.5 / 8.0 m"],
    ["반경 r 허용", "0.1 ~ 0.4 m"],
    ["PED_ROAD_HALF_WIDTH", "1.2 m"],
    ["PED_MIN_STOP_TICKS", "60 (3초)"],
    ["PED_COOLDOWN", "400 (20초)"],
])

h2("9.7 WAIT (출발)")
table(["파라미터", "값"], [
    ["GREEN 대기 시간", "72틱 (3.6초)"],
    ["출발 후 STOP 무시", "400틱 (20초)"],
])

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 10. 결론
# ═══════════════════════════════════════════════════════════
h1("10. 결론")

h2("10.1 완성된 기능")
bullet("IDLE → WAIT → CONE → LANE 자동 진행")
bullet("LANE 주행 중 PEDESTRIAN/SHORTCUT/OVERTAKE 분기 자동 처리")
bullet("CHILD_ZONE 자동 진입/해제 + 속도 제한")
bullet("RED_STOP: 정지선 + 빨간불 정지, GREEN으로 분기")
bullet("POLICE 교차로 자동 통과")
bullet("IMU heading 기반 오인식 방지")
bullet("RC 모드 수동 제어 (방향키 자동복귀)")

h2("10.2 git 버전 관리")
table(["태그", "내용"], [
    ["jeongtaek-0527", "정은택 ver (친구 sm2 적용 전, 모든 튜닝 완료)"],
    ["gosumin-0527", "고수민 ver (친구 sm2 적용본, 콘/어린이/cone_mode 추가)"],
])
p("복원: git checkout <tag>")

h2("10.3 향후 개선 여지")
bullet("YOLO 모델 추가 학습 (특정 시점/각도 누락)")
bullet("OVERTAKE 차량 옆 추월 시 안정성 (옆 차 너무 가깝거나 멀 때 대응)")
bullet("커브 구간 LANE 검출 강건성")
bullet("실차 적용 시 좌표계 및 캘리브레이션 재검증")

# ─── 저장 ───
out_dir = os.path.expanduser("~/xycar_ws/docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "track_drive_dev_doc.docx")
doc.save(out_path)
print(f"saved: {out_path}")
