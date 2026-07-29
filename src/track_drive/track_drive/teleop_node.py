#!/usr/bin/env python3
"""키보드 수동주행 노드 — 홈의 _teleop.py 계승 + CarInterface/mux 계약 적용.

역할: termios cbreak 키 입력(w/s/a/d/A/D/스페이스/c/x/k/q)으로 수동 조작.
      기본 모드는 /teleop_cmd 발행(mux 경유), --direct 인자 시 CarInterface를 거쳐
      /xycar_motor 직발행. 워치독(3s 무입력 감속) + 종료 시 0 발행 유지.
입력: 키보드 (stdin, TTY 필수)
출력: /teleop_cmd [angle, speed, estop] 20Hz (기본) 또는 /xycar_motor [angle, speed] (--direct)
예선 대비 변경점: _teleop.py의 트림/기계 스톱/램프 로직을 CarInterface 모듈로 교체
      (수치 산재 제거 — car.yaml 파라미터 공유). 조향 상태는 논리 단위(±100 스케일)로
      변경 — 기계 변환은 CarInterface(직발행) 또는 motion(mux)이 담당.

사용: ros2 run track_drive teleop [--direct] [--ros-args --params-file car.yaml]
"""

import sys
import time
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float32MultiArray

from .lib import drive_cmd
from .lib.car_interface import CarInterface

# 표시용 서보 환산 (실측: servo = out×0.0082518+0.5004) — 발행값 아님
SERVO_GAIN = 0.0082518
SERVO_OFF = 0.5004


class TeleopNode(Node):
    def __init__(self, direct):
        super().__init__("teleop_node")
        self.direct = direct

        # ---------------- 조작감 파라미터 ----------------
        self.declare_parameter("rate_hz", 20.0)        # 발행 주기 (모터는 연속 수신 필요)
        self.declare_parameter("speed_step", 2.0)      # w/s 한 번당 목표속도 변화
        self.declare_parameter("angle_step", 5.0)      # a/d 논리 조향 스텝 (예선 rel 2.0 상당)
        self.declare_parameter("angle_step_big", 24.0)  # A/D 논리 조향 스텝 (예선 rel 10 상당)
        self.declare_parameter("speed_max", 100.0)
        self.declare_parameter("speed_min", -50.0)     # 후진 한계 (직발행 시 allow_reverse 필요)
        self.declare_parameter("watchdog_sec", 3.0)    # 무입력 시 감속 시작

        # ---------------- CarInterface 파라미터 (car.yaml 공유 — motion과 동일 키) ----------------
        self.declare_parameter("steer_trim", -20.0)        # 직진 트림 (실측)
        self.declare_parameter("steer_scale", 0.42)        # 논리(±100)→기계 조향 스케일
        self.declare_parameter("steer_limit_left", -32.0)  # 트림 기준 좌 스톱 (실측 -34, 마진)
        self.declare_parameter("steer_limit_right", 62.0)  # 트림 기준 우 스톱 (실측 +64, 마진)
        self.declare_parameter("slew_angle_per_tick", 8.0)
        self.declare_parameter("slew_speed_per_tick", 1.5)  # 센서리스 기동 램프 (예선 teleop과 동일)
        self.declare_parameter("speed_deadzone", 4.0)      # 이 미만 명령은 안 구름 (실측)
        self.declare_parameter("allow_reverse", True)      # 텔레옵은 후진 허용 (motion은 False)
        self.declare_parameter("cmd_to_ms", 0.08)          # speed 명령 ×0.08 = m/s (실측)

        p = lambda name: self.get_parameter(name).value
        self.rate_hz = float(p("rate_hz"))
        self.speed_step = float(p("speed_step"))
        self.angle_step = float(p("angle_step"))
        self.angle_step_big = float(p("angle_step_big"))
        self.speed_max = float(p("speed_max"))
        self.speed_min = float(p("speed_min"))
        self.watchdog_sec = float(p("watchdog_sec"))

        cfg = {k: p(k) for k in (
            "steer_trim", "steer_scale", "steer_limit_left", "steer_limit_right",
            "slew_angle_per_tick", "slew_speed_per_tick", "speed_deadzone",
            "allow_reverse", "cmd_to_ms")}
        self.car = CarInterface(cfg)
        # 논리 조향 키 상태 한계 — 기계 스톱을 논리 단위로 환산 (한계 탐색 가능하도록)
        self.steer_lo = self.car.steer_limit_left / self.car.steer_scale
        self.steer_hi = self.car.steer_limit_right / self.car.steer_scale

        if direct:
            self.pub = self.create_publisher(Float32MultiArray, "/xycar_motor", 1)
        else:
            self.pub = self.create_publisher(Float32MultiArray, "/teleop_cmd", 1)

    def send_motor(self, angle_out, speed_out):
        m = Float32MultiArray()
        m.data = [float(angle_out), float(speed_out)]
        self.pub.publish(m)

    def send_teleop(self, angle, speed, estop):
        m = Float32MultiArray()
        m.data = drive_cmd.encode_teleop_cmd(
            {"angle": angle, "speed": speed, "estop": 1.0 if estop else 0.0})
        self.pub.publish(m)


def _help_text(node):
    mode = "직발행(/xycar_motor)" if node.direct else "mux 경유(/teleop_cmd)"
    return f"""
╔══════════════ XYCAR 키보드 주행 [{mode}] ══════════════╗
  w / s   : 속도 +/-  (스텝 {node.speed_step:.0f})
  a / d   : 조향 좌/우 미세 (논리 스텝 {node.angle_step:.0f})
  A / D   : 조향 좌/우 크게 (논리 스텝 {node.angle_step_big:.0f})
            기계 변환(트림 {node.car.steer_trim:+.0f}, 스톱 {node.car.steer_limit_left:+.0f}~{node.car.steer_limit_right:+.0f})은
            CarInterface{"" if node.direct else "(motion측)"}가 담당
  space   : 즉시 정지 (estop)   c : 조향 중립(0)
  x       : 전부 0 (estop 해제)
  k /엔터 : 유지 (워치독 갱신, 값 변화 없음)
  1~5     : 속도 프리셋 10/20/30/40/50
  q       : 종료 (정지 후 종료)
  ※ {node.watchdog_sec:.0f}초간 키 없으면 자동 감속
╚═══════════════════════════════════════════════════════╝
"""


def main(args=None):
    rclpy.init(args=args)
    argv = remove_ros_args(sys.argv if args is None else args)
    direct = "--direct" in argv
    node = TeleopNode(direct)

    if not sys.stdin.isatty():
        node.get_logger().error("stdin이 TTY가 아님 — 터미널에서 직접 실행할 것")
        node.destroy_node()
        rclpy.shutdown()
        return

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)

    target_speed = 0.0   # 목표 속도 (명령단위, 키로 조절)
    steer = 0.0          # 논리 조향 (±100 스케일, 0=직진)
    estop = False        # mux 모드에서 /teleop_cmd [2]로 발행
    last_key_t = time.time()
    running = True

    print(_help_text(node))
    print("시작합니다. (안전 확보 확인!)\n")

    try:
        tty.setcbreak(fd)   # Ctrl+C 살아있는 모드
        dt = 1.0 / node.rate_hz

        while running and rclpy.ok():
            loop_start = time.time()

            # ── 키 입력 (논블로킹) ──
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                last_key_t = time.time()

                if ch == 'w':
                    target_speed = min(target_speed + node.speed_step, node.speed_max)
                    estop = False
                elif ch == 's':
                    target_speed = max(target_speed - node.speed_step, node.speed_min)
                    estop = False
                elif ch == 'a':
                    steer = max(steer - node.angle_step, node.steer_lo)
                elif ch == 'd':
                    steer = min(steer + node.angle_step, node.steer_hi)
                elif ch == 'A':
                    steer = max(steer - node.angle_step_big, node.steer_lo)
                elif ch == 'D':
                    steer = min(steer + node.angle_step_big, node.steer_hi)
                elif ch == ' ':
                    target_speed = 0.0
                    estop = True
                    node.car.reset()   # 직발행: 램프 무시 즉시 0부터
                elif ch == 'c':
                    steer = 0.0        # 직진 (트림은 CarInterface가 적용)
                elif ch == 'x':
                    target_speed = 0.0
                    steer = 0.0
                    estop = False
                    node.car.reset()
                elif ch in ('k', '\n', '\r'):
                    pass               # keepalive: 워치독 갱신만
                elif ch in '12345':
                    target_speed = float(int(ch) * 10)
                    estop = False
                elif ch == 'q':
                    running = False

            # ── 워치독: 키가 끊기면 감속 ──
            wd_left = max(0.0, node.watchdog_sec - (time.time() - last_key_t))
            if wd_left <= 0.0 and target_speed != 0.0:
                target_speed = 0.0

            # ── 발행 ──
            if direct:
                # CarInterface가 트림/스톱/데드존/램프 전부 흡수
                angle_out, speed_out = node.car.to_motor(
                    steer, 0.0 if estop else target_speed)
                node.send_motor(angle_out, speed_out)
                shown_out = speed_out
            else:
                # mux 경유 — 기계 변환/램프는 motion측 CarInterface가 담당
                node.send_teleop(steer, target_speed, estop)
                # 표시용 변환만 로컬 계산 (발행값 아님)
                rel = max(node.car.steer_limit_left,
                          min(steer * node.car.steer_scale, node.car.steer_limit_right))
                angle_out = node.car.steer_trim + rel
                shown_out = target_speed

            servo = angle_out * SERVO_GAIN + SERVO_OFF
            flag = " ESTOP" if estop else ("  WD! " if wd_left <= 0.0 else "      ")
            sys.stdout.write(
                f"\r 목표{target_speed:6.1f} | 발행{shown_out:6.1f} "
                f"({node.car.cmd_to_ms(shown_out):5.2f} m/s) | 조향 논리{steer:7.1f} "
                f"(out{angle_out:6.1f} srv{servo:5.3f}) |{flag}| WD {wd_left:3.1f}s  ")
            sys.stdout.flush()

            sleep_left = dt - (time.time() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        pass
    finally:
        # 반드시 정지 명령을 충분히 보냄 (종료 시 0 발행 유지)
        node.car.reset()
        for _ in range(25):
            if direct:
                node.send_motor(node.car.steer_trim, 0.0)   # 트림 위치 + 정지
            else:
                node.send_teleop(0.0, 0.0, False)           # 0.3s 뒤 mux 자동 비활성
            time.sleep(0.02)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\n\n정지 완료. 종료합니다.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
