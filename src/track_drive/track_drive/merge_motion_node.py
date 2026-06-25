#!/usr/bin/env python3
"""
실험용 motion — Hermite 합류 곡선 명시 생성 + 추종.

pursuit이 미래 경로를 안 보여줘서 디버깅 힘든 문제 해결:
현재 위치(0,0, 전방+x)에서 레퍼런스 path(/center_path)로 합류하는
3차 곡선을 명시적으로 만들어 /merge_path로 발행(눈에 보임) + 그걸 따라 주행.

  x ≤ merge_dist : y = a·x³ + b·x²  (현재→레퍼런스 부드러운 합류, 시작 기울기 0)
  x > merge_dist : 레퍼런스 그대로 (완전히 겹침)

friend motion 대신 실행 (둘 다 /xycar_motor 발행 → 동시 금지).

파라미터(실시간):
  merge_dist     : 합류 완료 거리 (m, 기본 3.0)
  ctrl_lookahead : 조향 타겟 거리 (m, 기본 2.0)
  pursuit_gain   : 조향 게인 (기본 2.5)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool
from xycar_msgs.msg import XycarMotor

CONTROL_HZ = 20
LIDAR_FRAME = "lidar_frame"

ANGLE_MAX = 100.0
SPEED_DEFAULT = 8.0
SPEED_MIN = 3.0
SPEED_SLOW_DEG = 40.0       # 이 조향각이면 SPEED_MIN까지 감속
SAMPLE_N = 24
PATH_TIMEOUT_S = 0.5


class MergeMotionNode(Node):
    def __init__(self):
        super().__init__("merge_motion_node")

        self.declare_parameter("merge_dist", 3.0)
        self.declare_parameter("ctrl_lookahead", 2.0)
        self.declare_parameter("pursuit_gain", 2.5)
        self.declare_parameter("speed_default", SPEED_DEFAULT)
        self.declare_parameter("min_radius", 5.5)     # 최대조향 최소회전반경 (캘리브값)
        self.declare_parameter("merge_soft", 0.4)      # 곡률 여유 (작을수록 완만/부드러움)
        self.declare_parameter("steer_smooth", 0.4)    # 조향 스무딩 (너무 높으면 지연→진동, 0.3~0.5)
        self.declare_parameter("preview_dist", 4.0)    # (구 pursuit용, 미사용)
        self.declare_parameter("steer_gain", 1.0)      # (구 pursuit용, 미사용)
        self.declare_parameter("k_ff", 1.0)            # FF 게인 (도로곡률→조향)
        self.declare_parameter("k_fb", 12.0)           # FB 게인 (횡오차→조향, deg/m)

        self._ref_xs = None
        self._ref_ys = None
        self._ref_stamp = None
        self._e_stop = False
        self._prev_angle = 0.0

        self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
        self.create_subscription(Bool, "/emergency_stop", self._on_estop, 10)
        self._pub_motor = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self._pub_merge = self.create_publisher(PoseArray, "/merge_path", 10)

        self.create_timer(1.0 / CONTROL_HZ, self._tick)
        self.get_logger().info("merge_motion_node started (Hermite merge + follow)")

    def _on_path(self, msg):
        if not msg.poses:
            return
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size < 2:
            return
        order = np.argsort(xs)
        self._ref_xs = xs[order]
        self._ref_ys = ys[order]
        self._ref_stamp = self.get_clock().now()

    def _on_estop(self, msg):
        self._e_stop = bool(msg.data)

    def _ref_at(self, x):
        """레퍼런스 path의 x지점 y값 (선형보간)."""
        return float(np.interp(x, self._ref_xs, self._ref_ys))

    def _tick(self):
        if self._e_stop:
            self._publish(0.0, 0.0)
            return
        if self._ref_xs is None or self._ref_xs.size < 2:
            self._publish(0.0, 0.0)
            return
        age = (self.get_clock().now() - self._ref_stamp).nanoseconds * 1e-9
        if age > PATH_TIMEOUT_S:
            self._publish(0.0, 0.0)
            return

        stamp = self.get_clock().now().to_msg()
        L = float(self.get_parameter("merge_dist").value)
        ref_xmax = float(self._ref_xs[-1])
        L = min(L, max(0.5, ref_xmax))   # 레퍼런스가 짧으면 거기까지만 합류

        min_R = float(self.get_parameter("min_radius").value)
        kappa_max = 1.0 / max(min_R, 0.1)

        def _coef(L_):
            yL_ = self._ref_at(L_)
            dx = 0.3
            sL_ = (self._ref_at(min(L_ + dx, ref_xmax)) - self._ref_at(max(L_ - dx, 0.0))) \
                / (min(L_ + dx, ref_xmax) - max(L_ - dx, 0.0) + 1e-6)
            a_ = (sL_ * L_ - 2.0 * yL_) / (L_ ** 3)
            b_ = (3.0 * yL_ - sL_ * L_) / (L_ ** 2)
            return a_, b_, yL_, sL_

        # 합류 3차: y(0)=0, y'(0)=0, y(L)=yL, y'(L)=sL
        # 시작 곡률 κ0=|2b|를 κ_max*soft 이하로 (soft<1이면 여유 둬서 풀조향 방지=부드러움)
        soft = float(self.get_parameter("merge_soft").value)
        kappa_target = kappa_max * max(0.1, soft)
        a, b, yL, sL = _coef(L)
        for _ in range(6):
            kappa0 = abs(2.0 * b)
            if kappa0 <= kappa_target or L >= ref_xmax - 0.05:
                break
            L = min(ref_xmax, L * math.sqrt(kappa0 / kappa_target))
            a, b, yL, sL = _coef(L)

        # merge path 생성: [0,L] 3차, [L, xmax] 레퍼런스
        xs = np.linspace(0.0, ref_xmax, SAMPLE_N)
        ys = np.where(
            xs <= L,
            a * xs ** 3 + b * xs ** 2,
            np.interp(np.clip(xs, L, ref_xmax), self._ref_xs, self._ref_ys))
        self._pub_merge.publish(self._poses(stamp, xs, ys))

        # 조향: FF(도로 곡률) + FB(횡오차). 측정 정확도 없어도 FB가 보정.
        # 레퍼런스를 2차 피팅 → 부드러운 도로 곡률 + 차 기준 횡오차
        fmask = (self._ref_xs >= 0.5) & (self._ref_xs <= 6.0)
        fx, fy = self._ref_xs[fmask], self._ref_ys[fmask]
        if fx.size >= 4:
            try:
                c2, c1, c0 = np.polyfit(fx, fy, 2)
            except (np.linalg.LinAlgError, ValueError):
                c2, c1, c0 = 0.0, 0.0, self._ref_at(0.5)
        else:
            c2, c1, c0 = 0.0, 0.0, self._ref_at(0.5)
        kappa_road = 2.0 * c2          # 도로 곡률 (부드러움)
        cross_e = c0                   # 차 기준 차선 횡오차 (x=0 lane y; +면 차선이 왼쪽)

        kff = float(self.get_parameter("k_ff").value)
        kfb = float(self.get_parameter("k_fb").value)
        angle_ff = -(kappa_road * min_R) * ANGLE_MAX * kff   # 좌(+y,곡률+)→angle-
        angle_fb = -kfb * cross_e
        sm = float(self.get_parameter("steer_smooth").value)
        angle = float(np.clip(angle_ff + angle_fb, -ANGLE_MAX, ANGLE_MAX))
        angle = sm * self._prev_angle + (1.0 - sm) * angle
        self._prev_angle = angle

        # 속도: 조향 클수록 감속
        spd_def = float(self.get_parameter("speed_default").value)
        t = min(1.0, abs(angle) / SPEED_SLOW_DEG)
        speed = max(SPEED_MIN, spd_def - (spd_def - SPEED_MIN) * t)

        self._publish(speed, angle)

    def _poses(self, stamp, xs, ys):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = LIDAR_FRAME
        for x, y in zip(xs, ys):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            msg.poses.append(p)
        return msg

    def _publish(self, speed, angle):
        m = XycarMotor()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.speed = float(speed)
        m.angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        self._pub_motor.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MergeMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
