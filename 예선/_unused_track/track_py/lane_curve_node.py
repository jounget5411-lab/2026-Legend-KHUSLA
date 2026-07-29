#!/usr/bin/env python3
"""
실험용 — 노란 중앙선 3차 피팅 + 전 구간 경로 + 레이싱 라인(인코스).

기존 plan()(2차 + 가까운 범위)과 비교 실험용. 친구 plan()은 안 건드림.
LANE 구간에서만 의미 있음. path_planner 대신 단독 실행해서 테스트.

흐름:
  /fused/lane(z=8 노란 중앙선) 점
    → 3차 폴리핏 (EMA 스무딩)
    → 수직 1.5m 시프트 = 주행선
    → Laplacian 스무딩 (apex 깎기 = 레이싱 라인, corridor 안 클램프)
    → /center_path 발행

실시간 튜닝 (재시작 불필요):
  ros2 param set /lane_curve_node smooth_iters 30
  ros2 param set /lane_curve_node smooth_alpha 0.3
  ros2 param set /lane_curve_node fit_order 3
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PointStamped

from .lane_planner import PLAN_HZ

LIDAR_FRAME = "lidar_frame"

YELLOW_CLS = {8, 2}         # 노란 중앙선(점선) + 어린이구역 노란 실선(CHILD_LANE) — 같은 중앙 가이드
WHITE_CLS = {6}             # 흰 외곽 실선만
LANE_OFFSET = 0.0           # mid 소스: 중앙선 그대로
WHITE_INWARD = 1.5          # white 소스: 오른쪽 흰선 → 안쪽(좌, +) 시프트 = 주행선

FIT_X_MIN = 0.5
FIT_MIN_POINTS = 6         # 3차 피팅 최소 점
SAMPLE_N = 24
OUTLIER_THR = 0.5          # 이전 피팅 기준 이 거리 밖 점은 버림
EMA_ALPHA = 0.18           # 샘플 y값 시간 스무딩 (낮을수록 안정, 느림)
WHITE_X_BIN = 0.5          # white 소스: x구간별 최우측(min y) 추출 bin

TARGET_X = 3.0             # /target 포인트 위치


def _poses(stamp, xs, ys):
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = LIDAR_FRAME
    for x, y in zip(xs, ys):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        msg.poses.append(p)
    return msg


class LaneCurveNode(Node):
    def __init__(self):
        super().__init__("lane_curve_node")

        # 실시간 튜닝 파라미터
        self.declare_parameter("fit_order", 2)        # 2(안정, 기본) or 3
        self.declare_parameter("smooth_iters", 0)     # 레이싱라인 스무딩 반복 (0=순수 큐빅)
        self.declare_parameter("smooth_alpha", 0.25)  # 스무딩 강도 (0~0.5)
        self.declare_parameter("smooth_margin", 1.2)  # 주행선에서 벗어날 수 있는 최대 (m)
        self.declare_parameter("lane_offset", LANE_OFFSET)
        self.declare_parameter("lane_source", "mid")   # "mid"(점선 중앙) or "white"(실선 외곽)
        self.declare_parameter("white_inward", WHITE_INWARD)
        self.declare_parameter("gap_max", 8.0)         # 2차는 빈 구간 발산 안 함 → 크게 (먼 점까지 연결)
        self.declare_parameter("ema_alpha", 0.40)      # 시간 스무딩 (높을수록 빠른 반응, 낮을수록 안정)

        self._lane_xs = np.array([], dtype=np.float64)
        self._lane_ys = np.array([], dtype=np.float64)
        self._white_xs = np.array([], dtype=np.float64)
        self._white_ys = np.array([], dtype=np.float64)
        self._has_lane = False
        self._prev_fit = None
        # 고정 격자 + 샘플 y값 EMA (계수 EMA보다 안정)
        self._grid = np.linspace(FIT_X_MIN, 11.0, SAMPLE_N)
        self._ema_center = None        # 격자 위 중앙선 y (EMA 누적)

        self.create_subscription(PoseArray, "/fused/lane", self._on_lane, 10)

        self._pub_center = self.create_publisher(PoseArray, "/center_path", 10)
        self._pub_target = self.create_publisher(PointStamped, "/target", 10)
        self._pub_raw = self.create_publisher(PoseArray, "/lane_left", 10)   # 비교용: 순수 큐빅 중앙선
        self._pub_drive = self.create_publisher(PoseArray, "/lane_right", 10)  # 비교용: 주행선(스무딩 전)

        self.create_timer(1.0 / PLAN_HZ, self._tick)
        self._log = 0
        self.get_logger().info("lane_curve_node started (cubic + racing line)")

    def _on_lane(self, msg: PoseArray):
        self._has_lane = True
        if not msg.poses:
            self._lane_xs = np.array([], dtype=np.float64)
            self._lane_ys = np.array([], dtype=np.float64)
            return
        xs = np.fromiter((p.position.x for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        ys = np.fromiter((p.position.y for p in msg.poses),
                         dtype=np.float64, count=len(msg.poses))
        cls = np.fromiter((int(p.position.z) for p in msg.poses),
                          dtype=np.int32, count=len(msg.poses))
        ym = np.isin(cls, list(YELLOW_CLS))
        wm = np.isin(cls, list(WHITE_CLS))
        self._lane_xs = xs[ym]
        self._lane_ys = ys[ym]
        self._white_xs = xs[wm]
        self._white_ys = ys[wm]

    @staticmethod
    def _rightmost_per_bin(xs, ys):
        """x 구간별 최우측(min y) 점 = 오른쪽 흰 실선."""
        if xs.size == 0:
            return xs, ys
        bins = np.arange(FIT_X_MIN, 11.0, WHITE_X_BIN)
        rx, ry = [], []
        for bx in bins:
            mk = (xs >= bx) & (xs < bx + WHITE_X_BIN)
            if not mk.any():
                continue
            yy = ys[mk]
            xx = xs[mk]
            i = int(np.argmin(yy))
            rx.append(float(xx[i]))
            ry.append(float(yy[i]))
        return np.asarray(rx), np.asarray(ry)

    def _tick(self):
        if not self._has_lane:
            return
        stamp = self.get_clock().now().to_msg()

        order = int(self.get_parameter("fit_order").value)
        iters = int(self.get_parameter("smooth_iters").value)
        alpha = float(self.get_parameter("smooth_alpha").value)
        margin = float(self.get_parameter("smooth_margin").value)
        source = str(self.get_parameter("lane_source").value)

        # 소스 선택: mid(점선 중앙) or white(실선 외곽 + 안쪽 시프트)
        if source == "white":
            xs, ys = self._rightmost_per_bin(self._white_xs, self._white_ys)
            offset = float(self.get_parameter("white_inward").value)
        else:
            xs, ys = self._lane_xs, self._lane_ys
            offset = float(self.get_parameter("lane_offset").value)

        if xs.size < FIT_MIN_POINTS:
            return

        # x 정렬 + 첫 큰 gap에서 자르기 (gap 너머는 다른 구간 → 폴리노미얼로 연결 금지)
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        gap_max = float(self.get_parameter("gap_max").value)
        gaps = np.diff(xs)
        big = np.where(gaps > gap_max)[0]
        if big.size > 0:
            cut = int(big[0]) + 1
            xs, ys = xs[:cut], ys[:cut]
        if xs.size < FIT_MIN_POINTS:
            return
        src_xs = xs.copy()   # data_max 계산용

        # (아웃라이어 제거 삭제 — 커브에서 먼 차선을 노이즈로 오인해 버리는 버그였음)

        x_span = float(xs.max() - xs.min())
        if x_span < 1.0:
            return
        deg = order if (xs.size >= order + 2 and x_span >= 2.0) else 1
        try:
            coef = np.polyfit(xs, ys, deg)
        except (np.linalg.LinAlgError, ValueError):
            return
        coef = np.concatenate([np.zeros(4 - len(coef)), coef])  # 4계수로 패딩
        self._prev_fit = coef   # 아웃라이어 기준용 (EMA는 샘플 y값에)

        # 고정 격자에서 곡선 샘플 → 샘플 y값에 EMA (계수 EMA보다 훨씬 안정)
        ema_a = float(self.get_parameter("ema_alpha").value)
        data_max = float(src_xs.max())
        raw_center = np.polyval(coef, self._grid)
        in_range = self._grid <= (data_max + 0.3)
        if self._ema_center is None:
            self._ema_center = raw_center.copy()
        else:
            self._ema_center = (1.0 - ema_a) * self._ema_center + ema_a * raw_center
        # 범위 밖 격자는 stale 방지 — raw로 스냅 (다음에 범위 들어와도 안 튐)
        self._ema_center[~in_range] = raw_center[~in_range]
        center_ys = self._ema_center

        # 외삽 컷: 데이터 있는 격자까지만 사용
        valid = in_range
        if int(np.count_nonzero(valid)) < 3:
            return
        gx = self._grid[valid]
        cy = center_ys[valid]

        # 주행선 = 단순 횡이동 (미분 안 씀, plan()과 동일 방식)
        base_ys = cy + offset

        # 레이싱 라인: Laplacian 스무딩 (apex 깎기), corridor 안 클램프
        racing_ys = base_ys.copy()
        for _ in range(max(0, iters)):
            interior = racing_ys.copy()
            interior[1:-1] = racing_ys[1:-1] + alpha * (
                racing_ys[:-2] + racing_ys[2:] - 2.0 * racing_ys[1:-1])
            racing_ys = np.clip(interior, base_ys - margin, base_ys + margin)

        self._pub_center.publish(_poses(stamp, gx, racing_ys))
        self._pub_raw.publish(_poses(stamp, gx, cy))          # 순수 중앙선 (EMA)
        self._pub_drive.publish(_poses(stamp, gx, base_ys))   # 스무딩 전 주행선

        # /target
        ti = int(np.argmin(np.abs(gx - TARGET_X)))
        t = PointStamped()
        t.header.stamp = stamp
        t.header.frame_id = LIDAR_FRAME
        t.point.x = float(gx[ti])
        t.point.y = float(racing_ys[ti])
        self._pub_target.publish(t)

        self._log += 1
        if self._log >= PLAN_HZ:
            self._log = 0
            self.get_logger().info(
                f"deg={deg} pts={xs.size} x_end={data_max:.1f}m "
                f"iters={iters} a={alpha} margin={margin}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneCurveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
