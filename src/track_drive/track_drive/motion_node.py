#!/usr/bin/env python3
"""
모터 제어 노드 - /center_path 다점 추종으로 /xycar_motor 발행.

구독:
  /center_path (PoseArray, lidar_frame) - 중심선 샘플들
  /target      (PointStamped) - 단일 타겟 (fallback / 활성신호)
발행:
  /xycar_motor (XycarMotor)

test_viewer.compute_steering_angle 이식: 다점 pursuit + heading 가중평균.
slope는 샘플 ys의 finite difference로 추정.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseArray
from std_msgs.msg import Bool
from xycar_msgs.msg import XycarMotor

# ======================== 제어 파라미터 ========================

CONTROL_HZ = 20

# 다점 추종 게인 - 조향 lookahead(STEER_FOCUS_X) 주변 점에 가우시안 가중
# 직선 와리가리를 잡기 위해 pursuit/heading 게인을 보수적으로.
STRAIGHT_PURSUIT_GAIN = 0.08
STRAIGHT_HEADING_GAIN = 0.08
STRAIGHT_BOOST_PURSUIT_GAIN = 0.40
STRAIGHT_BOOST_HEADING_GAIN = 0.40
CURVE_PURSUIT_GAIN = 1.62
CURVE_HEADING_GAIN = 2.22
STEER_FOCUS_X = 3.8      # reanchor 이후 가중 peak 위치 (m)
STEER_FOCUS_SIGMA = 2.4  # 가중 폭 (m, sigma) - 커브를 조금 미리 본다
STEER_GAIN_DISTANCE_M = 2.4
STEER_PREVIEW_DISTANCE_M = 5.5
STEER_HEADING_START_DEG = 4.0
STEER_HEADING_FULL_DEG = 20.0
STEER_CURVE_SWITCH_RATIO = 0.50
STRAIGHT_BOOST_ON_DEG = 5.0
STRAIGHT_BOOST_FULL_DEG = 10.0
STRAIGHT_BOOST_OFF_DEG = 0.0
STRAIGHT_BOOST_MAX_LEVEL = 1.8
STRAIGHT_BOOST_BLEND_ALPHA = 0.45
STRAIGHT_BOOST_MAX_CURVE_RATIO = STEER_CURVE_SWITCH_RATIO
CORNER_EXIT_UNWIND_START_RATIO = 0.45
CORNER_EXIT_UNWIND_END_RATIO = 0.22
CORNER_EXIT_UNWIND_FRAMES = 5
CORNER_EXIT_UNWIND_SCALE = 0.80

ANGLE_MIN = -100.0
ANGLE_MAX = 100.0
LEFT_TURN_ANGLE = -100.0          # 좌회전 하드코딩 조향각
LEFT_TURN_SPEED = 10.0            # 좌회전 속도
LEFT_TURN1_TICKS = 48             # TURNING_1 지속 (48 ticks = 2.4초)
LEFT_TURN2_TICKS = 50             # TURNING_2 지속 (50 ticks = 2.5초)
CHILD_ZONE_SPEED = 6.0            # 어린이 보호구역 속도 제한
SLOW_AFTER_TURN_SPEED = 5.0       # 좌회전 직후 감속
SLOW_AFTER_TURN_TICKS = 60        # 3초 (20Hz)
SLOW_MERGE_SPEED = 7.0            # 추월 합류 시 속도 제한
# 응답 빠르게 하려고 alpha up (한 프레임에 변화의 75% 반영).
ANGLE_SMOOTH_ALPHA = 0.55
ANGLE_MAX_STEP = 12.0
# 작은 path 떨림은 컷해서 직선 진동 줄임.
ANGLE_DEADBAND = 0.6

# 속도
SPEED_CMD_MAX = 100.0
SIM_CORNER_OBSERVED_KMH = 5.0
SIM_CORNER_OBSERVED_CMD = 4.5
SIM_CORNER_TARGET_KMH = 8.0


def _map_kmh_to_speed_cmd(kmh):
    if SIM_CORNER_OBSERVED_KMH <= 0.0:
        return 0.0
    cmd = kmh * SIM_CORNER_OBSERVED_CMD / SIM_CORNER_OBSERVED_KMH
    return float(np.clip(cmd, 0.0, SPEED_CMD_MAX))


SPEED_DEFAULT = 14.0
SPEED_CURVE_BASE = 8.0
SPEED_MIN = 3.0
SPEED_STOP = 0.0
SPEED_SLOW_START_DEG = 12.0
SPEED_SLOW_FULL_DEG = 40.0
PREVIEW_SLOW_SPEED = 3.2
CORNER_INSIDE_SPEED = _map_kmh_to_speed_cmd(SIM_CORNER_TARGET_KMH)
CORNER_ENTER_RATIO = 0.55
CORNER_HOLD_FRAMES = 4
CORNER_EXIT_RATIO = 0.35
CORNER_EXIT_ANGLE_SCALE = 0.60
CORNER_INSIDE_ANGLE_SCALE = 0.82
PREVIEW_CURV_START = 0.018
PREVIEW_CURV_FULL = 0.080
PREVIEW_LATERAL_START_M = 0.45
PREVIEW_LATERAL_FULL_M = 1.35
PREVIEW_DISTANCE_M = 5.0

TARGET_TIMEOUT_S = 0.5
PATH_MEMORY_TIMEOUT_S = 0.5
TARGET_Y_LIMIT = 4.0
TARGET_X = 3.0
LOOKAHEAD_X_MIN = 1.0
LOOKAHEAD_X_MAX = 8.0
REANCHOR_FAR_M = 0.65


# ======================== ROS 노드 ========================

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        self._path_xs = None
        self._path_ys = None
        self._path_stamp = None
        self._target_stamp = None

        self._prev_angle = 0.0
        self._corner_frames = 0
        self._prev_steer_ratio = 0.0
        self._exit_unwind_frames = 0
        self._straight_boost_level = 0.0
        self._e_stop = False
        self._left_turn = False
        self._left_turn_ticks = 0
        self._child_zone = False
        self._slow_after_turn = 0
        self._slow_merge = False

        self.create_subscription(PoseArray, "/center_path", self._on_path, 10)
        self.create_subscription(PointStamped, "/target", self._on_target, 10)
        self.create_subscription(Bool, "/emergency_stop", self._on_estop, 10)
        self.create_subscription(Bool, "/left_turn1", self._on_left_turn1, 10)
        self.create_subscription(Bool, "/left_turn2", self._on_left_turn2, 10)
        self.create_subscription(Bool, "/child_zone", self._on_child_zone, 10)
        self.create_subscription(Bool, "/slow_after_turn", self._on_slow_turn, 10)
        self.create_subscription(Bool, "/slow_merge", self._on_slow_merge, 10)
        self._pub = self.create_publisher(XycarMotor, "/xycar_motor", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._tick)

        self.get_logger().info("motion_node started (centerline-pursuit)")

    def _on_path(self, msg: PoseArray):
        if not msg.poses:
            return
        xs = np.array([p.position.x for p in msg.poses], dtype=np.float64)
        ys = np.array([p.position.y for p in msg.poses], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 2:
            return
        order = np.argsort(xs)
        self._path_xs = xs[order]
        self._path_ys = ys[order]
        self._path_stamp = self.get_clock().now()

    def _on_target(self, msg: PointStamped):
        self._target_stamp = self.get_clock().now()

    def _on_estop(self, msg: Bool):
        self._e_stop = msg.data

    def _on_slow_turn(self, msg: Bool):
        if msg.data:
            self._slow_after_turn = SLOW_AFTER_TURN_TICKS
            self.get_logger().info(
                f"SLOW after turn: speed={SLOW_AFTER_TURN_SPEED} for {SLOW_AFTER_TURN_TICKS} ticks")

    def _on_slow_merge(self, msg: Bool):
        if msg.data != self._slow_merge:
            self.get_logger().info(
                f"SLOW_MERGE {'ON' if msg.data else 'OFF'} → speed cap "
                f"{'%.1f' % SLOW_MERGE_SPEED if msg.data else 'normal'}")
        self._slow_merge = msg.data

    def _on_child_zone(self, msg: Bool):
        if msg.data != self._child_zone:
            self.get_logger().info(
                f"CHILD_ZONE {'ON' if msg.data else 'OFF'} → speed cap "
                f"{'%.1f' % CHILD_ZONE_SPEED if msg.data else 'normal'}")
        self._child_zone = msg.data

    def _on_left_turn1(self, msg: Bool):
        if msg.data and not self._left_turn:
            self._left_turn = True
            self._left_turn_ticks = LEFT_TURN1_TICKS
            self.get_logger().info(
                f"LEFT_TURN1 start: ticks={LEFT_TURN1_TICKS}")

    def _on_left_turn2(self, msg: Bool):
        if msg.data and not self._left_turn:
            self._left_turn = True
            self._left_turn_ticks = LEFT_TURN2_TICKS
            self.get_logger().info(
                f"LEFT_TURN2 start: ticks={LEFT_TURN2_TICKS}")

    def _tick(self):
        if self._e_stop:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        if self._left_turn:
            self._publish_motor(LEFT_TURN_SPEED, LEFT_TURN_ANGLE)
            self._left_turn_ticks -= 1
            if self._left_turn_ticks <= 0:
                self._left_turn = False
                self.get_logger().info("LEFT_TURN done")
            return

        now = self.get_clock().now()

        if self._path_stamp is None and self._target_stamp is None:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        path_age = (
            (now - self._path_stamp).nanoseconds * 1e-9
            if self._path_stamp is not None else float("inf")
        )
        target_age = (
            (now - self._target_stamp).nanoseconds * 1e-9
            if self._target_stamp is not None else float("inf")
        )
        if path_age > PATH_MEMORY_TIMEOUT_S and target_age > TARGET_TIMEOUT_S:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        if self._path_xs is None or self._path_xs.size < 2:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        seg_xs, seg_ys, seg_s = self._reanchored_segment(self._path_xs, self._path_ys)
        if seg_xs.size < 2:
            self._publish_motor(SPEED_STOP, 0.0)
            return

        preview_ratio = self._preview_slow_ratio(seg_xs, seg_ys)
        near_steer_ratio, line_angle_error = self._steer_gain_ratio(
            seg_xs, seg_ys, seg_s, STEER_GAIN_DISTANCE_M)
        preview_steer_ratio, _ = self._steer_gain_ratio(
            seg_xs, seg_ys, seg_s, STEER_PREVIEW_DISTANCE_M)
        steer_ratio = max(near_steer_ratio, preview_steer_ratio)
        if preview_ratio >= CORNER_ENTER_RATIO:
            self._corner_frames = min(CORNER_HOLD_FRAMES, self._corner_frames + 1)
        elif preview_ratio <= CORNER_EXIT_RATIO:
            self._corner_frames = 0
        in_corner = self._corner_frames >= CORNER_HOLD_FRAMES
        if steer_ratio <= STRAIGHT_BOOST_MAX_CURVE_RATIO:
            boost_target = np.clip(
                (line_angle_error - STRAIGHT_BOOST_OFF_DEG)
                / max(STRAIGHT_BOOST_ON_DEG - STRAIGHT_BOOST_OFF_DEG, 1e-6),
                0.0, 1.0)
            boost_target = boost_target * boost_target
            if line_angle_error > STRAIGHT_BOOST_ON_DEG:
                extra_boost = np.clip(
                    (line_angle_error - STRAIGHT_BOOST_ON_DEG)
                    / max(STRAIGHT_BOOST_FULL_DEG - STRAIGHT_BOOST_ON_DEG, 1e-6),
                    0.0, 1.0)
                boost_target = 1.0 + (
                    (STRAIGHT_BOOST_MAX_LEVEL - 1.0) * extra_boost)
        else:
            boost_target = 0.0
        self._straight_boost_level += (
            STRAIGHT_BOOST_BLEND_ALPHA
            * (float(boost_target) - self._straight_boost_level)
        )
        if (
            self._prev_steer_ratio >= CORNER_EXIT_UNWIND_START_RATIO
            and near_steer_ratio <= CORNER_EXIT_UNWIND_END_RATIO
        ):
            self._exit_unwind_frames = CORNER_EXIT_UNWIND_FRAMES
        elif self._exit_unwind_frames > 0:
            self._exit_unwind_frames -= 1

        angle = self._compute_steering(
            seg_xs, seg_ys, seg_s, steer_ratio, self._straight_boost_level)
        angle = self._apply_corner_exit_unwind(angle)
        angle = self._smooth_angle(angle)
        self._prev_steer_ratio = near_steer_ratio
        speed = self._speed_from_angle(angle, seg_xs, seg_ys, preview_ratio, in_corner)
        self._publish_motor(speed, angle)

    @staticmethod
    def _path_distance(xs, ys):
        if xs.size == 0:
            return np.array([], dtype=np.float64)
        seg = np.hypot(np.diff(xs), np.diff(ys))
        return np.concatenate(([0.0], np.cumsum(seg)))

    def _reanchored_segment(self, xs, ys):
        xs = np.asarray(xs, dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0)
        if int(np.count_nonzero(valid)) < 2:
            valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 2:
            return xs, ys, np.zeros_like(xs)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return xs, ys, np.zeros_like(xs)

        anchor_idx = int(np.argmin(xs * xs + ys * ys))
        anchor_dist = math.hypot(float(xs[anchor_idx]), float(ys[anchor_idx]))
        dist = self._path_distance(xs, ys)
        rel_s = dist - float(dist[anchor_idx])

        # 경로에서 멀리 떨어졌으면 가까운 점에서 다시 붙고, 아니면 앞쪽 preview만 쓴다.
        min_s = 0.0 if anchor_dist > REANCHOR_FAR_M else LOOKAHEAD_X_MIN
        mask = (rel_s >= min_s) & (rel_s <= LOOKAHEAD_X_MAX)
        if int(np.count_nonzero(mask)) < 2:
            mask = rel_s >= 0.0
        return xs[mask], ys[mask], rel_s[mask]

    def _compute_steering(self, xs, ys, rel_s, steer_ratio, straight_boost_level=0.0):
        if xs.size < 2:
            return 0.0

        ys = np.clip(ys, -TARGET_Y_LIMIT, TARGET_Y_LIMIT)
        pursuit_angles = np.degrees(np.arctan2(-ys, np.maximum(xs, 1e-3)))

        if xs.size >= 2:
            slopes = np.gradient(ys, xs)
        else:
            slopes = np.zeros_like(xs)
        heading_angles = np.degrees(np.arctan(-slopes))

        use_curve_gain = steer_ratio >= STEER_CURVE_SWITCH_RATIO
        if use_curve_gain:
            pursuit_gain = CURVE_PURSUIT_GAIN
            heading_gain = CURVE_HEADING_GAIN
        else:
            pursuit_gain = STRAIGHT_PURSUIT_GAIN
            heading_gain = STRAIGHT_HEADING_GAIN
            boost_mix = float(np.clip(straight_boost_level, 0.0, STRAIGHT_BOOST_MAX_LEVEL))
            pursuit_gain += (
                STRAIGHT_BOOST_PURSUIT_GAIN - STRAIGHT_PURSUIT_GAIN
            ) * boost_mix
            heading_gain += (
                STRAIGHT_BOOST_HEADING_GAIN - STRAIGHT_HEADING_GAIN
            ) * boost_mix

        sample_angles = (
            pursuit_gain * pursuit_angles +
            heading_gain * heading_angles
        )
        focus = rel_s if rel_s is not None and rel_s.size == xs.size else xs
        weights = np.exp(-((focus - STEER_FOCUS_X) / STEER_FOCUS_SIGMA) ** 2)
        angle = float(np.average(sample_angles, weights=weights))
        return float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    def _apply_corner_exit_unwind(self, angle):
        if self._exit_unwind_frames <= 0:
            return angle
        if self._prev_angle == 0.0 or angle == 0.0:
            return angle
        if math.copysign(1.0, angle) != math.copysign(1.0, self._prev_angle):
            return angle
        max_mag = min(abs(angle), abs(self._prev_angle))
        angle = math.copysign(max_mag * CORNER_EXIT_UNWIND_SCALE, angle)
        return float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))

    def _smooth_angle(self, angle):
        delta = float(angle - self._prev_angle)
        if abs(delta) < ANGLE_DEADBAND:
            angle = self._prev_angle
        else:
            angle = self._prev_angle + ANGLE_SMOOTH_ALPHA * delta
        step = float(np.clip(angle - self._prev_angle, -ANGLE_MAX_STEP, ANGLE_MAX_STEP))
        angle = float(np.clip(self._prev_angle + step, ANGLE_MIN, ANGLE_MAX))
        self._prev_angle = angle
        return angle

    @staticmethod
    def _steer_gain_ratio(xs, ys, rel_s=None, distance_m=STEER_GAIN_DISTANCE_M):
        if xs is None or ys is None or xs.size < 3:
            return 0.0, 0.0

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        rel_s = (
            np.asarray(rel_s, dtype=np.float64)
            if rel_s is not None and rel_s.size == xs.size else None
        )
        valid = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs >= 0.0)
        )
        if rel_s is not None:
            valid &= np.isfinite(rel_s) & (rel_s >= 0.0) & (rel_s <= distance_m)
        else:
            valid &= xs <= distance_m
        xs = xs[valid]
        ys = ys[valid]
        if rel_s is not None:
            rel_s = rel_s[valid]
        if xs.size < 3:
            return 0.0, 0.0

        order = np.argsort(rel_s if rel_s is not None else xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 3:
            return 0.0, 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
            headings = np.degrees(np.arctan(dy))
            bend_delta = float(np.percentile(np.abs(headings - headings[0]), 85))
            line_angle_error = float(abs(np.average(headings)))
        except (FloatingPointError, ValueError):
            kappa = 0.0
            bend_delta = 0.0
            line_angle_error = 0.0

        curv_ratio = np.clip(
            (kappa - PREVIEW_CURV_START)
            / max(PREVIEW_CURV_FULL - PREVIEW_CURV_START, 1e-6),
            0.0, 1.0)
        bend_ratio = np.clip(
            (bend_delta - STEER_HEADING_START_DEG)
            / max(STEER_HEADING_FULL_DEG - STEER_HEADING_START_DEG, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, bend_ratio)), line_angle_error

    @staticmethod
    def _preview_slow_ratio(xs, ys):
        if xs is None or ys is None or xs.size < 4:
            return 0.0

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        valid = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs >= 0.0) & (xs <= PREVIEW_DISTANCE_M)
        )
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < 4:
            return 0.0

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 4:
            return 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
        except (FloatingPointError, ValueError):
            kappa = 0.0

        lateral_delta = float(np.max(np.abs(ys - ys[0])))
        curv_ratio = np.clip(
            (kappa - PREVIEW_CURV_START)
            / max(PREVIEW_CURV_FULL - PREVIEW_CURV_START, 1e-6),
            0.0, 1.0)
        lateral_ratio = np.clip(
            (lateral_delta - PREVIEW_LATERAL_START_M)
            / max(PREVIEW_LATERAL_FULL_M - PREVIEW_LATERAL_START_M, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, lateral_ratio))

    def _speed_from_angle(self, angle, xs, ys, preview_ratio=None, in_corner=False):
        if preview_ratio is None:
            preview_ratio = self._preview_slow_ratio(xs, ys)

        exit_ratio = np.clip(
            float(preview_ratio) / max(CORNER_EXIT_RATIO, 1e-6),
            0.0, 1.0,
        )
        angle_scale = CORNER_EXIT_ANGLE_SCALE + (1.0 - CORNER_EXIT_ANGLE_SCALE) * exit_ratio
        if in_corner and preview_ratio >= CORNER_EXIT_RATIO:
            angle_scale *= CORNER_INSIDE_ANGLE_SCALE
        a = abs(angle) * angle_scale
        if a <= SPEED_SLOW_START_DEG:
            angle_speed = SPEED_DEFAULT
        else:
            t = min(1.0, (a - SPEED_SLOW_START_DEG) / (SPEED_SLOW_FULL_DEG - SPEED_SLOW_START_DEG))
            angle_speed = max(SPEED_MIN, SPEED_DEFAULT - (SPEED_DEFAULT - SPEED_MIN) * t)

        corner_low_speed = CORNER_INSIDE_SPEED if in_corner else PREVIEW_SLOW_SPEED
        curve_preview_speed = SPEED_CURVE_BASE - (SPEED_CURVE_BASE - corner_low_speed) * preview_ratio
        if preview_ratio < CORNER_EXIT_RATIO:
            exit_t = preview_ratio / max(CORNER_EXIT_RATIO, 1e-6)
            preview_speed = SPEED_DEFAULT - (SPEED_DEFAULT - curve_preview_speed) * exit_t
        else:
            preview_speed = curve_preview_speed
        speed = min(angle_speed, preview_speed)
        if in_corner:
            speed = max(CORNER_INSIDE_SPEED, speed)
        return float(np.clip(max(SPEED_MIN, speed), 0.0, SPEED_CMD_MAX))

    def _publish_motor(self, speed, angle):
        if self._slow_after_turn > 0:
            self._slow_after_turn -= 1
            if speed > SLOW_AFTER_TURN_SPEED:
                speed = SLOW_AFTER_TURN_SPEED
        if self._slow_merge and speed > SLOW_MERGE_SPEED:
            speed = SLOW_MERGE_SPEED
        if self._child_zone and speed > CHILD_ZONE_SPEED:
            speed = CHILD_ZONE_SPEED
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(np.clip(speed, 0.0, SPEED_CMD_MAX))
        msg.angle = float(max(ANGLE_MIN, min(ANGLE_MAX, angle)))
        self._pub.publish(msg)


# ======================== main ========================

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
