#!/usr/bin/env python3
"""Capture K (camera intrinsics) and tf_static (lidar_frame -> usb_cam_front) from sim.

Run with simulator active:
    ros2 run ros_tcp_endpoint default_server_endpoint   # in one terminal
    python3 scripts/capture_tf_caminfo.py               # in another

Saves K, R, t to src/track_drive/config/tf_caminfo.json and .npz
"""

import sys, json, time, math, threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
import tf2_ros


LIDAR_FRAME = "lidar_frame"
CAM_FRONT_FRAME = "usb_cam_front"
CAM_INFO_TOPIC = "/usb_cam/camera_info/front"


def quat_to_R(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class CaptureNode(Node):
    def __init__(self):
        super().__init__("tf_caminfo_capture")
        self.K = None
        self.tf_data = None

        self.create_subscription(CameraInfo, CAM_INFO_TOPIC,
                                 self._on_caminfo, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        tf_qos = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_static, tf_qos)
        self.create_subscription(TFMessage, '/tf', self._on_tf_static, tf_qos)

        self.create_timer(1.0, self._try_capture)
        self.get_logger().info("Waiting for CameraInfo + tf_static ...")

    def _on_caminfo(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Got K:\n{self.K}")

    def _on_tf_static(self, msg):
        for tr in msg.transforms:
            self.tf_buffer.set_transform_static(tr, "capture")
            self.get_logger().info(
                f"TF: {tr.header.frame_id} -> {tr.child_frame_id}")

    def _try_capture(self):
        if self.K is None:
            self.get_logger().info("Still waiting for CameraInfo...")
            return

        try:
            tr = self.tf_buffer.lookup_transform(
                LIDAR_FRAME, CAM_FRONT_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().info(f"TF not ready: {e}")
            # List available frames
            frames = self.tf_buffer.all_frames_as_string()
            if frames:
                self.get_logger().info(f"Available frames:\n{frames}")
            return

        t = np.array([tr.transform.translation.x,
                       tr.transform.translation.y,
                       tr.transform.translation.z], dtype=np.float64)
        q = tr.transform.rotation
        R = quat_to_R(q.x, q.y, q.z, q.w)

        self.tf_data = {
            't': t,
            'R': R,
            'quat': [q.x, q.y, q.z, q.w],
        }

        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"Transform: {LIDAR_FRAME} -> {CAM_FRONT_FRAME}")
        self.get_logger().info(f"t = [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}]")
        self.get_logger().info(f"t_z (camera height) = {t[2]:+.6f} m")
        self.get_logger().info(f"q = [{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]")
        self.get_logger().info(f"R =\n{R}")
        self.get_logger().info(f"\nK =\n{self.K}")

        # Check if tf is dummy (all zeros or identity)
        is_dummy = (np.allclose(t, 0, atol=1e-4) and
                    np.allclose(R, np.eye(3), atol=1e-4))
        if is_dummy:
            self.get_logger().warn("WARNING: tf appears to be DUMMY (identity/zero)!")
        elif abs(t[2]) < 0.01:
            self.get_logger().warn(f"WARNING: t_z={t[2]:.4f} is near zero — camera height suspicious")

        # Save
        import os
        save_dir = os.path.expanduser("~/xycar_ws/src/track_drive/config")
        os.makedirs(save_dir, exist_ok=True)

        np.savez(os.path.join(save_dir, "tf_caminfo.npz"),
                 K=self.K, R=R, t=t,
                 quat=np.array([q.x, q.y, q.z, q.w]))

        with open(os.path.join(save_dir, "tf_caminfo.json"), 'w') as f:
            json.dump({
                'K': self.K.tolist(),
                'R': R.tolist(),
                't': t.tolist(),
                'quat': [q.x, q.y, q.z, q.w],
                't_z_camera_height': float(t[2]),
                'is_dummy': bool(is_dummy),
                'frames': f'{LIDAR_FRAME} -> {CAM_FRONT_FRAME}',
            }, f, indent=2)

        self.get_logger().info(f"\nSaved to {save_dir}/tf_caminfo.{{npz,json}}")
        self.get_logger().info("Capture complete. Ctrl+C to exit.")
        # Stop the timer
        self.destroy_timer(self._try_capture)


def main():
    rclpy.init()
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
