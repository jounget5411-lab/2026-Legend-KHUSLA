#!/usr/bin/env python3
"""
Live BEV verification: overlay camera-detected cones (via H) and lidar cones
on the same BEV plane. Run with simulator active + ROS_DOMAIN_ID=7.

Captures multiple frames, picks ones with good cone detections,
and generates overlay images for visual verification.
"""

import sys, time, threading, math, os
import numpy as np
import cv2
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")
OUT_DIR = os.path.expanduser("~/calib_out_vis")
os.makedirs(OUT_DIR, exist_ok=True)

# Load cone-based H
with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    H_cone = np.array(json.load(f)['H'])

CHASSIS_DIST = 0.5
MAX_CONE_DIST = 10.0


def lidar_to_cones(ranges, angle_min, angle_inc):
    """Extract cone clusters from lidar scan. Returns list of (x, y, r, n_pts)."""
    n = len(ranges)
    angles_rad = angle_min + np.arange(n) * angle_inc
    angles_deg = np.rad2deg(angles_rad)

    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    forward = (angles_deg <= 70) | (angles_deg >= 290)
    mask = valid & forward
    if not np.any(mask):
        return [], np.array([]), np.array([])

    idx = np.where(mask)[0]
    r = ranges[idx]
    a = angles_rad[idx]

    # x=right, y=forward (our convention: 0=front CW → x=r·sin, y=r·cos)
    x_all = r * np.sin(a)
    y_all = r * np.cos(a)

    # Cluster by index gap > 3
    clusters = []
    cs = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] > 3:
            clusters.append((cs, i))
            cs = i
    clusters.append((cs, len(idx)))

    cones = []
    for s, e in clusters:
        if e - s < 1:
            continue
        cx = np.mean(x_all[s:e])
        cy = np.mean(y_all[s:e])
        cr = np.mean(r[s:e])
        cones.append({'x': cx, 'y': cy, 'r': cr, 'n': e - s})

    return cones, x_all, y_all


def detect_camera_cones(img_rgb):
    """Detect orange cones in RGB image. Returns list of (u_bottom, v_bottom, area)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 60])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 80, 60])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Restrict to lower image (road level)
    mask[:170, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 5 or w > 120 or h < w * 0.4:
            continue
        bottom_cx = x + w // 2
        bottom_cy = y + h
        cones.append({'u': bottom_cx, 'v': bottom_cy, 'area': area,
                      'bbox': (x, y, w, h)})

    return cones, mask


def pixel_to_meter(u, v, H):
    """Apply homography: pixel → (X_right, Y_forward) meters."""
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


class CaptureNode(Node):
    def __init__(self):
        super().__init__("live_bev_verify")
        self.frames = []
        self._lock = threading.Lock()
        self._img = None
        self._scan = None

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, "/scan",
            self._on_scan, qos_profile_sensor_data)

        self._capture_count = 0
        self._max_captures = 8
        self.create_timer(1.0, self._try_capture)
        self.get_logger().info("Capturing frames for BEV verification...")

    def _on_image(self, msg):
        if msg.encoding != "rgb8":
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()
        except ValueError:
            return
        with self._lock:
            self._img = arr

    def _on_scan(self, msg):
        with self._lock:
            self._scan = {
                'ranges': np.array(msg.ranges, dtype=np.float32),
                'angle_min': msg.angle_min,
                'angle_increment': msg.angle_increment,
            }

    def _try_capture(self):
        with self._lock:
            img = self._img
            scan = self._scan
        if img is None or scan is None:
            self.get_logger().info("Waiting for image + scan...")
            return

        self.frames.append({'img': img.copy(), 'scan': dict(scan)})
        self._capture_count += 1
        self.get_logger().info(f"Captured frame {self._capture_count}/{self._max_captures}")

        if self._capture_count >= self._max_captures:
            self.get_logger().info("Done capturing. Processing...")
            raise SystemExit


def main():
    rclpy.init()
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass

    frames = node.frames
    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print("No frames captured!")
        return

    print(f"\nCaptured {len(frames)} frames. Generating BEV overlays...\n")

    for fi, frame in enumerate(frames):
        img = frame['img']
        scan = frame['scan']
        ranges = scan['ranges']
        amin = scan['angle_min']
        ainc = scan['angle_increment']

        # Detect cones
        cam_cones, orange_mask = detect_camera_cones(img)
        lid_cones, lid_x, lid_y = lidar_to_cones(ranges, amin, ainc)

        # Transform camera cones to meters via H
        cam_meters = []
        for c in cam_cones:
            mx, my = pixel_to_meter(c['u'], c['v'], H_cone)
            if abs(mx) < 10 and 0 < my < 15:
                cam_meters.append({'mx': mx, 'my': my, 'u': c['u'], 'v': c['v'],
                                   'area': c['area'], 'bbox': c['bbox']})

        print(f"Frame {fi}: {len(cam_cones)} cam detections → "
              f"{len(cam_meters)} valid meter pts, {len(lid_cones)} lidar cones")

        # === Create 3-panel figure ===
        fig = plt.figure(figsize=(22, 9))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.6, 1.4], wspace=0.3)

        # Panel 1: Camera image with detected cones
        ax_cam = fig.add_subplot(gs[0, 0])
        ax_cam.set_title(f"Camera (frame {fi}) — {len(cam_meters)} cones detected", fontsize=11)
        ax_cam.imshow(img)
        for cm in cam_meters:
            x, y, w, h = cm['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, ec='lime', lw=1.5)
            ax_cam.add_patch(rect)
            ax_cam.plot(cm['u'], cm['v'], 'r+', ms=10, mew=2)
            ax_cam.annotate(f"({cm['mx']:+.1f},{cm['my']:.1f})",
                            (cm['u'], cm['v']), fontsize=6, color='yellow',
                            xytext=(3, -8), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6))

        # Panel 2: Orange mask
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_mask.set_title("Orange mask (y>170)", fontsize=11)
        ax_mask.imshow(orange_mask, cmap='gray')
        ax_mask.set_xticks([]); ax_mask.set_yticks([])

        # Panel 3: BEV overlay — THE KEY PLOT
        ax_bev = fig.add_subplot(gs[0, 2])
        ax_bev.set_title("BEV overlay: Camera H (red ×) vs Lidar (cyan •)", fontsize=12)
        ax_bev.set_aspect('equal')

        # All lidar valid points (gray dots)
        if len(lid_x) > 0:
            ax_bev.scatter(lid_x, lid_y, s=2, c='#555555', alpha=0.15, zorder=1)

        # Lidar cone clusters (cyan circles)
        for lc in lid_cones:
            circ = Circle((lc['x'], lc['y']), 0.15, fill=False, ec='cyan',
                          lw=2, zorder=3)
            ax_bev.add_patch(circ)
            ax_bev.plot(lc['x'], lc['y'], 'o', color='cyan', ms=5, zorder=4)
            ax_bev.annotate(f"{lc['r']:.1f}m", (lc['x'], lc['y']),
                            fontsize=7, color='cyan',
                            xytext=(8, 4), textcoords='offset points')

        # Camera-projected cones (red x)
        for cm in cam_meters:
            ax_bev.plot(cm['mx'], cm['my'], 'x', color='red', ms=10, mew=2.5, zorder=5)
            ax_bev.annotate(f"px({cm['u']},{cm['v']})",
                            (cm['mx'], cm['my']), fontsize=6, color='red',
                            xytext=(-8, -12), textcoords='offset points')

        # Draw match lines between nearest camera-projected and lidar cone
        matched_dists = []
        for cm in cam_meters:
            best_d = float('inf')
            best_lc = None
            for lc in lid_cones:
                d = math.hypot(cm['mx'] - lc['x'], cm['my'] - lc['y'])
                if d < best_d:
                    best_d = d
                    best_lc = lc
            if best_lc is not None and best_d < 3.0:
                ax_bev.plot([cm['mx'], best_lc['x']], [cm['my'], best_lc['y']],
                            '-', color='yellow', lw=1.2, alpha=0.7, zorder=2)
                mid_x = (cm['mx'] + best_lc['x']) / 2
                mid_y = (cm['my'] + best_lc['y']) / 2
                ax_bev.annotate(f"{best_d:.2f}m", (mid_x, mid_y),
                                fontsize=8, color='yellow', fontweight='bold',
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7),
                                zorder=6)
                matched_dists.append(best_d)

        # Vehicle marker
        ax_bev.plot(0, 0, 'ko', ms=10, zorder=10)
        ax_bev.annotate('CAR', (0, 0), fontsize=8, ha='center', va='top',
                         xytext=(0, -0.3), textcoords='offset points',
                         color='white',
                         bbox=dict(fc='black', alpha=0.7, pad=2))
        ax_bev.arrow(0, 0, 0, 1.2, head_width=0.15, fc='lime', ec='lime', zorder=10)

        ax_bev.set_xlim(-5, 5)
        ax_bev.set_ylim(-1, 10)
        ax_bev.set_xlabel("X (right+, m)", fontsize=10)
        ax_bev.set_ylabel("Y (forward+, m)", fontsize=10)
        ax_bev.grid(True, alpha=0.2)

        # Stats text
        if matched_dists:
            stats = (f"Matched pairs: {len(matched_dists)}\n"
                     f"Mean gap: {np.mean(matched_dists):.3f} m\n"
                     f"Max gap: {np.max(matched_dists):.3f} m\n"
                     f"Min gap: {np.min(matched_dists):.3f} m")
        else:
            stats = "No matched pairs"
        ax_bev.text(0.98, 0.98, stats, transform=ax_bev.transAxes,
                    fontsize=9, va='top', ha='right', family='monospace',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))

        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"20_live_bev_{fi:02d}.png")
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"  → {path}")
        if matched_dists:
            print(f"    gaps: {[f'{d:.3f}' for d in sorted(matched_dists)]}")

    # === Summary across all frames ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {len(frames)}")
    print("Check images in:", OUT_DIR)
    print("Look for: red × (camera H) should overlap cyan ○ (lidar)")
    print("Yellow lines show nearest-match distance.")


if __name__ == "__main__":
    main()
