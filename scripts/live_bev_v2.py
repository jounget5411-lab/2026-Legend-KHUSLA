#!/usr/bin/env python3
"""
Live BEV v2: Better cone filtering + directional gap analysis.

For each lidar cone, find the NEAREST camera-projected cone within 1.5m.
Show ΔX, ΔY to check systematic offset direction.
Also: drive the car forward slowly to see cones at different distances.
"""

import sys, time, threading, math, os
import numpy as np
import cv2
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")
OUT_DIR = os.path.expanduser("~/calib_out_vis")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    H_cone = np.array(json.load(f)['H'])

CHASSIS_DIST = 0.5
MAX_CONE_DIST = 10.0


def lidar_to_cones(ranges, angle_min, angle_inc):
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
    x_all = r * np.sin(a)
    y_all = r * np.cos(a)

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
        cones.append({
            'x': float(np.mean(x_all[s:e])),
            'y': float(np.mean(y_all[s:e])),
            'r': float(np.mean(r[s:e])),
            'n': e - s,
        })
    return cones, x_all, y_all


def detect_camera_cones_strict(img_rgb):
    """Stricter cone detection: higher thresholds, better filtering."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 80])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([165, 100, 80])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask[:200, :] = 0  # stricter: below y=200 only

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 8 or w > 80:
            continue
        # Aspect ratio: cone should be taller than wide
        if h < w * 0.6:
            continue
        # Size sanity: not too big (probably a building)
        if area > 5000:
            continue

        bottom_cx = x + w // 2
        bottom_cy = y + h
        cones.append({'u': bottom_cx, 'v': bottom_cy, 'area': area,
                      'bbox': (x, y, w, h)})
    return cones, mask


def pixel_to_meter(u, v, H):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


class CaptureNode(Node):
    def __init__(self, n_frames=5, drive=False):
        super().__init__("live_bev_v2")
        self.frames = []
        self._lock = threading.Lock()
        self._img = None
        self._scan = None
        self._count = 0
        self._max = n_frames
        self._drive = drive

        self.create_subscription(Image, "/usb_cam/image_raw/front",
                                 self._on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan",
                                 self._on_scan, qos_profile_sensor_data)

        if drive:
            self._motor_pub = self.create_publisher(XycarMotor, "/xycar_motor", 10)

        self.create_timer(0.8, self._tick)
        self.get_logger().info(f"Capturing {n_frames} frames" +
                               (" (driving forward)" if drive else " (stationary)"))

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

    def _tick(self):
        with self._lock:
            img = self._img
            scan = self._scan

        if img is None or scan is None:
            return

        # Optionally drive forward slowly
        if self._drive:
            msg = XycarMotor()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 3.0
            msg.angle = 0.0
            self._motor_pub.publish(msg)

        self.frames.append({'img': img.copy(), 'scan': dict(scan)})
        self._count += 1
        self.get_logger().info(f"Frame {self._count}/{self._max}")
        if self._count >= self._max:
            # Stop car
            if self._drive:
                msg = XycarMotor()
                msg.speed = 0.0; msg.angle = 0.0
                self._motor_pub.publish(msg)
            raise SystemExit


def process_frames(frames):
    all_gaps = []

    for fi, frame in enumerate(frames):
        img = frame['img']
        scan = frame['scan']
        ranges = scan['ranges']
        amin = scan['angle_min']
        ainc = scan['angle_increment']

        cam_cones, orange_mask = detect_camera_cones_strict(img)
        lid_cones, lid_x, lid_y = lidar_to_cones(ranges, amin, ainc)

        # Transform camera cones to meters
        cam_meters = []
        for c in cam_cones:
            mx, my = pixel_to_meter(c['u'], c['v'], H_cone)
            if abs(mx) < 8 and 0 < my < 12:
                cam_meters.append({'mx': mx, 'my': my, **c})

        # Match: for each LIDAR cone, find nearest camera-projected cone
        matches = []
        used_cam = set()
        for lc in sorted(lid_cones, key=lambda c: c['r']):  # nearest lidar first
            best_ci = -1
            best_d = float('inf')
            for ci, cm in enumerate(cam_meters):
                if ci in used_cam:
                    continue
                d = math.hypot(cm['mx'] - lc['x'], cm['my'] - lc['y'])
                if d < best_d:
                    best_d = d
                    best_ci = ci
            if best_ci >= 0 and best_d < 1.5:
                cm = cam_meters[best_ci]
                dx = cm['mx'] - lc['x']
                dy = cm['my'] - lc['y']
                matches.append({
                    'lid': lc, 'cam': cm,
                    'dx': dx, 'dy': dy, 'dist': best_d,
                    'lid_r': lc['r'],
                })
                used_cam.add(best_ci)
                all_gaps.append({'dx': dx, 'dy': dy, 'dist': best_d,
                                 'lid_r': lc['r'], 'frame': fi})

        print(f"Frame {fi}: {len(cam_cones)} cam → {len(cam_meters)} valid, "
              f"{len(lid_cones)} lidar, {len(matches)} matched")
        for m in matches:
            print(f"  lidar({m['lid']['x']:+.2f},{m['lid']['y']:.2f}) r={m['lid_r']:.1f}m "
                  f"↔ cam({m['cam']['mx']:+.2f},{m['cam']['my']:.2f}) "
                  f"Δ=({m['dx']:+.3f},{m['dy']:+.3f}) |{m['dist']:.3f}|m")

        # === Figure ===
        fig = plt.figure(figsize=(20, 8))

        # Camera
        ax1 = fig.add_axes([0.02, 0.05, 0.33, 0.88])
        ax1.set_title(f"Frame {fi}: {len(cam_cones)} det → {len(matches)} matched", fontsize=10)
        ax1.imshow(img)
        for cm in cam_meters:
            x, y, w, h = cm['bbox']
            matched = any(m['cam'] is cm for m in matches)
            ec = 'lime' if matched else 'gray'
            rect = plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.5 if matched else 0.8)
            ax1.add_patch(rect)
            if matched:
                ax1.plot(cm['u'], cm['v'], 'r+', ms=12, mew=2)

        # BEV overlay
        ax2 = fig.add_axes([0.40, 0.05, 0.58, 0.88])
        ax2.set_title("BEV: Camera×H (red ×) vs Lidar (cyan ○) — yellow=gap", fontsize=11)
        ax2.set_aspect('equal')

        if len(lid_x) > 0:
            ax2.scatter(lid_x, lid_y, s=2, c='#666666', alpha=0.12, zorder=1)

        for lc in lid_cones:
            ax2.add_patch(Circle((lc['x'], lc['y']), 0.12, fill=False,
                                 ec='cyan', lw=2, zorder=3))
            ax2.plot(lc['x'], lc['y'], '.', color='cyan', ms=6, zorder=4)
            ax2.annotate(f"L {lc['r']:.1f}m", (lc['x'], lc['y']),
                         fontsize=7, color='cyan',
                         xytext=(6, 6), textcoords='offset points')

        for cm in cam_meters:
            matched = any(m['cam'] is cm for m in matches)
            color = 'red' if matched else '#ff666688'
            ax2.plot(cm['mx'], cm['my'], 'x', color=color, ms=9 if matched else 6,
                     mew=2.5 if matched else 1, zorder=5)

        for m in matches:
            cx, cy = m['cam']['mx'], m['cam']['my']
            lx, ly = m['lid']['x'], m['lid']['y']
            ax2.plot([cx, lx], [cy, ly], '-', color='yellow', lw=1.5, alpha=0.8, zorder=2)
            mid_x, mid_y = (cx+lx)/2, (cy+ly)/2
            ax2.annotate(
                f"|{m['dist']:.2f}|m\nΔX={m['dx']:+.2f}\nΔY={m['dy']:+.2f}",
                (mid_x, mid_y), fontsize=7, color='yellow', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.75),
                zorder=6)

        # Vehicle
        ax2.plot(0, 0, 'ks', ms=10, zorder=10)
        ax2.arrow(0, 0.1, 0, 0.8, head_width=0.12, fc='lime', ec='lime', zorder=10)

        ax2.set_xlim(-5, 5)
        ax2.set_ylim(-0.5, 10)
        ax2.set_xlabel("X (right+, m)")
        ax2.set_ylabel("Y (forward+, m)")
        ax2.grid(True, alpha=0.15)

        if matches:
            dxs = [m['dx'] for m in matches]
            dys = [m['dy'] for m in matches]
            stats = (f"Matches: {len(matches)}\n"
                     f"ΔX mean: {np.mean(dxs):+.3f}m\n"
                     f"ΔY mean: {np.mean(dys):+.3f}m\n"
                     f"|Δ| mean: {np.mean([m['dist'] for m in matches]):.3f}m")
            ax2.text(0.98, 0.98, stats, transform=ax2.transAxes,
                     fontsize=9, va='top', ha='right', family='monospace',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.85))

        path = os.path.join(OUT_DIR, f"21_bev_v2_{fi:02d}.png")
        plt.savefig(path, dpi=130)
        plt.close()
        print(f"  → {path}")

    # === Overall gap analysis ===
    if all_gaps:
        print(f"\n{'='*60}")
        print("OVERALL GAP ANALYSIS (camera_H - lidar)")
        print(f"{'='*60}")
        dxs = [g['dx'] for g in all_gaps]
        dys = [g['dy'] for g in all_gaps]
        dists = [g['dist'] for g in all_gaps]
        print(f"Total matched pairs: {len(all_gaps)}")
        print(f"ΔX (right bias):  mean={np.mean(dxs):+.3f}m  std={np.std(dxs):.3f}m")
        print(f"ΔY (forward bias): mean={np.mean(dys):+.3f}m  std={np.std(dys):.3f}m")
        print(f"|Δ| distance:      mean={np.mean(dists):.3f}m  max={np.max(dists):.3f}m")
        print(f"\nBy lidar distance:")
        for g in sorted(all_gaps, key=lambda g: g['lid_r']):
            print(f"  r={g['lid_r']:.1f}m → ΔX={g['dx']:+.3f} ΔY={g['dy']:+.3f} |{g['dist']:.3f}|")

        # Gap direction plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.set_title("Gap vectors (camera_H − lidar)")
        ax.set_aspect('equal')
        rs = [g['lid_r'] for g in all_gaps]
        sc = ax.scatter(dxs, dys, c=rs, cmap='viridis', s=60, edgecolors='black', lw=0.5)
        plt.colorbar(sc, ax=ax, label='Lidar distance (m)')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.plot(np.mean(dxs), np.mean(dys), 'r+', ms=20, mew=3, label='mean')
        ax.set_xlabel("ΔX (right+, m)")
        ax.set_ylabel("ΔY (forward+, m)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.set_title("|Δ| vs lidar distance")
        ax.scatter(rs, dists, c='steelblue', s=50, edgecolors='black', lw=0.5)
        ax.set_xlabel("Lidar distance (m)")
        ax.set_ylabel("|Δ| gap (m)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "22_gap_analysis.png"), dpi=130)
        plt.close()
        print(f"\n→ {os.path.join(OUT_DIR, '22_gap_analysis.png')}")


def main():
    rclpy.init()
    node = CaptureNode(n_frames=5, drive=False)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass

    frames = node.frames
    node.destroy_node()
    rclpy.shutdown()

    if frames:
        process_frames(frames)
    else:
        print("No frames captured!")


if __name__ == "__main__":
    main()
