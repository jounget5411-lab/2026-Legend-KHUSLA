#!/usr/bin/env python3
"""Calibration data analysis: lidar convention check + cone distribution."""

import numpy as np
import cv2
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

CALIB_DIR = os.path.expanduser("~/calib_data")
OUT_DIR = os.path.expanduser("~/calib_out_vis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_pair(idx):
    img = cv2.imread(os.path.join(CALIB_DIR, f"img_{idx:03d}.png"))
    npz = np.load(os.path.join(CALIB_DIR, f"scan_{idx:03d}.npz"))
    ranges = npz['ranges']
    angle_min = float(npz['angle_min'])
    angle_inc = float(npz['angle_increment'])
    angles_rad = angle_min + np.arange(len(ranges)) * angle_inc
    return img, ranges, angles_rad


def lidar_to_xy(ranges, angles_rad):
    """Convert lidar polar to vehicle XY.

    Need to figure out the convention from data.
    Try two conventions and see which matches the camera view.
    """
    valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < 15.0)
    r = ranges[valid]
    a = angles_rad[valid]
    idx = np.where(valid)[0]
    return r, a, idx


def detect_orange_cones(img):
    """Detect orange cones in BGR image, return bounding boxes and bottom centers."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Orange in HSV - wide range to handle shadows
    lower1 = np.array([0, 80, 60])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([165, 80, 60])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 | mask2

    # Morphology to connect shadow-split regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # too small
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Cone-like aspect ratio: taller than wide, or at least not extremely flat
        if h < 5:
            continue
        # Bottom center of bounding box = cone base
        bottom_cx = x + w // 2
        bottom_cy = y + h
        cones.append({
            'bbox': (x, y, w, h),
            'bottom': (bottom_cx, bottom_cy),
            'area': area
        })

    return cones, mask


# ============================================================
# 1) Check lidar angular convention with a sample
# ============================================================
print("=" * 60)
print("STEP 1: Lidar angular convention analysis")
print("=" * 60)

img0, ranges0, angles0 = load_pair(0)
r_valid, a_valid, idx_valid = lidar_to_xy(ranges0, angles0)

print(f"Total points: {len(ranges0)}")
print(f"Valid points (0.05-15m): {len(r_valid)}")
print(f"Angle range: {np.rad2deg(angles0[0]):.1f}° to {np.rad2deg(angles0[-1]):.1f}°")

# Find clusters of close-range points (likely cones)
close_mask = r_valid < 5.0
close_r = r_valid[close_mask]
close_a = a_valid[close_mask]
close_idx = idx_valid[close_mask]

print(f"\nClose points (<5m): {len(close_r)}")
print(f"Their indices: {sorted(close_idx)}")
print(f"Their angles (deg): {sorted(np.rad2deg(close_a))}")
print(f"Their distances: {[f'{d:.2f}' for d in close_r[np.argsort(close_idx)]]}")

# Plot lidar in polar view to see structure
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Raw polar plot
ax = axes[0]
ax.set_title("Lidar scan_000: index vs distance")
ax.bar(np.arange(len(ranges0)), np.where(np.isfinite(ranges0) & (ranges0 < 15), ranges0, 0),
       width=1, color='steelblue')
ax.set_xlabel("Index (0-359)")
ax.set_ylabel("Distance (m)")
ax.axvline(x=90, color='red', ls='--', label='idx 90')
ax.axvline(x=0, color='green', ls='--', label='idx 0')
ax.axvline(x=270, color='orange', ls='--', label='idx 270')
ax.legend()

# Convention A: user's claim (0=front, CW) → x=r·sin(θ), y=r·cos(θ)
ax = axes[1]
ax.set_title("Conv A: 0°=front, CW\nx=r·sin(θ), y=r·cos(θ)")
ax.set_aspect('equal')
xa = r_valid * np.sin(a_valid)
ya = r_valid * np.cos(a_valid)
ax.scatter(xa, ya, s=3, c='steelblue', alpha=0.5)
xa_close = close_r * np.sin(close_a)
ya_close = close_r * np.cos(close_a)
ax.scatter(xa_close, ya_close, s=20, c='red', zorder=5)
ax.plot(0, 0, 'ko', ms=8)
ax.annotate('FRONT', xy=(0, 2), fontsize=10, ha='center', color='green')
ax.set_xlabel("x (right+)")
ax.set_ylabel("y (forward+)")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.grid(True, alpha=0.3)

# Convention B: viewer's implied (0=left, CCW?) → adapted from viewer code
ax = axes[2]
ax.set_title("Conv B: viewer code convention\nangles-90°, -cos, -sin")
ax.set_aspect('equal')
a_shifted = a_valid - np.deg2rad(90)
xb = -r_valid * np.cos(a_shifted)
yb = -r_valid * np.sin(a_shifted)
ax.scatter(xb, yb, s=3, c='steelblue', alpha=0.5)
a_shifted_c = close_a - np.deg2rad(90)
xb_close = -close_r * np.cos(a_shifted_c)
yb_close = -close_r * np.sin(a_shifted_c)
ax.scatter(xb_close, yb_close, s=20, c='red', zorder=5)
ax.plot(0, 0, 'ko', ms=8)
ax.annotate('FRONT (viewer +y)', xy=(0, 2), fontsize=10, ha='center', color='green')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_lidar_convention.png"), dpi=120)
plt.close()
print(f"\nSaved: 01_lidar_convention.png")


# ============================================================
# 2) Camera cone detection on sample frames
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Camera cone detection samples")
print("=" * 60)

for sample_idx in [0, 20, 40, 60]:
    img, ranges, angles = load_pair(sample_idx)
    cones, mask = detect_orange_cones(img)
    print(f"\nFrame {sample_idx:03d}: {len(cones)} cones detected")
    for c in sorted(cones, key=lambda c: c['bottom'][0]):
        bx, by = c['bottom']
        print(f"  bottom=({bx},{by}) area={c['area']}")


# ============================================================
# 3) Visualize sample frames: camera + lidar side by side
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Sample visualizations")
print("=" * 60)

for sample_idx in [0, 10, 30, 50, 70]:
    img, ranges, angles = load_pair(sample_idx)
    cones, mask = detect_orange_cones(img)

    r_v, a_v, idx_v = lidar_to_xy(ranges, angles)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Camera with detected cones
    ax = axes[0]
    ax.set_title(f"Camera img_{sample_idx:03d} ({len(cones)} cones)")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    for c in cones:
        x, y, w, h = c['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
        bx, by = c['bottom']
        ax.plot(bx, by, 'r+', ms=10, mew=2)

    # Cone mask
    ax = axes[1]
    ax.set_title("Orange mask")
    ax.imshow(mask, cmap='gray')

    # Lidar BEV (using convention A for now)
    ax = axes[2]
    ax.set_title(f"Lidar BEV (conv A)")
    ax.set_aspect('equal')
    x_all = r_v * np.sin(a_v)
    y_all = r_v * np.cos(a_v)
    ax.scatter(x_all, y_all, s=3, c='steelblue', alpha=0.3)

    # Highlight close points
    close = r_v < 5.0
    ax.scatter(x_all[close], y_all[close], s=15, c='red', zorder=5)

    ax.plot(0, 0, 'ko', ms=8)
    ax.arrow(0, 0, 0, 1, head_width=0.2, fc='green', ec='green')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 10)
    ax.set_xlabel("x (right+)")
    ax.set_ylabel("y (forward+)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"02_sample_{sample_idx:03d}.png"), dpi=120)
    plt.close()
    print(f"Saved: 02_sample_{sample_idx:03d}.png")


# ============================================================
# 4) Overall cone distribution across all 80 frames
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Overall distribution across all 80 frames")
print("=" * 60)

all_cam_bottoms = []
all_lidar_close_xy = []

for i in range(80):
    img, ranges, angles = load_pair(i)

    # Camera cones
    cones, _ = detect_orange_cones(img)
    for c in cones:
        all_cam_bottoms.append((i, c['bottom'][0], c['bottom'][1], c['area']))

    # Lidar close points
    r_v, a_v, _ = lidar_to_xy(ranges, angles)
    close = r_v < 5.0
    x_close = r_v[close] * np.sin(a_v[close])
    y_close = r_v[close] * np.cos(a_v[close])
    for x, y in zip(x_close, y_close):
        all_lidar_close_xy.append((i, x, y))

all_cam = np.array(all_cam_bottoms)
all_lid = np.array(all_lidar_close_xy)

print(f"Total camera cone detections: {len(all_cam)}")
print(f"Total lidar close points: {len(all_lid)}")

if len(all_cam) > 0:
    print(f"Camera bottom X range: {all_cam[:,1].min():.0f} - {all_cam[:,1].max():.0f} (img width=640)")
    print(f"Camera bottom Y range: {all_cam[:,2].min():.0f} - {all_cam[:,2].max():.0f} (img height=480)")

if len(all_lid) > 0:
    print(f"Lidar X range: {all_lid[:,1].min():.2f} - {all_lid[:,1].max():.2f} m")
    print(f"Lidar Y range: {all_lid[:,2].min():.2f} - {all_lid[:,2].max():.2f} m")

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

if len(all_cam) > 0:
    ax = axes[0]
    ax.set_title("Camera: cone bottom positions (all frames)")
    ax.scatter(all_cam[:,1], all_cam[:,2], s=5, alpha=0.3, c=all_cam[:,0], cmap='viridis')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)  # image coords
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Frame #')

if len(all_lid) > 0:
    ax = axes[1]
    ax.set_title("Lidar: close points BEV (all frames, conv A)")
    sc = ax.scatter(all_lid[:,1], all_lid[:,2], s=3, alpha=0.2, c=all_lid[:,0], cmap='viridis')
    ax.plot(0, 0, 'ko', ms=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Frame #')

# Per-frame count
ax = axes[2]
ax.set_title("Detections per frame")
cam_counts = []
lid_counts = []
for i in range(80):
    cam_counts.append(np.sum(all_cam[:,0] == i) if len(all_cam) > 0 else 0)
    lid_counts.append(np.sum(all_lid[:,0] == i) if len(all_lid) > 0 else 0)
ax.bar(range(80), cam_counts, alpha=0.6, label='Camera cones', width=0.8)
ax.bar(range(80), lid_counts, alpha=0.4, label='Lidar close pts', width=0.8)
ax.set_xlabel("Frame #")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_distribution.png"), dpi=120)
plt.close()
print("Saved: 03_distribution.png")

print("\nDone! Check outputs in:", OUT_DIR)
