#!/usr/bin/env python3
"""
Camera-Lidar Homography Calibration.

Detects orange cones in camera images, matches them to lidar cone clusters,
and computes a homography H: pixel (u,v) -> vehicle (X,Y) meters.

Lidar convention: 0°=front, clockwise. x=r*sin(θ) (right+), y=r*cos(θ) (forward+).
"""

import numpy as np
import cv2
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel

CALIB_DIR = os.path.expanduser("~/calib_data")
OUT_DIR = os.path.expanduser("~/calib_out_vis")
os.makedirs(OUT_DIR, exist_ok=True)

# Lidar filtering parameters
CHASSIS_DIST = 0.5       # Minimum distance to exclude car chassis
MAX_CONE_DIST = 8.0      # Maximum cone distance
FORWARD_ANGLE = 80       # ±degrees from front to consider


def load_pair(idx):
    img = cv2.imread(os.path.join(CALIB_DIR, f"img_{idx:03d}.png"))
    npz = np.load(os.path.join(CALIB_DIR, f"scan_{idx:03d}.npz"))
    ranges = npz['ranges'].astype(np.float64)
    angle_min = float(npz['angle_min'])
    angle_inc = float(npz['angle_increment'])
    angles_deg = angle_min * 180 / np.pi + np.arange(len(ranges)) * (angle_inc * 180 / np.pi)
    angles_rad = np.deg2rad(angles_deg)
    return img, ranges, angles_rad, angles_deg


def lidar_cones(ranges, angles_rad, angles_deg):
    """Extract cone-like clusters from lidar data."""
    # Filter: valid, not chassis, not too far, forward-facing
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)

    # Forward region: angle in [0, FORWARD_ANGLE] or [360-FORWARD_ANGLE, 360]
    forward = (angles_deg <= FORWARD_ANGLE) | (angles_deg >= 360 - FORWARD_ANGLE)
    mask = valid & forward

    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    r = ranges[mask]
    a = angles_rad[mask]

    # Convert to XY
    x = r * np.sin(a)
    y = r * np.cos(a)

    # Cluster nearby points (simple: gap > 2° in index = new cluster)
    clusters = []
    cluster_start = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] > 2:  # gap of >2 indices = new cluster
            clusters.append((cluster_start, i))
            cluster_start = i
    clusters.append((cluster_start, len(idx)))

    cones = []
    for s, e in clusters:
        cx = np.mean(x[s:e])
        cy = np.mean(y[s:e])
        cr = np.mean(r[s:e])
        n_pts = e - s
        mean_angle_deg = np.mean(angles_deg[mask][s:e])
        # Normalize angle to [-180, 180] relative to front
        if mean_angle_deg > 180:
            mean_angle_deg -= 360
        cones.append({
            'x': cx, 'y': cy, 'r': cr,
            'angle_deg': mean_angle_deg,
            'n_pts': n_pts
        })

    return cones


def detect_camera_cones(img):
    """Detect orange cones in the lower portion of the image."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Focus on lower portion (road level, skip sky/buildings)
    # But not too restrictive since close cones can be higher in the image
    road_y_min = int(h * 0.35)  # roughly below horizon

    # Orange HSV range - tuned for simulation
    lower1 = np.array([0, 100, 80])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([165, 100, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 | mask2

    # Zero out above road
    mask[:road_y_min, :] = 0

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Aspect ratio filter: cones are taller than wide
        aspect = bh / max(bw, 1)
        if aspect < 0.5 or bw > 150:  # too flat or too wide = not a cone
            continue

        # Bottom center
        bottom_cx = x + bw // 2
        bottom_cy = y + bh

        cones.append({
            'bbox': (x, y, bw, bh),
            'bottom': (bottom_cx, bottom_cy),
            'area': area,
            'aspect': aspect
        })

    # Sort left to right by bottom x
    cones.sort(key=lambda c: c['bottom'][0])
    return cones, mask


def match_cones(cam_cones, lid_cones):
    """Match camera cones to lidar cones by angular order.

    Camera: left pixel = negative angle (left of car), right pixel = positive angle (right)
    Lidar: negative angle_deg = left, positive = right

    Sort both by angle/x-position and match.
    """
    if len(cam_cones) == 0 or len(lid_cones) == 0:
        return []

    # Sort lidar cones by angle (left to right = negative to positive)
    lid_sorted = sorted(lid_cones, key=lambda c: c['angle_deg'])

    # Camera cones already sorted left to right by x
    cam_sorted = cam_cones

    # Simple greedy matching: for each camera cone, find closest lidar cone by angular position
    # Camera x position maps roughly to angle: x=0 → far left, x=640 → far right
    # Normalize camera x to [-1, 1]
    cam_angles = [(c['bottom'][0] - 320) / 320 for c in cam_sorted]
    lid_angles_norm = [c['angle_deg'] / FORWARD_ANGLE for c in lid_sorted]

    pairs = []
    used_lid = set()

    for i, cam_c in enumerate(cam_sorted):
        best_j = -1
        best_dist = float('inf')
        for j, lid_c in enumerate(lid_sorted):
            if j in used_lid:
                continue
            d = abs(cam_angles[i] - lid_angles_norm[j])
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0 and best_dist < 1.0:
            pairs.append((cam_c, lid_sorted[best_j]))
            used_lid.add(best_j)

    return pairs


# ============================================================
# Step 1: Extract matched cone pairs from all frames
# ============================================================
print("=" * 60)
print("Extracting matched cone pairs from all 80 frames...")
print("=" * 60)

all_pixel_pts = []   # (u, v) pixel coordinates of cone bottoms
all_meter_pts = []   # (X, Y) vehicle coordinates from lidar
all_frame_info = []  # for debugging

for i in range(80):
    img, ranges, angles_rad, angles_deg = load_pair(i)

    cam_cones = detect_camera_cones(img)[0]
    lid_cones = lidar_cones(ranges, angles_rad, angles_deg)

    pairs = match_cones(cam_cones, lid_cones)

    for cam_c, lid_c in pairs:
        u, v = cam_c['bottom']
        X, Y = lid_c['x'], lid_c['y']
        all_pixel_pts.append([u, v])
        all_meter_pts.append([X, Y])
        all_frame_info.append({
            'frame': i,
            'pixel': (u, v),
            'meter': (X, Y),
            'cam_area': cam_c['area'],
            'lid_r': lid_c['r'],
            'lid_angle': lid_c['angle_deg'],
            'lid_npts': lid_c['n_pts']
        })

    if i % 20 == 0:
        print(f"  Frame {i:03d}: {len(cam_cones)} cam cones, {len(lid_cones)} lid cones, {len(pairs)} matched")

pixel_pts = np.array(all_pixel_pts, dtype=np.float64)
meter_pts = np.array(all_meter_pts, dtype=np.float64)

print(f"\nTotal matched pairs: {len(pixel_pts)}")
if len(pixel_pts) > 0:
    print(f"Pixel range: u=[{pixel_pts[:,0].min():.0f}, {pixel_pts[:,0].max():.0f}], "
          f"v=[{pixel_pts[:,1].min():.0f}, {pixel_pts[:,1].max():.0f}]")
    print(f"Meter range: X=[{meter_pts[:,0].min():.2f}, {meter_pts[:,0].max():.2f}], "
          f"Y=[{meter_pts[:,1].min():.2f}, {meter_pts[:,1].max():.2f}]")


# ============================================================
# Step 2: Visualize matched pairs on a few frames
# ============================================================
print("\n" + "=" * 60)
print("Visualizing matched pairs...")
print("=" * 60)

for sample_idx in [0, 20, 40, 60]:
    img, ranges, angles_rad, angles_deg = load_pair(sample_idx)
    cam_cones, mask = detect_camera_cones(img)
    lid_cones = lidar_cones(ranges, angles_rad, angles_deg)
    pairs = match_cones(cam_cones, lid_cones)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Camera view with matched cones
    ax = axes[0]
    ax.set_title(f"Frame {sample_idx:03d}: {len(cam_cones)} cam, {len(lid_cones)} lid, {len(pairs)} matched")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    for c in cam_cones:
        x, y, w, h = c['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=1.5)
        ax.add_patch(rect)
    for cam_c, lid_c in pairs:
        u, v = cam_c['bottom']
        ax.plot(u, v, 'r+', ms=12, mew=2)
        ax.annotate(f"({lid_c['x']:.1f},{lid_c['y']:.1f})",
                    (u, v), fontsize=7, color='yellow',
                    xytext=(5, -10), textcoords='offset points')

    # Lidar BEV with matched cones
    ax = axes[1]
    ax.set_title("Lidar BEV + matched cones")
    ax.set_aspect('equal')
    # All valid points
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    r_v = ranges[valid]
    a_v = angles_rad[valid]
    ax.scatter(r_v * np.sin(a_v), r_v * np.cos(a_v), s=3, c='steelblue', alpha=0.3)
    # Matched cone centers
    for cam_c, lid_c in pairs:
        ax.plot(lid_c['x'], lid_c['y'], 'ro', ms=8)
        ax.annotate(f"px({cam_c['bottom'][0]},{cam_c['bottom'][1]})",
                    (lid_c['x'], lid_c['y']), fontsize=7, color='red',
                    xytext=(5, 5), textcoords='offset points')
    ax.plot(0, 0, 'ko', ms=8)
    ax.arrow(0, 0, 0, 1, head_width=0.15, fc='green', ec='green')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 8)
    ax.set_xlabel("x (right+, m)")
    ax.set_ylabel("y (forward+, m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"04_matched_{sample_idx:03d}.png"), dpi=120)
    plt.close()
    print(f"Saved: 04_matched_{sample_idx:03d}.png")


# ============================================================
# Step 3: Compute Homography
# ============================================================
print("\n" + "=" * 60)
print("Computing Homography...")
print("=" * 60)

if len(pixel_pts) < 10:
    print(f"ERROR: Only {len(pixel_pts)} matched pairs. Need at least 10 for reliable homography.")
    print("Check cone detection and matching quality in visualization images.")
else:
    # Use RANSAC for robustness against mismatches
    H, inlier_mask = cv2.findHomography(pixel_pts, meter_pts, cv2.RANSAC, 0.3)

    n_inliers = int(inlier_mask.sum())
    print(f"Inliers: {n_inliers}/{len(pixel_pts)} ({100*n_inliers/len(pixel_pts):.1f}%)")
    print(f"\nHomography H (pixel → vehicle meters):")
    print(H)

    # Evaluate reprojection error
    pixel_h = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
    projected = (H @ pixel_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected_xy - meter_pts, axis=1)

    inlier_errors = errors[inlier_mask.ravel().astype(bool)]
    print(f"\nReprojection error (inliers): mean={inlier_errors.mean():.4f}m, "
          f"median={np.median(inlier_errors):.4f}m, max={inlier_errors.max():.4f}m")
    print(f"Reprojection error (all):     mean={errors.mean():.4f}m, "
          f"median={np.median(errors):.4f}m, max={errors.max():.4f}m")

    # Test: where do image corners map to?
    corners = np.array([[0, 480], [640, 480], [320, 240], [320, 480]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    mapped = (H @ corners_h.T).T
    mapped_xy = mapped[:, :2] / mapped[:, 2:3]
    print("\nImage corner mapping (pixel → meters):")
    for c, m in zip(corners, mapped_xy):
        print(f"  ({c[0]:.0f},{c[1]:.0f}) → ({m[0]:.2f},{m[1]:.2f})m")

    # Visualize: reprojection error + point distribution
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Pixel space with inlier/outlier coloring
    ax = axes[0]
    ax.set_title(f"Pixel space: {n_inliers} inliers (green) / {len(pixel_pts)-n_inliers} outliers (red)")
    inl = inlier_mask.ravel().astype(bool)
    ax.scatter(pixel_pts[inl, 0], pixel_pts[inl, 1], s=15, c='green', alpha=0.5, label='inlier')
    ax.scatter(pixel_pts[~inl, 0], pixel_pts[~inl, 1], s=15, c='red', alpha=0.5, label='outlier')
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.legend()

    # Meter space
    ax = axes[1]
    ax.set_title("Vehicle meter space")
    ax.set_aspect('equal')
    ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=15, c='green', alpha=0.5, label='inlier')
    ax.scatter(meter_pts[~inl, 0], meter_pts[~inl, 1], s=15, c='red', alpha=0.5, label='outlier')
    ax.scatter(projected_xy[inl, 0], projected_xy[inl, 1], s=10, c='blue', alpha=0.3, marker='x', label='projected')
    ax.plot(0, 0, 'ko', ms=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[2]
    ax.set_title("Reprojection error distribution")
    ax.hist(inlier_errors, bins=30, alpha=0.7, color='green', label='inliers')
    ax.hist(errors[~inl], bins=30, alpha=0.5, color='red', label='outliers')
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_homography_eval.png"), dpi=120)
    plt.close()
    print("Saved: 05_homography_eval.png")

    # Visualize H as a grid mapping
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Homography: pixel grid → vehicle meters")
    ax.set_aspect('equal')
    # Draw grid lines in pixel space, map to meters
    for u in range(0, 641, 80):
        vs = np.arange(200, 481, 5, dtype=np.float64)
        pts = np.column_stack([np.full_like(vs, u), vs, np.ones_like(vs)])
        mapped = (H @ pts.T).T
        xy = mapped[:, :2] / mapped[:, 2:3]
        # Filter out points that map to unreasonable locations
        ok = (np.abs(xy[:, 0]) < 20) & (xy[:, 1] > -5) & (xy[:, 1] < 30)
        if np.any(ok):
            ax.plot(xy[ok, 0], xy[ok, 1], 'b-', alpha=0.3, linewidth=0.8)
    for v in range(200, 481, 40):
        us = np.arange(0, 641, 5, dtype=np.float64)
        pts = np.column_stack([us, np.full_like(us, v), np.ones_like(us)])
        mapped = (H @ pts.T).T
        xy = mapped[:, :2] / mapped[:, 2:3]
        ok = (np.abs(xy[:, 0]) < 20) & (xy[:, 1] > -5) & (xy[:, 1] < 30)
        if np.any(ok):
            ax.plot(xy[ok, 0], xy[ok, 1], 'r-', alpha=0.3, linewidth=0.8)

    ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=20, c='green', zorder=5, label='data inliers')
    ax.plot(0, 0, 'ko', ms=10, zorder=5)
    ax.arrow(0, 0, 0, 1, head_width=0.2, fc='black', ec='black', zorder=5)
    ax.set_xlabel("X (right+, m)")
    ax.set_ylabel("Y (forward+, m)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 15)
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_grid_mapping.png"), dpi=120)
    plt.close()
    print("Saved: 06_grid_mapping.png")

    # Save H
    save_path = os.path.expanduser("~/xycar_ws/src/track_drive/config")
    os.makedirs(save_path, exist_ok=True)
    h_path = os.path.join(save_path, "homography.npz")
    np.savez(h_path, H=H, pixel_pts=pixel_pts, meter_pts=meter_pts,
             inlier_mask=inlier_mask,
             reprojection_error_mean=inlier_errors.mean(),
             reprojection_error_median=np.median(inlier_errors))
    print(f"\nH saved to: {h_path}")

    # Also save as JSON for easy inspection
    h_json_path = os.path.join(save_path, "homography.json")
    with open(h_json_path, 'w') as f:
        json.dump({
            'H': H.tolist(),
            'n_inliers': n_inliers,
            'n_total': len(pixel_pts),
            'reproj_error_mean_m': float(inlier_errors.mean()),
            'reproj_error_median_m': float(np.median(inlier_errors)),
            'description': 'Homography: pixel (u,v) -> vehicle (X,Y) meters. '
                           'X=right+, Y=forward+. Convention: 0deg=front, CW.',
        }, f, indent=2)
    print(f"H saved to: {h_json_path}")

print("\nDone!")
