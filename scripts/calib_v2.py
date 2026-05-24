#!/usr/bin/env python3
"""
Camera-Lidar Homography Calibration v2.

Strategy change: Use lidar as ground truth to guide camera search.
1. Find cone clusters in lidar (reliable, 4-7 per frame)
2. Estimate which pixel column each lidar cone should appear in
3. Search for orange blobs only in that column neighborhood
4. Match bottom pixel to lidar XY
"""

import numpy as np
import cv2
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALIB_DIR = os.path.expanduser("~/calib_data")
OUT_DIR = os.path.expanduser("~/calib_out_vis")
os.makedirs(OUT_DIR, exist_ok=True)

CHASSIS_DIST = 0.5
MAX_CONE_DIST = 10.0
FORWARD_ANGLE = 80  # ±degrees from front


def load_pair(idx):
    img = cv2.imread(os.path.join(CALIB_DIR, f"img_{idx:03d}.png"))
    npz = np.load(os.path.join(CALIB_DIR, f"scan_{idx:03d}.npz"))
    ranges = npz['ranges'].astype(np.float64)
    angle_min = float(npz['angle_min'])
    angle_inc = float(npz['angle_increment'])
    angles_deg = angle_min * 180.0 / np.pi + np.arange(len(ranges)) * (angle_inc * 180.0 / np.pi)
    angles_rad = np.deg2rad(angles_deg)
    return img, ranges, angles_rad, angles_deg


def lidar_cones(ranges, angles_rad, angles_deg):
    """Extract cone clusters from lidar: forward region, skip chassis."""
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    forward = (angles_deg <= FORWARD_ANGLE) | (angles_deg >= 360 - FORWARD_ANGLE)
    mask = valid & forward
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    r = ranges[idx]
    a_rad = angles_rad[idx]
    a_deg = angles_deg[idx]

    # Cluster by angular gap (>3° gap = new cluster)
    clusters = []
    cs = 0
    for i in range(1, len(idx)):
        gap = idx[i] - idx[i - 1]
        if gap > 3:
            clusters.append((cs, i))
            cs = i
    clusters.append((cs, len(idx)))

    cones = []
    for s, e in clusters:
        n = e - s
        if n < 1:
            continue
        cr = np.mean(r[s:e])
        ca_rad = np.mean(a_rad[s:e])
        ca_deg = np.mean(a_deg[s:e])
        # Normalize angle: convert to signed (negative=left, positive=right)
        if ca_deg > 180:
            ca_deg -= 360
        cx = cr * np.sin(ca_rad)
        cy = cr * np.cos(ca_rad)
        cones.append({
            'x': cx, 'y': cy, 'r': cr,
            'angle_deg': ca_deg,  # signed: neg=left, pos=right
            'n_pts': n,
            'indices': idx[s:e].tolist()
        })

    return cones


# ============================================================
# Step 1: Estimate camera FOV by analyzing cone angular spread
# ============================================================
print("=" * 60)
print("Step 1: Estimating camera FOV and angular mapping")
print("=" * 60)

# First pass: collect all lidar cones and their rough image positions
# We need to figure out the camera's angular FOV and center offset
# Strategy: check if the extreme-angle lidar cones are visible in camera

# Collect stats from all frames
all_lid_cones = []
for i in range(80):
    _, ranges, angles_rad, angles_deg = load_pair(i)
    lc = lidar_cones(ranges, angles_rad, angles_deg)
    for c in lc:
        all_lid_cones.append({'frame': i, **c})

lid_angles = [c['angle_deg'] for c in all_lid_cones]
print(f"Total lidar cone clusters across 80 frames: {len(all_lid_cones)}")
print(f"Lidar cone angle range: [{min(lid_angles):.1f}°, {max(lid_angles):.1f}°]")
print(f"Lidar cone distance range: [{min(c['r'] for c in all_lid_cones):.2f}, "
      f"{max(c['r'] for c in all_lid_cones):.2f}] m")


# ============================================================
# Step 2: Direct approach - detect ALL orange in lower image,
#          then match each lidar cone to nearest orange blob
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Improved cone detection + lidar-guided matching")
print("=" * 60)


def get_orange_mask(img):
    """Get orange pixel mask for the full image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 80])
    upper1 = np.array([22, 255, 255])
    lower2 = np.array([165, 100, 80])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_cone_bottom_in_column_range(mask, u_min, u_max, v_min=168, v_max=480):
    """Find the bottom-most orange pixel in a column range.

    Returns (u, v) of the cone bottom center, or None.
    """
    roi = mask[v_min:v_max, u_min:u_max]
    if roi.sum() == 0:
        return None

    # Find contours in ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the contour with the lowest bottom point (most likely the cone base)
    best = None
    best_bottom_y = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bottom_y = y + h
        if bottom_y > best_bottom_y:
            best_bottom_y = bottom_y
            best = (u_min + x + w // 2, v_min + y + h)
            best_area = area

    return best


def find_cone_bottom_v2(mask, u_center, search_width=60, v_min=168, v_max=480):
    """Find cone bottom near a predicted column."""
    u_min = max(0, u_center - search_width)
    u_max = min(640, u_center + search_width)
    return find_cone_bottom_in_column_range(mask, u_min, u_max, v_min, v_max)


# ============================================================
# Step 3: Iterative FOV estimation
# ============================================================
# Try different FOV values and see which gives the best matching
# Camera model: u = 320 + (angle_deg / half_fov) * 320
# where angle_deg is signed (neg=left, pos=right)

print("\nTrying different camera FOV values...")

best_fov = None
best_n_matches = 0
best_error = float('inf')

for half_fov in range(25, 65, 5):
    n_matches = 0
    for i in range(0, 80, 5):  # Sample every 5th frame
        img, ranges, angles_rad, angles_deg = load_pair(i)
        lid_c = lidar_cones(ranges, angles_rad, angles_deg)
        mask = get_orange_mask(img)

        for cone in lid_c:
            # Predict pixel column
            u_pred = 320 + (cone['angle_deg'] / half_fov) * 320
            if 0 <= u_pred <= 640:
                bottom = find_cone_bottom_v2(mask, int(u_pred), search_width=50)
                if bottom is not None:
                    n_matches += 1

    print(f"  half_fov={half_fov}°: {n_matches} matches (16 frames)")
    if n_matches > best_n_matches:
        best_n_matches = n_matches
        best_fov = half_fov

print(f"\nBest half_fov: {best_fov}° (total FOV: {2*best_fov}°)")

# Fine-tune around best
print("Fine-tuning...")
for half_fov_10 in range(max(20, best_fov - 8), best_fov + 9):
    half_fov = half_fov_10
    n_matches = 0
    for i in range(80):
        img, ranges, angles_rad, angles_deg = load_pair(i)
        lid_c = lidar_cones(ranges, angles_rad, angles_deg)
        mask = get_orange_mask(img)

        for cone in lid_c:
            u_pred = 320 + (cone['angle_deg'] / half_fov) * 320
            if 0 <= u_pred <= 640:
                bottom = find_cone_bottom_v2(mask, int(u_pred), search_width=40)
                if bottom is not None:
                    n_matches += 1

    if n_matches > best_n_matches:
        best_n_matches = n_matches
        best_fov = half_fov
    print(f"  half_fov={half_fov}°: {n_matches} matches (80 frames)")

HALF_FOV = best_fov
print(f"\nFinal half_fov: {HALF_FOV}° (total FOV: {2*HALF_FOV}°)")


# ============================================================
# Step 4: Collect all matched pairs with optimal FOV
# ============================================================
print("\n" + "=" * 60)
print(f"Step 4: Collecting matched pairs (half_fov={HALF_FOV}°)")
print("=" * 60)

all_pixel = []
all_meter = []
all_info = []

for i in range(80):
    img, ranges, angles_rad, angles_deg = load_pair(i)
    lid_c = lidar_cones(ranges, angles_rad, angles_deg)
    mask = get_orange_mask(img)

    for cone in lid_c:
        u_pred = 320 + (cone['angle_deg'] / HALF_FOV) * 320
        if not (10 <= u_pred <= 630):
            continue

        bottom = find_cone_bottom_v2(mask, int(u_pred), search_width=40)
        if bottom is None:
            continue

        u, v = bottom
        X, Y = cone['x'], cone['y']
        all_pixel.append([u, v])
        all_meter.append([X, Y])
        all_info.append({
            'frame': i, 'pixel': (u, v), 'meter': (X, Y),
            'u_pred': u_pred, 'u_actual': u,
            'angle_deg': cone['angle_deg'], 'distance': cone['r']
        })

pixel_pts = np.array(all_pixel, dtype=np.float64)
meter_pts = np.array(all_meter, dtype=np.float64)

print(f"Total matched pairs: {len(pixel_pts)}")
print(f"Pixel range: u=[{pixel_pts[:,0].min():.0f}, {pixel_pts[:,0].max():.0f}], "
      f"v=[{pixel_pts[:,1].min():.0f}, {pixel_pts[:,1].max():.0f}]")
print(f"Meter range: X=[{meter_pts[:,0].min():.2f}, {meter_pts[:,0].max():.2f}], "
      f"Y=[{meter_pts[:,1].min():.2f}, {meter_pts[:,1].max():.2f}]")

# Check u_pred vs u_actual correlation
u_preds = [info['u_pred'] for info in all_info]
u_actuals = [info['u_actual'] for info in all_info]
u_errs = [abs(info['u_pred'] - info['u_actual']) for info in all_info]
print(f"U prediction error: mean={np.mean(u_errs):.1f}px, median={np.median(u_errs):.1f}px")


# ============================================================
# Step 5: Visualize matches on sample frames
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Visualizing matches")
print("=" * 60)

for sample_idx in [0, 10, 25, 40, 55, 70]:
    img, ranges, angles_rad, angles_deg = load_pair(sample_idx)
    lid_c = lidar_cones(ranges, angles_rad, angles_deg)
    mask = get_orange_mask(img)

    frame_pairs = [(info, info['pixel'], info['meter'])
                   for info in all_info if info['frame'] == sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Camera with matches
    ax = axes[0]
    ax.set_title(f"Frame {sample_idx:03d}: {len(frame_pairs)} matches")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    for info, px, mt in frame_pairs:
        u, v = px
        ax.plot(u, v, 'r+', ms=12, mew=2)
        # Draw predicted column
        ax.axvline(x=info['u_pred'], color='cyan', alpha=0.3, linewidth=1)
        ax.annotate(f"({mt[0]:.1f},{mt[1]:.1f})",
                    (u, v), fontsize=7, color='yellow',
                    xytext=(5, -12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Orange mask
    ax = axes[1]
    ax.set_title("Orange mask (lower half)")
    ax.imshow(mask, cmap='gray')

    # Lidar BEV
    ax = axes[2]
    ax.set_title("Lidar BEV")
    ax.set_aspect('equal')
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    r_v = ranges[valid]
    a_v = angles_rad[valid]
    ax.scatter(r_v * np.sin(a_v), r_v * np.cos(a_v), s=3, c='steelblue', alpha=0.3)
    for cone in lid_c:
        ax.plot(cone['x'], cone['y'], 'go', ms=6, alpha=0.5)
    for info, px, mt in frame_pairs:
        ax.plot(mt[0], mt[1], 'ro', ms=8)
    ax.plot(0, 0, 'ko', ms=8)
    ax.arrow(0, 0, 0, 1, head_width=0.15, fc='green', ec='green')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 10)
    ax.set_xlabel("x (right+, m)")
    ax.set_ylabel("y (forward+, m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"07_v2_matched_{sample_idx:03d}.png"), dpi=120)
    plt.close()
    print(f"Saved: 07_v2_matched_{sample_idx:03d}.png")


# ============================================================
# Step 6: Compute Homography with RANSAC
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Computing Homography")
print("=" * 60)

if len(pixel_pts) < 10:
    print(f"ERROR: Only {len(pixel_pts)} pairs. Need more.")
    exit(1)

H, inlier_mask = cv2.findHomography(pixel_pts, meter_pts, cv2.RANSAC, 0.2)
inl = inlier_mask.ravel().astype(bool)
n_inliers = int(inl.sum())

print(f"Inliers: {n_inliers}/{len(pixel_pts)} ({100*n_inliers/len(pixel_pts):.1f}%)")

# Reprojection error
pixel_h = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
proj = (H @ pixel_h.T).T
proj_xy = proj[:, :2] / proj[:, 2:3]
errors = np.linalg.norm(proj_xy - meter_pts, axis=1)
inl_errors = errors[inl]

print(f"\nHomography H:")
print(H)
print(f"\nReprojection error (inliers): mean={inl_errors.mean():.4f}m, "
      f"median={np.median(inl_errors):.4f}m, max={inl_errors.max():.4f}m")
print(f"Reprojection error (all):     mean={errors.mean():.4f}m, "
      f"median={np.median(errors):.4f}m")

# Corner mapping sanity check
print("\nCorner mapping (pixel → meters):")
test_pts = np.array([
    [0, 480], [640, 480], [320, 480],  # bottom row
    [0, 350], [640, 350], [320, 350],  # mid
    [0, 240], [640, 240], [320, 240],  # upper-mid
], dtype=np.float64)
test_h = np.hstack([test_pts, np.ones((len(test_pts), 1))])
mapped = (H @ test_h.T).T
mapped_xy = mapped[:, :2] / mapped[:, 2:3]
for p, m in zip(test_pts, mapped_xy):
    print(f"  ({p[0]:3.0f},{p[1]:3.0f}) → ({m[0]:6.2f}, {m[1]:6.2f})m")


# ============================================================
# Step 7: Re-fit with inliers only (least-squares) for better H
# ============================================================
if n_inliers >= 8:
    print("\nRe-fitting with inliers only...")
    H_refined, _ = cv2.findHomography(pixel_pts[inl], meter_pts[inl], 0)  # method=0 = least squares

    proj2 = (H_refined @ pixel_h.T).T
    proj2_xy = proj2[:, :2] / proj2[:, 2:3]
    errors2 = np.linalg.norm(proj2_xy - meter_pts, axis=1)
    inl_errors2 = errors2[inl]

    print(f"Refined reprojection (inliers): mean={inl_errors2.mean():.4f}m, "
          f"median={np.median(inl_errors2):.4f}m")

    H_final = H_refined
else:
    H_final = H

print(f"\nFinal H:")
print(H_final)

# ============================================================
# Step 8: Evaluation plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
ax.set_title(f"Pixel: {n_inliers} inliers / {len(pixel_pts)-n_inliers} outliers")
ax.scatter(pixel_pts[inl, 0], pixel_pts[inl, 1], s=10, c='green', alpha=0.5, label='inlier')
ax.scatter(pixel_pts[~inl, 0], pixel_pts[~inl, 1], s=10, c='red', alpha=0.5, label='outlier')
ax.set_xlim(0, 640)
ax.set_ylim(480, 0)
ax.set_xlabel("u")
ax.set_ylabel("v")
ax.legend()

ax = axes[1]
ax.set_title("Vehicle meters")
ax.set_aspect('equal')
ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=10, c='green', alpha=0.5, label='truth')
proj_final = (H_final @ pixel_h.T).T
proj_final_xy = proj_final[:, :2] / proj_final[:, 2:3]
ax.scatter(proj_final_xy[inl, 0], proj_final_xy[inl, 1], s=8, c='blue', alpha=0.3,
           marker='x', label='proj')
ax.plot(0, 0, 'ko', ms=8)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.set_title("Error distribution")
final_err = np.linalg.norm(proj_final_xy - meter_pts, axis=1)
ax.hist(final_err[inl], bins=30, alpha=0.7, color='green', label='inlier')
ax.hist(final_err[~inl], bins=30, alpha=0.5, color='red', label='outlier')
ax.set_xlabel("Error (m)")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_v2_homography_eval.png"), dpi=120)
plt.close()
print("Saved: 08_v2_homography_eval.png")

# Grid mapping visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("H grid: pixel → vehicle meters")
ax.set_aspect('equal')
for u in range(0, 641, 80):
    vs = np.arange(200, 481, 2, dtype=np.float64)
    pts = np.column_stack([np.full_like(vs, u), vs, np.ones_like(vs)])
    m = (H_final @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 15) & (xy[:, 1] > -2) & (xy[:, 1] < 20)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'b-', alpha=0.3, lw=0.8)
for v in range(200, 481, 40):
    us = np.arange(0, 641, 2, dtype=np.float64)
    pts = np.column_stack([us, np.full_like(us, v), np.ones_like(us)])
    m = (H_final @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 15) & (xy[:, 1] > -2) & (xy[:, 1] < 20)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'r-', alpha=0.3, lw=0.8)
ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=15, c='green', zorder=5, label='inliers')
ax.plot(0, 0, 'ko', ms=10, zorder=5)
ax.arrow(0, 0, 0, 1.5, head_width=0.2, fc='black', ec='black', zorder=5)
ax.set_xlabel("X (right+, m)")
ax.set_ylabel("Y (forward+, m)")
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 15)
ax.grid(True, alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_v2_grid.png"), dpi=120)
plt.close()
print("Saved: 09_v2_grid.png")

# ============================================================
# Save final H
# ============================================================
save_path = os.path.expanduser("~/xycar_ws/src/track_drive/config")
os.makedirs(save_path, exist_ok=True)

np.savez(os.path.join(save_path, "homography.npz"),
         H=H_final,
         pixel_pts=pixel_pts[inl],
         meter_pts=meter_pts[inl],
         reprojection_error_mean=float(inl_errors.mean()),
         reprojection_error_median=float(np.median(inl_errors)),
         half_fov_deg=float(HALF_FOV))

with open(os.path.join(save_path, "homography.json"), 'w') as f:
    json.dump({
        'H': H_final.tolist(),
        'n_inliers': n_inliers,
        'n_total': len(pixel_pts),
        'reproj_error_mean_m': float(inl_errors.mean()),
        'reproj_error_median_m': float(np.median(inl_errors)),
        'half_fov_deg': float(HALF_FOV),
        'description': 'Homography pixel(u,v)->vehicle(X,Y) meters. X=right+, Y=forward+.',
    }, f, indent=2)

print(f"\nSaved to: {save_path}/homography.{{npz,json}}")
print("Done!")
