#!/usr/bin/env python3
"""
Camera-Lidar Homography Calibration v3.

Key improvement over v2: Use green ground detection to filter cone candidates.
Cones sit on green grass, so valid cone bottoms must be adjacent to green pixels.
Also uses expected pixel size based on lidar distance for validation.
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
FORWARD_ANGLE = 70


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
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    forward = (angles_deg <= FORWARD_ANGLE) | (angles_deg >= 360 - FORWARD_ANGLE)
    mask = valid & forward
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    r = ranges[idx]
    a_rad = angles_rad[idx]
    a_deg = angles_deg[idx]

    clusters = []
    cs = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > 3:
            clusters.append((cs, i))
            cs = i
    clusters.append((cs, len(idx)))

    cones = []
    for s, e in clusters:
        if e - s < 1:
            continue
        cr = np.mean(r[s:e])
        ca_rad = np.mean(a_rad[s:e])
        ca_deg = np.mean(a_deg[s:e])
        if ca_deg > 180:
            ca_deg -= 360
        cones.append({
            'x': cr * np.sin(ca_rad),
            'y': cr * np.cos(ca_rad),
            'r': cr,
            'angle_deg': ca_deg,
            'n_pts': e - s,
        })
    return cones


def get_green_mask(img):
    """Detect green ground surface."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def get_orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 90, 70])
    upper1 = np.array([22, 255, 255])
    lower2 = np.array([165, 90, 70])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_cone_in_region(orange_mask, green_mask, u_center, search_half_w=40,
                        v_min=170, v_max=480, expected_dist=None):
    """Find a cone bottom near u_center, validated by green ground adjacency."""
    u_lo = max(0, u_center - search_half_w)
    u_hi = min(640, u_center + search_half_w)

    roi_orange = orange_mask[v_min:v_max, u_lo:u_hi].copy()
    roi_green = green_mask[v_min:v_max, u_lo:u_hi]

    if roi_orange.sum() == 0:
        return None

    contours, _ = cv2.findContours(roi_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 3 or w > 100:
            continue

        bottom_y = y + h
        bottom_cx = x + w // 2

        # Check if green pixels are adjacent below the cone bottom
        check_y_start = min(bottom_y, roi_green.shape[0] - 1)
        check_y_end = min(bottom_y + 10, roi_green.shape[0])
        if check_y_end > check_y_start:
            green_below = roi_green[check_y_start:check_y_end,
                                    max(0, bottom_cx - 15):min(roi_green.shape[1], bottom_cx + 15)]
            green_ratio = green_below.sum() / max(1, green_below.size * 255)
        else:
            green_ratio = 0.0

        # Also check green pixels adjacent to the sides
        check_x_left = max(0, x - 5)
        check_x_right = min(roi_green.shape[1], x + w + 5)
        green_side = roi_green[y:y + h, check_x_left:check_x_right]
        green_side_ratio = green_side.sum() / max(1, green_side.size * 255)

        # Combined green adjacency score
        green_score = green_ratio * 0.7 + green_side_ratio * 0.3

        # Size validation based on expected distance
        size_score = 1.0
        if expected_dist is not None:
            # Rough model: cone pixel height ≈ k / distance
            # At 2m → ~40px, at 5m → ~16px, at 8m → ~10px
            expected_h = 80.0 / expected_dist
            size_ratio = h / max(expected_h, 1)
            if 0.3 < size_ratio < 3.0:
                size_score = 1.0 - min(abs(size_ratio - 1.0), 0.7)
            else:
                size_score = 0.1

        # Prefer: bottom-most, green-adjacent, correct-sized
        score = (bottom_y / (v_max - v_min)) * 0.3 + green_score * 0.5 + size_score * 0.2

        if green_score > 0.05 or bottom_y > (v_max - v_min) * 0.7:
            candidates.append({
                'u': u_lo + bottom_cx,
                'v': v_min + bottom_y,
                'area': area,
                'h': h,
                'w': w,
                'green_score': green_score,
                'size_score': size_score,
                'score': score,
            })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c['score'])
    return (best['u'], best['v'])


# ============================================================
# Main pipeline
# ============================================================

# Use FOV from v2 as starting point, then refine
HALF_FOV_OPTIONS = list(range(50, 75))

print("Scanning FOV with green-ground-validated matching...")
best_fov = 60
best_matches = 0

for hfov in HALF_FOV_OPTIONS:
    n = 0
    for i in range(0, 80, 4):
        img, ranges, angles_rad, angles_deg = load_pair(i)
        lid_c = lidar_cones(ranges, angles_rad, angles_deg)
        omask = get_orange_mask(img)
        gmask = get_green_mask(img)

        for cone in lid_c:
            u_pred = 320 + (cone['angle_deg'] / hfov) * 320
            if 10 <= u_pred <= 630:
                result = find_cone_in_region(omask, gmask, int(u_pred),
                                             expected_dist=cone['r'])
                if result is not None:
                    n += 1
    if n > best_matches:
        best_matches = n
        best_fov = hfov

print(f"Best half_fov: {best_fov}° ({best_matches} matches on 20 sampled frames)")

# Collect all pairs
print(f"\nCollecting all matched pairs with half_fov={best_fov}°...")
all_pixel = []
all_meter = []
all_info = []

for i in range(80):
    img, ranges, angles_rad, angles_deg = load_pair(i)
    lid_c = lidar_cones(ranges, angles_rad, angles_deg)
    omask = get_orange_mask(img)
    gmask = get_green_mask(img)

    for cone in lid_c:
        u_pred = 320 + (cone['angle_deg'] / best_fov) * 320
        if not (5 <= u_pred <= 635):
            continue
        result = find_cone_in_region(omask, gmask, int(u_pred),
                                     expected_dist=cone['r'])
        if result is None:
            continue
        u, v = result
        all_pixel.append([u, v])
        all_meter.append([cone['x'], cone['y']])
        all_info.append({
            'frame': i, 'u_pred': u_pred, 'u_actual': u, 'v': v,
            'angle': cone['angle_deg'], 'dist': cone['r'],
            'mx': cone['x'], 'my': cone['y']
        })

pixel_pts = np.array(all_pixel, dtype=np.float64)
meter_pts = np.array(all_meter, dtype=np.float64)
print(f"Total matched pairs: {len(pixel_pts)}")
print(f"Pixel: u=[{pixel_pts[:,0].min():.0f},{pixel_pts[:,0].max():.0f}], "
      f"v=[{pixel_pts[:,1].min():.0f},{pixel_pts[:,1].max():.0f}]")
print(f"Meter: X=[{meter_pts[:,0].min():.2f},{meter_pts[:,0].max():.2f}], "
      f"Y=[{meter_pts[:,1].min():.2f},{meter_pts[:,1].max():.2f}]")

u_errs = [abs(info['u_pred'] - info['u_actual']) for info in all_info]
print(f"U prediction error: mean={np.mean(u_errs):.1f}px, median={np.median(u_errs):.1f}px")

# Compute homography
print("\n--- Computing Homography ---")
H, inlier_mask = cv2.findHomography(pixel_pts, meter_pts, cv2.RANSAC, 0.15)
inl = inlier_mask.ravel().astype(bool)
n_inliers = int(inl.sum())
print(f"RANSAC inliers: {n_inliers}/{len(pixel_pts)} ({100*n_inliers/len(pixel_pts):.1f}%)")

pixel_h = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
proj = (H @ pixel_h.T).T
proj_xy = proj[:, :2] / proj[:, 2:3]
errors = np.linalg.norm(proj_xy - meter_pts, axis=1)
inl_err = errors[inl]
print(f"Inlier reproj error: mean={inl_err.mean():.4f}m, median={np.median(inl_err):.4f}m, "
      f"max={inl_err.max():.4f}m")

# Re-fit with inliers
if n_inliers >= 8:
    H_final, _ = cv2.findHomography(pixel_pts[inl], meter_pts[inl], 0)
    proj2 = (H_final @ pixel_h.T).T
    proj2_xy = proj2[:, :2] / proj2[:, 2:3]
    err2 = np.linalg.norm(proj2_xy[inl] - meter_pts[inl], axis=1)
    print(f"Refined reproj (inliers): mean={err2.mean():.4f}m, median={np.median(err2):.4f}m")
else:
    H_final = H

print(f"\nFinal H:")
for row in H_final:
    print(f"  [{row[0]:15.8e}, {row[1]:15.8e}, {row[2]:15.8e}]")

# Corner mapping
print("\nCorner mapping:")
test_pts = np.array([
    [0, 480], [320, 480], [640, 480],
    [0, 400], [320, 400], [640, 400],
    [0, 300], [320, 300], [640, 300],
], dtype=np.float64)
test_h = np.hstack([test_pts, np.ones((len(test_pts), 1))])
mapped = (H_final @ test_h.T).T
mapped_xy = mapped[:, :2] / mapped[:, 2:3]
for p, m in zip(test_pts, mapped_xy):
    print(f"  px({p[0]:3.0f},{p[1]:3.0f}) → ({m[0]:+7.2f}, {m[1]:+7.2f})m")

# ============================================================
# Visualizations
# ============================================================

# 1. Sample frame overlays
for si in [0, 15, 35, 55, 75]:
    img, ranges, angles_rad, angles_deg = load_pair(si)
    lid_c = lidar_cones(ranges, angles_rad, angles_deg)
    omask = get_orange_mask(img)
    gmask = get_green_mask(img)

    frame_info = [info for info in all_info if info['frame'] == si]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    ax = axes[0]
    ax.set_title(f"Frame {si:03d}: {len(frame_info)} matches")
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for info in frame_info:
        u, v = info['u_actual'], info['v']
        color = 'lime' if inl[all_info.index(info)] else 'red'
        ax.plot(u, v, '+', color=color, ms=12, mew=2)
        ax.axvline(x=info['u_pred'], color='cyan', alpha=0.2, lw=1)
        if inl[all_info.index(info)]:
            ax.annotate(f"({info['mx']:.1f},{info['my']:.1f})",
                        (u, v), fontsize=7, color='yellow',
                        xytext=(3, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6))

    ax = axes[1]
    ax.set_title("Green ground mask")
    ax.imshow(gmask, cmap='Greens')

    ax = axes[2]
    ax.set_title("Lidar BEV")
    ax.set_aspect('equal')
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    rv, av = ranges[valid], angles_rad[valid]
    ax.scatter(rv * np.sin(av), rv * np.cos(av), s=3, c='steelblue', alpha=0.3)
    for cone in lid_c:
        ax.plot(cone['x'], cone['y'], 'go', ms=6, alpha=0.5)
    for info in frame_info:
        if inl[all_info.index(info)]:
            ax.plot(info['mx'], info['my'], 'ro', ms=8)
    ax.plot(0, 0, 'ko', ms=8)
    ax.arrow(0, 0, 0, 1, head_width=0.15, fc='green', ec='green')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 10)
    ax.set_xlabel("X (right+, m)")
    ax.set_ylabel("Y (forward+, m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"10_v3_{si:03d}.png"), dpi=120)
    plt.close()
    print(f"Saved: 10_v3_{si:03d}.png")

# 2. Evaluation summary
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
ax.set_title(f"Pixel: {n_inliers} inliers / {len(pixel_pts)-n_inliers} outliers")
ax.scatter(pixel_pts[inl, 0], pixel_pts[inl, 1], s=12, c='green', alpha=0.6, label='inlier')
ax.scatter(pixel_pts[~inl, 0], pixel_pts[~inl, 1], s=8, c='red', alpha=0.3, label='outlier')
ax.set_xlim(0, 640); ax.set_ylim(480, 0)
ax.set_xlabel("u"); ax.set_ylabel("v"); ax.legend()

ax = axes[1]
ax.set_title("Vehicle meters (inliers: truth vs proj)")
ax.set_aspect('equal')
proj_final = (H_final @ pixel_h.T).T
proj_final_xy = proj_final[:, :2] / proj_final[:, 2:3]
ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=12, c='green', alpha=0.6, label='truth')
ax.scatter(proj_final_xy[inl, 0], proj_final_xy[inl, 1], s=8, c='blue', alpha=0.4,
           marker='x', label='projected')
for i in np.where(inl)[0]:
    ax.plot([meter_pts[i, 0], proj_final_xy[i, 0]],
            [meter_pts[i, 1], proj_final_xy[i, 1]], 'k-', alpha=0.15, lw=0.5)
ax.plot(0, 0, 'ko', ms=8)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.set_title("Reprojection error")
final_err = np.linalg.norm(proj_final_xy - meter_pts, axis=1)
ax.hist(final_err[inl], bins=30, alpha=0.7, color='green', label=f'inlier (n={n_inliers})')
if (~inl).sum() > 0:
    ax.hist(final_err[~inl], bins=30, alpha=0.4, color='red', label=f'outlier (n={int((~inl).sum())})')
ax.set_xlabel("Error (m)"); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_v3_eval.png"), dpi=120)
plt.close()
print("Saved: 11_v3_eval.png")

# 3. Grid mapping
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Homography grid: pixel → vehicle meters")
ax.set_aspect('equal')
for u in range(0, 641, 80):
    vs = np.arange(240, 481, 2, dtype=np.float64)
    pts = np.column_stack([np.full_like(vs, u), vs, np.ones_like(vs)])
    m = (H_final @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 15) & (xy[:, 1] > -1) & (xy[:, 1] < 15)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'b-', alpha=0.4, lw=1)
        if np.any(ok):
            ax.annotate(f'u={u}', xy=(xy[ok][-1, 0], xy[ok][-1, 1]),
                        fontsize=7, color='blue', alpha=0.6)
for v in range(250, 481, 30):
    us = np.arange(0, 641, 2, dtype=np.float64)
    pts = np.column_stack([us, np.full_like(us, v), np.ones_like(us)])
    m = (H_final @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 15) & (xy[:, 1] > -1) & (xy[:, 1] < 15)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'r-', alpha=0.3, lw=0.8)
        mid = len(xy[ok]) // 2
        ax.annotate(f'v={v}', xy=(xy[ok][mid, 0], xy[ok][mid, 1]),
                    fontsize=7, color='red', alpha=0.6)

ax.scatter(meter_pts[inl, 0], meter_pts[inl, 1], s=20, c='green', zorder=5,
           edgecolors='darkgreen', linewidth=0.5, label=f'inliers (n={n_inliers})')
ax.plot(0, 0, 'ko', ms=10, zorder=5)
ax.arrow(0, 0, 0, 1.5, head_width=0.2, fc='black', ec='black', zorder=5)
ax.set_xlabel("X (right+, m)")
ax.set_ylabel("Y (forward+, m)")
ax.set_xlim(-8, 8)
ax.set_ylim(-1, 12)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "12_v3_grid.png"), dpi=150)
plt.close()
print("Saved: 12_v3_grid.png")

# ============================================================
# Save
# ============================================================
save_path = os.path.expanduser("~/xycar_ws/src/track_drive/config")
os.makedirs(save_path, exist_ok=True)

np.savez(os.path.join(save_path, "homography.npz"),
         H=H_final,
         pixel_pts=pixel_pts[inl],
         meter_pts=meter_pts[inl],
         reprojection_error_mean=float(inl_err.mean()),
         reprojection_error_median=float(np.median(inl_err)),
         half_fov_deg=float(best_fov))

with open(os.path.join(save_path, "homography.json"), 'w') as f:
    json.dump({
        'H': H_final.tolist(),
        'n_inliers': n_inliers,
        'n_total': int(len(pixel_pts)),
        'reproj_error_mean_m': float(inl_err.mean()),
        'reproj_error_median_m': float(np.median(inl_err)),
        'half_fov_deg': float(best_fov),
        'coordinate_system': 'X=right+, Y=forward+, lidar 0deg=front CW',
        'usage': 'pt_m = (H @ [u, v, 1]) -> normalize by 3rd component -> (X, Y) meters',
    }, f, indent=2)

print(f"\nSaved to: {save_path}/homography.{{npz,json}}")
print("Done!")
