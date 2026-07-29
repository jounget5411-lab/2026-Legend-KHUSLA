#!/usr/bin/env python3
"""
Camera-Lidar Homography Calibration v4.

Strategy: Match only the NEAREST left + right cone per frame.
These are the most reliably detectable in both modalities.
- Lidar: find two closest clusters with opposite sign angles
- Camera: find two lowest orange blobs on opposite sides of center
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


def load_pair(idx):
    img = cv2.imread(os.path.join(CALIB_DIR, f"img_{idx:03d}.png"))
    npz = np.load(os.path.join(CALIB_DIR, f"scan_{idx:03d}.npz"))
    ranges = npz['ranges'].astype(np.float64)
    angle_min = float(npz['angle_min'])
    angle_inc = float(npz['angle_increment'])
    angles_deg = angle_min * 180.0 / np.pi + np.arange(len(ranges)) * (angle_inc * 180.0 / np.pi)
    angles_rad = np.deg2rad(angles_deg)
    return img, ranges, angles_rad, angles_deg


def lidar_all_cones(ranges, angles_rad, angles_deg, max_angle=80):
    """Extract all cone clusters in forward region."""
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    forward = (angles_deg <= max_angle) | (angles_deg >= 360 - max_angle)
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
        if idx[i] - idx[i-1] > 3:
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
    return sorted(cones, key=lambda c: c['r'])  # sort by distance (nearest first)


def camera_all_cone_candidates(img, min_y=200):
    """Find all orange cone candidate bottoms in lower image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 60])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 80, 60])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask[:min_y, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 5 or w > 120:
            continue
        # Aspect ratio: cone should be taller than wide
        if h < w * 0.4:
            continue

        bottom_cx = x + w // 2
        bottom_cy = y + h

        candidates.append({
            'u': bottom_cx,
            'v': bottom_cy,
            'area': area,
            'h': h,
            'w': w,
            'bbox': (x, y, w, h),
        })

    # Sort by v descending (bottom-most first = closest)
    candidates.sort(key=lambda c: -c['v'])
    return candidates, mask


def match_nearest_pair(cam_candidates, lid_cones):
    """Match camera and lidar cones by finding best left-right pair.

    Returns list of (cam_candidate, lid_cone) pairs.
    """
    if len(cam_candidates) < 2 or len(lid_cones) < 2:
        return []

    # Separate lidar cones into left (negative angle) and right (positive)
    lid_left = [c for c in lid_cones if c['angle_deg'] < -3]
    lid_right = [c for c in lid_cones if c['angle_deg'] > 3]

    if not lid_left or not lid_right:
        return []

    # Separate camera candidates into left (<280) and right (>360) of image center
    cam_left = [c for c in cam_candidates if c['u'] < 280]
    cam_right = [c for c in cam_candidates if c['u'] > 360]

    pairs = []

    # For each side, match the nearest lidar cone to the lowest camera blob
    for lid_list, cam_list, side in [(lid_left, cam_left, 'left'), (lid_right, cam_right, 'right')]:
        if not cam_list:
            continue
        # Try matching each lidar cone (sorted by distance) with the best camera candidate
        for lid_c in lid_list:
            # Closer cones appear lower in the image (higher v)
            # Find camera candidate with highest v (closest) that hasn't been used
            best_cam = None
            best_score = -1
            for cam_c in cam_list:
                # Score: prefer bottom-most (high v), reasonable area
                score = cam_c['v'] / 480.0
                if score > best_score:
                    best_score = score
                    best_cam = cam_c

            if best_cam is not None:
                pairs.append((best_cam, lid_c))
                cam_list.remove(best_cam)

    return pairs


# ============================================================
# Step 1: Collect all nearest-pair matches
# ============================================================
print("=" * 60)
print("Collecting nearest left/right cone matches from all 80 frames")
print("=" * 60)

all_pixel = []
all_meter = []
all_info = []

for i in range(80):
    img, ranges, angles_rad, angles_deg = load_pair(i)
    lid_cones = lidar_all_cones(ranges, angles_rad, angles_deg)
    cam_cands, mask = camera_all_cone_candidates(img)

    pairs = match_nearest_pair(cam_cands, lid_cones)

    for cam_c, lid_c in pairs:
        all_pixel.append([cam_c['u'], cam_c['v']])
        all_meter.append([lid_c['x'], lid_c['y']])
        all_info.append({
            'frame': i,
            'cam': cam_c,
            'lid': lid_c,
        })

    if i % 20 == 0:
        print(f"  Frame {i:03d}: {len(cam_cands)} cam candidates, "
              f"{len(lid_cones)} lid cones, {len(pairs)} pairs")

pixel_pts = np.array(all_pixel, dtype=np.float64)
meter_pts = np.array(all_meter, dtype=np.float64)

print(f"\nTotal pairs: {len(pixel_pts)}")
print(f"Pixel: u=[{pixel_pts[:,0].min():.0f},{pixel_pts[:,0].max():.0f}], "
      f"v=[{pixel_pts[:,1].min():.0f},{pixel_pts[:,1].max():.0f}]")
print(f"Meter: X=[{meter_pts[:,0].min():.2f},{meter_pts[:,0].max():.2f}], "
      f"Y=[{meter_pts[:,1].min():.2f},{meter_pts[:,1].max():.2f}]")


# ============================================================
# Step 2: Also add MORE correspondences from further cones
# Now that we have a rough mapping, use it to predict pixel
# locations for ALL lidar cones and verify
# ============================================================
print("\n--- Initial Homography from nearest pairs ---")
H_init, init_mask = cv2.findHomography(pixel_pts, meter_pts, cv2.RANSAC, 0.2)
init_inl = init_mask.ravel().astype(bool)
print(f"Initial RANSAC: {init_inl.sum()}/{len(pixel_pts)} inliers "
      f"({100*init_inl.sum()/len(pixel_pts):.1f}%)")

# Compute inverse H for meter→pixel prediction
H_inv = np.linalg.inv(H_init)

print("\n--- Expanding matches using inverse H prediction ---")
expanded_pixel = list(pixel_pts[init_inl])
expanded_meter = list(meter_pts[init_inl])
expanded_info = []

for i in range(80):
    img, ranges, angles_rad, angles_deg = load_pair(i)
    lid_cones = lidar_all_cones(ranges, angles_rad, angles_deg)
    cam_cands, mask = camera_all_cone_candidates(img)

    for lid_c in lid_cones:
        # Predict pixel position using H_inv
        meter_h = np.array([lid_c['x'], lid_c['y'], 1.0])
        pred = H_inv @ meter_h
        u_pred = pred[0] / pred[2]
        v_pred = pred[1] / pred[2]

        if not (0 <= u_pred <= 640 and 200 <= v_pred <= 480):
            continue

        # Find nearest camera candidate to predicted position
        best_cam = None
        best_dist = float('inf')
        for cam_c in cam_cands:
            d = np.sqrt((cam_c['u'] - u_pred)**2 + (cam_c['v'] - v_pred)**2)
            if d < best_dist and d < 40:  # within 40px of prediction
                best_dist = d
                best_cam = cam_c

        if best_cam is not None:
            expanded_pixel.append([best_cam['u'], best_cam['v']])
            expanded_meter.append([lid_c['x'], lid_c['y']])
            expanded_info.append({
                'frame': i,
                'u_pred': u_pred,
                'v_pred': v_pred,
                'u_actual': best_cam['u'],
                'v_actual': best_cam['v'],
                'dist': best_dist,
                'lid': lid_c,
            })

exp_pixel = np.array(expanded_pixel, dtype=np.float64)
exp_meter = np.array(expanded_meter, dtype=np.float64)
print(f"Expanded pairs: {len(exp_pixel)}")

# Remove duplicates (same pixel within 5px)
unique_mask = np.ones(len(exp_pixel), dtype=bool)
for i in range(len(exp_pixel)):
    if not unique_mask[i]:
        continue
    for j in range(i+1, len(exp_pixel)):
        if not unique_mask[j]:
            continue
        if np.linalg.norm(exp_pixel[i] - exp_pixel[j]) < 5:
            unique_mask[j] = False

exp_pixel = exp_pixel[unique_mask]
exp_meter = exp_meter[unique_mask]
print(f"After dedup: {len(exp_pixel)}")

# ============================================================
# Step 3: Final homography with expanded set
# ============================================================
print("\n--- Final Homography ---")
H_final, final_mask = cv2.findHomography(exp_pixel, exp_meter, cv2.RANSAC, 0.15)
finl = final_mask.ravel().astype(bool)
n_inliers = int(finl.sum())
print(f"RANSAC: {n_inliers}/{len(exp_pixel)} inliers ({100*n_inliers/len(exp_pixel):.1f}%)")

exp_h = np.hstack([exp_pixel, np.ones((len(exp_pixel), 1))])
proj = (H_final @ exp_h.T).T
proj_xy = proj[:, :2] / proj[:, 2:3]
errors = np.linalg.norm(proj_xy - exp_meter, axis=1)
inl_err = errors[finl]
print(f"Inlier error: mean={inl_err.mean():.4f}m, median={np.median(inl_err):.4f}m, "
      f"max={inl_err.max():.4f}m")

# Re-fit with inliers only
H_ls, _ = cv2.findHomography(exp_pixel[finl], exp_meter[finl], 0)
proj_ls = (H_ls @ exp_h.T).T
proj_ls_xy = proj_ls[:, :2] / proj_ls[:, 2:3]
err_ls = np.linalg.norm(proj_ls_xy[finl] - exp_meter[finl], axis=1)
print(f"LS refined: mean={err_ls.mean():.4f}m, median={np.median(err_ls):.4f}m")

H_out = H_ls

print(f"\nH:")
for row in H_out:
    print(f"  [{row[0]:+15.8e}, {row[1]:+15.8e}, {row[2]:+15.8e}]")

# Sanity check
print("\nCorner mapping:")
tests = np.array([[0,480],[320,480],[640,480],
                   [0,380],[320,380],[640,380],
                   [0,300],[320,300],[640,300]], dtype=np.float64)
tests_h = np.hstack([tests, np.ones((len(tests),1))])
mapped = (H_out @ tests_h.T).T
mxy = mapped[:, :2] / mapped[:, 2:3]
for p, m in zip(tests, mxy):
    print(f"  px({p[0]:3.0f},{p[1]:3.0f}) → ({m[0]:+7.2f},{m[1]:+7.2f})m")

# ============================================================
# Visualizations
# ============================================================
print("\n--- Generating visualizations ---")

for si in [0, 10, 25, 40, 55, 70]:
    img, ranges, angles_rad, angles_deg = load_pair(si)
    lid_cones = lidar_all_cones(ranges, angles_rad, angles_deg)
    cam_cands, mask = camera_all_cone_candidates(img)

    # Use H_inv to predict pixel locations of all lidar cones
    H_inv_final = np.linalg.inv(H_out)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    ax = axes[0]
    ax.set_title(f"Frame {si:03d}")
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Show all camera candidates
    for c in cam_cands:
        x, y, w, h = c['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, ec='lime', lw=1, alpha=0.4)
        ax.add_patch(rect)
    # Show predicted locations from lidar
    for lc in lid_cones:
        m_h = np.array([lc['x'], lc['y'], 1.0])
        pred = H_inv_final @ m_h
        up, vp = pred[0]/pred[2], pred[1]/pred[2]
        if 0 <= up <= 640 and 0 <= vp <= 480:
            ax.plot(up, vp, 'c^', ms=10, mew=2, mfc='none')
            ax.annotate(f"d={lc['r']:.1f}m", (up, vp), fontsize=6, color='cyan',
                        xytext=(3, -8), textcoords='offset points')

    ax = axes[1]
    ax.set_title("Orange mask")
    ax.imshow(mask, cmap='gray')

    ax = axes[2]
    ax.set_title("Lidar BEV")
    ax.set_aspect('equal')
    valid = np.isfinite(ranges) & (ranges > CHASSIS_DIST) & (ranges < MAX_CONE_DIST)
    rv, av = ranges[valid], angles_rad[valid]
    ax.scatter(rv * np.sin(av), rv * np.cos(av), s=3, c='steelblue', alpha=0.3)
    for lc in lid_cones:
        ax.plot(lc['x'], lc['y'], 'ro', ms=6)
        ax.annotate(f"{lc['angle_deg']:.0f}°", (lc['x'], lc['y']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.plot(0, 0, 'ko', ms=8)
    ax.arrow(0, 0, 0, 1, head_width=0.15, fc='green', ec='green')
    ax.set_xlim(-6, 6); ax.set_ylim(-1, 10)
    ax.set_xlabel("X (right+)"); ax.set_ylabel("Y (forward+)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"13_v4_{si:03d}.png"), dpi=120)
    plt.close()
    print(f"  13_v4_{si:03d}.png")

# Eval plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ax = axes[0]
ax.set_title(f"Pixel ({n_inliers} inliers)")
ax.scatter(exp_pixel[finl, 0], exp_pixel[finl, 1], s=12, c='green', alpha=0.5)
ax.scatter(exp_pixel[~finl, 0], exp_pixel[~finl, 1], s=8, c='red', alpha=0.3)
ax.set_xlim(0, 640); ax.set_ylim(480, 0)

ax = axes[1]
ax.set_title("Vehicle meters")
ax.set_aspect('equal')
ax.scatter(exp_meter[finl, 0], exp_meter[finl, 1], s=12, c='green', alpha=0.5)
proj_out = (H_out @ exp_h.T).T
proj_out_xy = proj_out[:, :2] / proj_out[:, 2:3]
ax.scatter(proj_out_xy[finl, 0], proj_out_xy[finl, 1], s=8, c='blue', alpha=0.3, marker='x')
ax.plot(0, 0, 'ko', ms=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.set_title("Error histogram")
err_out = np.linalg.norm(proj_out_xy - exp_meter, axis=1)
ax.hist(err_out[finl], bins=30, alpha=0.7, color='green')
if (~finl).sum() > 0:
    ax.hist(err_out[~finl], bins=30, alpha=0.4, color='red')
ax.set_xlabel("Error (m)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_v4_eval.png"), dpi=120)
plt.close()
print("  14_v4_eval.png")

# Grid
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("H grid: pixel → meters")
ax.set_aspect('equal')
for u in range(0, 641, 80):
    vs = np.arange(250, 481, 2, dtype=np.float64)
    pts = np.column_stack([np.full_like(vs, u), vs, np.ones_like(vs)])
    m = (H_out @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 12) & (xy[:, 1] > -1) & (xy[:, 1] < 15)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'b-', alpha=0.4, lw=1)
        idx = np.where(ok)[0]
        if len(idx) > 0:
            ax.annotate(f'u={u}', xy=(xy[idx[-1], 0], xy[idx[-1], 1]),
                        fontsize=7, color='blue', alpha=0.7)
for v in range(260, 481, 30):
    us = np.arange(0, 641, 2, dtype=np.float64)
    pts = np.column_stack([us, np.full_like(us, v), np.ones_like(us)])
    m = (H_out @ pts.T).T
    xy = m[:, :2] / m[:, 2:3]
    ok = (np.abs(xy[:, 0]) < 12) & (xy[:, 1] > -1) & (xy[:, 1] < 15)
    if np.any(ok):
        ax.plot(xy[ok, 0], xy[ok, 1], 'r-', alpha=0.3, lw=0.8)

ax.scatter(exp_meter[finl, 0], exp_meter[finl, 1], s=20, c='green', zorder=5,
           edgecolors='darkgreen', lw=0.5)
ax.plot(0, 0, 'ko', ms=10, zorder=5)
ax.arrow(0, 0, 0, 1.5, head_width=0.2, fc='black', ec='black', zorder=5)
ax.set_xlabel("X (right+, m)"); ax.set_ylabel("Y (forward+, m)")
ax.set_xlim(-8, 8); ax.set_ylim(-1, 12)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "15_v4_grid.png"), dpi=150)
plt.close()
print("  15_v4_grid.png")

# ============================================================
# Save
# ============================================================
save_dir = os.path.expanduser("~/xycar_ws/src/track_drive/config")
os.makedirs(save_dir, exist_ok=True)

np.savez(os.path.join(save_dir, "homography.npz"),
         H=H_out,
         pixel_pts_inlier=exp_pixel[finl],
         meter_pts_inlier=exp_meter[finl],
         reprojection_error_mean=float(inl_err.mean()),
         reprojection_error_median=float(np.median(inl_err)))

with open(os.path.join(save_dir, "homography.json"), 'w') as f:
    json.dump({
        'H': H_out.tolist(),
        'n_inliers': n_inliers,
        'n_total': int(len(exp_pixel)),
        'reproj_error_mean_m': float(inl_err.mean()),
        'reproj_error_median_m': float(np.median(inl_err)),
        'coordinate_system': 'X=right+(m), Y=forward+(m), lidar 0deg=front CW',
        'usage': 'pt_m = H @ [u, v, 1].T; X,Y = pt_m[:2] / pt_m[2]',
    }, f, indent=2)

print(f"\nSaved to {save_dir}")
print("Done!")
