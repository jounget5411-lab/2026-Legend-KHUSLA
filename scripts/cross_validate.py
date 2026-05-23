#!/usr/bin/env python3
"""
Cross-validate cone-based H vs friend's tf_static-based H.

Cone H: pixel → (X=right+, Y=forward+) meters
Friend H_pix2lidar: pixel → (X=forward+, Y=left+) meters (ROS lidar_frame)

Conversion: (friend_X, friend_Y) = (cone_Y, -cone_X)
         or: (cone_X, cone_Y) = (-friend_Y, friend_X)
"""

import numpy as np
import json
import os

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")

# ============================================================
# 1. Load cone-based H
# ============================================================
with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    cone_data = json.load(f)
H_cone = np.array(cone_data['H'])

# Load inlier data for reprojection test
cone_npz = np.load(os.path.join(CONFIG_DIR, "homography.npz"))
inlier_pixels = cone_npz['pixel_pts_inlier']
inlier_meters = cone_npz['meter_pts_inlier']  # in cone coords (X=right, Y=forward)

print("=" * 70)
print("CONE-BASED H (from calibration data)")
print("=" * 70)
print(f"H_cone:\n{H_cone}")
print(f"Inliers: {len(inlier_pixels)}, reproj error: {cone_data['reproj_error_mean_m']:.4f}m")

# ============================================================
# 2. Build friend's H_pix2lidar from captured K/R/t
# ============================================================
tf_npz = np.load(os.path.join(CONFIG_DIR, "tf_caminfo.npz"))
K = tf_npz['K']
R_raw = tf_npz['R']
t = tf_npz['t']

print(f"\n{'=' * 70}")
print("TF/CAMINFO DATA")
print("=" * 70)
print(f"K:\n{K}")
print(f"R_raw:\n{R_raw}")
print(f"t = [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}]")
print(f"t_z (cam height above lidar) = {t[2]:+.4f} m")

# Friend's R_ROBOT_TO_OPTICAL
R_ROBOT_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

d = t[2]  # camera height
T_homog = np.array([
    [-d,  0.0, t[0]],
    [0.0, -d,  t[1]],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
K_inv = np.linalg.inv(K)

# Test both conventions
u0 = K[0, 2]
v0 = K[1, 2] + K[1, 1] * 0.30
test_pix = np.array([u0, v0, 1.0])

print(f"\n--- Convention auto-detection (test pixel: u={u0:.0f}, v={v0:.1f}) ---")

candidates = [
    ("optical", R_raw),
    ("robot",   R_raw @ R_ROBOT_TO_OPTICAL),
]

best = None
for label, R_eff in candidates:
    H_pl = T_homog @ R_eff @ K_inv
    p = H_pl @ test_pix
    if p[2] == 0:
        print(f"  [{label}] singular — skipped")
        continue
    X_chk = p[0] / p[2]
    Y_chk = p[1] / p[2]
    score = X_chk - 2.0 * abs(Y_chk)
    ok = (0.5 < X_chk < 20) and (abs(Y_chk) < 2.0)
    print(f"  [{label}] → (X={X_chk:+.2f}, Y={Y_chk:+.2f})m  score={score:+.2f}  {'✓' if ok else '✗'}")
    if ok and (best is None or score > best[0]):
        best = (score, label, H_pl, R_eff)

if best is None:
    print("\nERROR: Neither convention produces valid BEV. Friend's method invalid.")
    exit(1)

_, chosen_label, H_friend, R_eff_chosen = best
print(f"\nChosen convention: [{chosen_label}]")
print(f"H_pix2lidar (friend):\n{H_friend}")

# ============================================================
# 3. Compare at representative pixels
# ============================================================
print(f"\n{'=' * 70}")
print("COMPARISON AT REPRESENTATIVE PIXELS")
print("=" * 70)

test_pixels = [
    (320, 480, "bottom-center"),
    (  0, 480, "bottom-left"),
    (640, 480, "bottom-right"),
    (320, 400, "mid-center"),
    (320, 300, "upper-center"),
]

print(f"\n{'Pixel':<18} {'Cone (X_r, Y_f)':<22} {'Friend→Cone (X_r, Y_f)':<24} {'Δ (m)':<10} {'Friend raw (X_f, Y_l)'}")
print("-" * 100)

for u, v, label in test_pixels:
    pix = np.array([u, v, 1.0])

    # Cone H: pixel → (X=right, Y=forward)
    c = H_cone @ pix
    cx, cy = c[0]/c[2], c[1]/c[2]

    # Friend H: pixel → (X=forward, Y=left) in lidar_frame
    f = H_friend @ pix
    fx, fy = f[0]/f[2], f[1]/f[2]

    # Convert friend to cone coords: cone_X = -friend_Y, cone_Y = friend_X
    fx_conv = -fy   # right = -(left)
    fy_conv = fx     # forward = forward

    # Difference
    delta = np.sqrt((cx - fx_conv)**2 + (cy - fy_conv)**2)

    print(f"  ({u:3d},{v:3d}) {label:<10}  ({cx:+6.2f}, {cy:+6.2f})     ({fx_conv:+6.2f}, {fy_conv:+6.2f})       {delta:6.3f}    ({fx:+6.2f}, {fy:+6.2f})")

# ============================================================
# 4. Re-project cone inlier data through friend's H
# ============================================================
print(f"\n{'=' * 70}")
print("REPROJECTION OF CONE INLIER POINTS THROUGH FRIEND'S H")
print("=" * 70)

# Cone inliers: pixel → meter (cone coords X=right, Y=forward)
# Convert ground truth to friend coords: friend_X = cone_Y, friend_Y = -cone_X
gt_friend_X = inlier_meters[:, 1]   # forward
gt_friend_Y = -inlier_meters[:, 0]  # left

# Project through friend's H
pix_h = np.hstack([inlier_pixels, np.ones((len(inlier_pixels), 1))])
proj_friend = (H_friend @ pix_h.T).T
proj_friend_xy = proj_friend[:, :2] / proj_friend[:, 2:3]

# Errors in friend coords
errors_friend = np.sqrt(
    (proj_friend_xy[:, 0] - gt_friend_X)**2 +
    (proj_friend_xy[:, 1] - gt_friend_Y)**2
)

# Also compute cone H errors for same points (for comparison)
proj_cone = (H_cone @ pix_h.T).T
proj_cone_xy = proj_cone[:, :2] / proj_cone[:, 2:3]
errors_cone = np.linalg.norm(proj_cone_xy - inlier_meters, axis=1)

print(f"\nCone inlier points: {len(inlier_pixels)}")
print(f"                     Cone H          Friend H")
print(f"  Mean error:       {errors_cone.mean():8.4f} m     {errors_friend.mean():8.4f} m")
print(f"  Median error:     {np.median(errors_cone):8.4f} m     {np.median(errors_friend):8.4f} m")
print(f"  Max error:        {errors_cone.max():8.4f} m     {errors_friend.max():8.4f} m")
print(f"  Std error:        {errors_cone.std():8.4f} m     {errors_friend.std():8.4f} m")

# Per-point comparison for first/last few
print(f"\nSample points:")
print(f"{'Pixel':<16} {'GT(cone)':<20} {'Cone H proj':<20} {'Friend proj→cone':<20} {'ΔCone':<8} {'ΔFriend'}")
print("-" * 105)
sample_idx = list(range(min(5, len(inlier_pixels)))) + list(range(max(0, len(inlier_pixels)-5), len(inlier_pixels)))
sample_idx = sorted(set(sample_idx))
for i in sample_idx:
    u, v = inlier_pixels[i]
    gx, gy = inlier_meters[i]
    cx, cy = proj_cone_xy[i]
    # Friend projection in cone coords
    fx_raw, fy_raw = proj_friend_xy[i]
    fx_conv = -fy_raw  # right
    fy_conv = fx_raw   # forward
    dc = errors_cone[i]
    df = errors_friend[i]
    print(f"  ({u:5.0f},{v:5.0f})    ({gx:+5.2f},{gy:+5.2f})      ({cx:+5.2f},{cy:+5.2f})      ({fx_conv:+5.2f},{fy_conv:+5.2f})      {dc:6.3f}   {df:6.3f}")

# ============================================================
# 5. Judgment
# ============================================================
print(f"\n{'=' * 70}")
print("JUDGMENT")
print("=" * 70)

# Compute average difference at representative pixels
diffs = []
for u, v, _ in test_pixels:
    pix = np.array([u, v, 1.0])
    c = H_cone @ pix; cx, cy = c[0]/c[2], c[1]/c[2]
    f = H_friend @ pix; fx, fy = f[0]/f[2], f[1]/f[2]
    fx_conv, fy_conv = -fy, fx
    diffs.append(np.sqrt((cx-fx_conv)**2 + (cy-fy_conv)**2))

mean_diff = np.mean(diffs)
max_diff = np.max(diffs)

print(f"\nRepresentative pixel differences:")
print(f"  Mean: {mean_diff:.3f} m")
print(f"  Max:  {max_diff:.3f} m")

if mean_diff < 0.5:
    print(f"\n→ PASS: 두 H가 평균 {mean_diff:.2f}m 차이로 잘 일치.")
    print(f"  라바콘 H를 확신할 수 있음. 통합 단계로 진행.")
elif mean_diff < 1.5:
    print(f"\n→ WARNING: {mean_diff:.2f}m 차이. 중간 수준.")
    if errors_cone.mean() < errors_friend.mean():
        print(f"  라바콘 H (reproj {errors_cone.mean():.3f}m) > 친구 H ({errors_friend.mean():.3f}m)")
    else:
        print(f"  친구 H (reproj {errors_friend.mean():.3f}m) > 라바콘 H ({errors_cone.mean():.3f}m)")
else:
    print(f"\n→ FAIL: {mean_diff:.2f}m 차이. 크게 불일치.")

# Check center pixel maps forward
c = H_cone @ np.array([320, 400, 1.0])
center_forward_cone = c[1]/c[2]
f = H_friend @ np.array([320, 400, 1.0])
center_forward_friend = f[0]/f[2]
print(f"\n중앙 픽셀 (320,400) → 전방:")
print(f"  라바콘 H: Y_forward = {center_forward_cone:+.2f} m")
print(f"  친구   H: X_forward = {center_forward_friend:+.2f} m")

if center_forward_cone > 0 and center_forward_friend > 0:
    print(f"  둘 다 전방 양수 ✓")
else:
    print(f"  ※ 전방 음수가 있으면 해당 H의 좌표계가 의심됨")

# Save friend's H to config for future reference
with open(os.path.join(CONFIG_DIR, "homography_friend.json"), 'w') as fj:
    json.dump({
        'H_pix2lidar': H_friend.tolist(),
        'convention': chosen_label,
        'coordinate_system': 'X=forward+(m), Y=left+(m) (ROS lidar_frame)',
        'camera_height_m': float(d),
        'K': K.tolist(),
    }, fj, indent=2)
print(f"\n친구 H 저장: {CONFIG_DIR}/homography_friend.json")
print("Done!")
