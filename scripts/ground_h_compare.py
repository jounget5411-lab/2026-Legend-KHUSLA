#!/usr/bin/env python3
"""
Compare: Cone H vs Friend H (lidar plane z=0) vs Friend H (ground plane z=-0.228).

The friend's original H projects to z=0 in lidar frame (lidar scanning plane).
The ground-corrected version projects to z=-h_lidar (actual ground).
Since lidar scans horizontally at its height, vertical objects have the same XY
at both planes, so the ground projection should match lidar measurements better
for points the camera sees at ground level (cone bottoms).
"""

import numpy as np
import json
import os
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")
OUT_DIR = os.path.expanduser("~/calib_out_vis")

# Load data
with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    H_cone = np.array(json.load(f)['H'])

tf_npz = np.load(os.path.join(CONFIG_DIR, "tf_caminfo.npz"))
K = tf_npz['K']
R_raw = tf_npz['R']
t = tf_npz['t']

cone_npz = np.load(os.path.join(CONFIG_DIR, "homography.npz"))
inlier_px = cone_npz['pixel_pts_inlier']
inlier_mt = cone_npz['meter_pts_inlier']

# ============================================================
# Build friend H variants
# ============================================================
R_ROBOT_TO_OPTICAL = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)
R_eff = R_raw @ R_ROBOT_TO_OPTICAL  # "robot" convention was chosen
K_inv = np.linalg.inv(K)

LIDAR_HEIGHT = 0.228  # lidar height above ground (meters)

# Original: project to lidar plane (z=0 in lidar frame)
d_lidar = t[2]  # = 1.51m
T_lidar = np.array([[-d_lidar, 0, t[0]], [0, -d_lidar, t[1]], [0, 0, 1]], dtype=np.float64)
H_friend_lidar = T_lidar @ R_eff @ K_inv

# Ground-corrected: project to ground plane (z = -LIDAR_HEIGHT in lidar frame)
d_ground = t[2] + LIDAR_HEIGHT  # = 1.738m (camera height above ground)
T_ground = np.array([[-d_ground, 0, t[0]], [0, -d_ground, t[1]], [0, 0, 1]], dtype=np.float64)
H_friend_ground = T_ground @ R_eff @ K_inv

print(f"t = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]")
print(f"d_lidar = {d_lidar:.4f}m (cam above lidar)")
print(f"d_ground = {d_ground:.4f}m (cam above ground)")
print(f"LIDAR_HEIGHT = {LIDAR_HEIGHT:.4f}m")

print(f"\nH_friend_lidar:\n{H_friend_lidar}")
print(f"\nH_friend_ground:\n{H_friend_ground}")

# ============================================================
# Helper: friend H output (lidar frame X=fwd,Y=left) → our frame (X=right,Y=fwd)
# ============================================================
def friend_to_ours(fx, fy):
    return -fy, fx  # X_ours = -Y_friend, Y_ours = X_friend

def apply_H(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0]/p[2], p[1]/p[2]


# ============================================================
# Live match data
# ============================================================
live_matches = [
    {'lid_x': -2.20, 'lid_y': 2.48, 'pixel_u': 92, 'pixel_v': 379, 'lid_r': 3.3},
    {'lid_x': -2.20, 'lid_y': 6.05, 'pixel_u': 196, 'pixel_v': 301, 'lid_r': 6.4},
    {'lid_x': +2.41, 'lid_y': 6.63, 'pixel_u': 432, 'pixel_v': 307, 'lid_r': 7.1},
]

# Approximate pixel positions from H_inv of cone H
H_cone_inv = np.linalg.inv(H_cone)
for m in live_matches:
    p = H_cone_inv @ np.array([m['lid_x'], m['lid_y'], 1.0])
    m['pixel_u_expected'] = p[0]/p[2]
    m['pixel_v_expected'] = p[1]/p[2]

print(f"\n{'='*80}")
print("LIVE CONE COMPARISON: 3 methods × 3 cones")
print(f"{'='*80}")
print(f"\n{'Cone':<14} {'Lidar GT':<18} {'ConeH→ours':<18} {'Friend(lidar)':<18} {'Friend(ground)':<18}")
print(f"{'':14} {'(X_r, Y_f)':<18} {'ΔX    ΔY   |Δ|':<18} {'ΔX    ΔY   |Δ|':<18} {'ΔX    ΔY   |Δ|':<18}")
print("-" * 88)

all_cone_dx, all_cone_dy = [], []
all_flidar_dx, all_flidar_dy = [], []
all_fground_dx, all_fground_dy = [], []

for m in live_matches:
    u, v = m['pixel_u'], m['pixel_v']
    lx, ly = m['lid_x'], m['lid_y']

    # Cone H (already in our coords)
    cx, cy = apply_H(H_cone, u, v)
    cdx, cdy = cx - lx, cy - ly
    cd = math.hypot(cdx, cdy)

    # Friend lidar-plane (output in lidar frame, convert to ours)
    fx_l, fy_l = apply_H(H_friend_lidar, u, v)
    fx_l_ours, fy_l_ours = friend_to_ours(fx_l, fy_l)
    fldx, fldy = fx_l_ours - lx, fy_l_ours - ly
    fld = math.hypot(fldx, fldy)

    # Friend ground-plane (output in lidar frame, convert to ours)
    fx_g, fy_g = apply_H(H_friend_ground, u, v)
    fx_g_ours, fy_g_ours = friend_to_ours(fx_g, fy_g)
    fgdx, fgdy = fx_g_ours - lx, fy_g_ours - ly
    fgd = math.hypot(fgdx, fgdy)

    all_cone_dx.append(cdx); all_cone_dy.append(cdy)
    all_flidar_dx.append(fldx); all_flidar_dy.append(fldy)
    all_fground_dx.append(fgdx); all_fground_dy.append(fgdy)

    print(f"r={m['lid_r']:.1f}m px({u},{v}) ({lx:+.2f},{ly:.2f})  "
          f"{cdx:+.2f} {cdy:+.2f} {cd:.2f}   "
          f"{fldx:+.2f} {fldy:+.2f} {fld:.2f}   "
          f"{fgdx:+.2f} {fgdy:+.2f} {fgd:.2f}")

print(f"\n{'Mean':<30}  "
      f"{np.mean(all_cone_dx):+.2f} {np.mean(all_cone_dy):+.2f} {np.mean([math.hypot(dx,dy) for dx,dy in zip(all_cone_dx, all_cone_dy)]):.2f}   "
      f"{np.mean(all_flidar_dx):+.2f} {np.mean(all_flidar_dy):+.2f} {np.mean([math.hypot(dx,dy) for dx,dy in zip(all_flidar_dx, all_flidar_dy)]):.2f}   "
      f"{np.mean(all_fground_dx):+.2f} {np.mean(all_fground_dy):+.2f} {np.mean([math.hypot(dx,dy) for dx,dy in zip(all_fground_dx, all_fground_dy)]):.2f}")


# ============================================================
# Calibration inlier reprojection (all 53 points)
# ============================================================
print(f"\n{'='*80}")
print("CALIBRATION INLIER REPROJECTION (53 points)")
print(f"{'='*80}")

# Convert inlier GT from our coords (X=right,Y=fwd) to friend coords (X=fwd,Y=left)
# friend_X = our_Y, friend_Y = -our_X
gt_friend_X = inlier_mt[:, 1]
gt_friend_Y = -inlier_mt[:, 0]

pix_h = np.hstack([inlier_px, np.ones((len(inlier_px), 1))])

# Cone H
proj_cone = (H_cone @ pix_h.T).T
proj_cone_xy = proj_cone[:, :2] / proj_cone[:, 2:3]
err_cone = np.linalg.norm(proj_cone_xy - inlier_mt, axis=1)

# Friend lidar
proj_fl = (H_friend_lidar @ pix_h.T).T
proj_fl_xy = proj_fl[:, :2] / proj_fl[:, 2:3]
err_fl = np.sqrt((proj_fl_xy[:,0] - gt_friend_X)**2 + (proj_fl_xy[:,1] - gt_friend_Y)**2)

# Friend ground
proj_fg = (H_friend_ground @ pix_h.T).T
proj_fg_xy = proj_fg[:, :2] / proj_fg[:, 2:3]
err_fg = np.sqrt((proj_fg_xy[:,0] - gt_friend_X)**2 + (proj_fg_xy[:,1] - gt_friend_Y)**2)

# Convert friend projections to our coords for residual analysis
proj_fl_ours = np.column_stack([-proj_fl_xy[:,1], proj_fl_xy[:,0]])
proj_fg_ours = np.column_stack([-proj_fg_xy[:,1], proj_fg_xy[:,0]])
res_fl = proj_fl_ours - inlier_mt
res_fg = proj_fg_ours - inlier_mt

print(f"\n{'Method':<25} {'Mean err':<10} {'Median':<10} {'Max':<10} {'ΔX mean':<12} {'ΔY mean':<12}")
print("-" * 80)
print(f"{'Cone H':<25} {err_cone.mean():.4f}m   {np.median(err_cone):.4f}m   {err_cone.max():.4f}m   "
      f"{(proj_cone_xy[:,0]-inlier_mt[:,0]).mean():+.4f}m   {(proj_cone_xy[:,1]-inlier_mt[:,1]).mean():+.4f}m")
print(f"{'Friend (lidar z=0)':<25} {err_fl.mean():.4f}m   {np.median(err_fl):.4f}m   {err_fl.max():.4f}m   "
      f"{res_fl[:,0].mean():+.4f}m   {res_fl[:,1].mean():+.4f}m")
print(f"{'Friend (ground z=-0.23)':<25} {err_fg.mean():.4f}m   {np.median(err_fg):.4f}m   {err_fg.max():.4f}m   "
      f"{res_fg[:,0].mean():+.4f}m   {res_fg[:,1].mean():+.4f}m")

# Distance-band analysis for ground-corrected friend H
print(f"\nFriend (ground) residuals by distance band:")
gt_Y = inlier_mt[:, 1]
for y_lo, y_hi in [(1, 3), (3, 5), (5, 7), (7, 10)]:
    mask = (gt_Y >= y_lo) & (gt_Y < y_hi)
    if mask.sum() > 0:
        print(f"  Y=[{y_lo},{y_hi})m: n={mask.sum():3d}, "
              f"ΔX={res_fg[mask,0].mean():+.4f}±{res_fg[mask,0].std():.4f}, "
              f"ΔY={res_fg[mask,1].mean():+.4f}±{res_fg[mask,1].std():.4f}, "
              f"|Δ|={err_fg[mask].mean():.4f}m")


# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# Row 1: Live cone comparison
labels = ['Cone H', 'Friend (lidar)', 'Friend (ground)']
dxs_all = [all_cone_dx, all_flidar_dx, all_fground_dx]
dys_all = [all_cone_dy, all_flidar_dy, all_fground_dy]
colors = ['red', 'blue', 'green']
rs = [m['lid_r'] for m in live_matches]

for i, (label, dxs, dys, col) in enumerate(zip(labels, dxs_all, dys_all, colors)):
    ax = axes[0, i]
    ax.set_title(f"Live gaps: {label}", fontsize=11)
    ax.set_aspect('equal')
    sc = ax.scatter(dxs, dys, c=rs, cmap='viridis', s=100, edgecolors='black', zorder=5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.plot(np.mean(dxs), np.mean(dys), '+', color=col, ms=20, mew=3)
    mean_d = np.mean([math.hypot(dx, dy) for dx, dy in zip(dxs, dys)])
    ax.set_xlabel("ΔX (right+, m)")
    ax.set_ylabel("ΔY (forward+, m)")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.text(0.05, 0.95, f"mean |Δ|={mean_d:.3f}m\nΔX={np.mean(dxs):+.3f}\nΔY={np.mean(dys):+.3f}",
            transform=ax.transAxes, fontsize=9, va='top', family='monospace',
            bbox=dict(fc='white', alpha=0.8))
    ax.grid(True, alpha=0.3)

plt.colorbar(sc, ax=axes[0, 2], label='Lidar distance (m)', shrink=0.8)

# Row 2: Calibration inlier errors
# 2a: Error vs distance for all 3 methods
ax = axes[1, 0]
ax.set_title("Calib inlier: |Δ| vs Y (forward)")
ax.scatter(gt_Y, err_cone, s=12, alpha=0.5, c='red', label='Cone H')
ax.scatter(gt_Y, err_fg, s=12, alpha=0.5, c='green', label='Friend (ground)')
ax.scatter(gt_Y, err_fl, s=12, alpha=0.5, c='blue', label='Friend (lidar)')
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("|Δ| error (m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2b: Friend ground ΔX vs Y (distance dependence check)
ax = axes[1, 1]
ax.set_title("Friend (ground): ΔX vs Y")
ax.scatter(gt_Y, res_fg[:, 0], s=12, c='green', alpha=0.5)
ax.axhline(0, color='gray', lw=0.5)
ax.axhline(res_fg[:, 0].mean(), color='green', ls=':', label=f'mean={res_fg[:,0].mean():+.3f}')
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("ΔX residual (m)")
ax.legend()
ax.grid(True, alpha=0.3)

# 2c: Summary bar chart
ax = axes[1, 2]
ax.set_title("Summary: mean |Δ| comparison")
methods = ['Cone H\n(calib)', 'Cone H\n(live)', 'Friend\nlidar(calib)',
           'Friend\nground(calib)', 'Friend\nground(live)']
live_cone_mean = np.mean([math.hypot(dx,dy) for dx,dy in zip(all_cone_dx, all_cone_dy)])
live_fg_mean = np.mean([math.hypot(dx,dy) for dx,dy in zip(all_fground_dx, all_fground_dy)])
vals = [err_cone.mean(), live_cone_mean, err_fl.mean(), err_fg.mean(), live_fg_mean]
cols_bar = ['red', 'salmon', 'royalblue', 'green', 'lightgreen']
bars = ax.bar(methods, vals, color=cols_bar, edgecolor='black', lw=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}m',
            ha='center', fontsize=9)
ax.set_ylabel("Mean |Δ| (m)")
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("Cone H vs Friend H (lidar plane) vs Friend H (ground plane)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "24_ground_h_compare.png"), dpi=130)
plt.close()
print(f"\n→ {os.path.join(OUT_DIR, '24_ground_h_compare.png')}")

# ============================================================
# Save ground-corrected friend H
# ============================================================
# Convert to our coordinate frame (X=right, Y=forward)
# H_friend_ground outputs (X_fwd, Y_left) → (X_right, Y_fwd) = (-Y_left, X_fwd)
# Apply coord swap matrix: [[0,-1,0],[1,0,0],[0,0,1]] @ H_friend_ground
swap = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
H_ground_ours = swap @ H_friend_ground

print(f"\nH_friend_ground in our coords (X=right, Y=forward):")
print(H_ground_ours)

# Verify
for u, v in [(320, 480), (320, 400), (320, 300)]:
    p = H_ground_ours @ np.array([u, v, 1.0])
    x, y = p[0]/p[2], p[1]/p[2]
    print(f"  ({u},{v}) → ({x:+.3f}, {y:+.3f})m")

np.savez(os.path.join(CONFIG_DIR, "homography_ground.npz"), H=H_ground_ours)
with open(os.path.join(CONFIG_DIR, "homography_ground.json"), 'w') as f:
    json.dump({
        'H': H_ground_ours.tolist(),
        'method': 'tf_static + ground plane projection (z=-0.228 in lidar frame)',
        'coordinate_system': 'X=right+(m), Y=forward+(m)',
        'camera_height_above_ground_m': float(d_ground),
        'lidar_height_m': float(LIDAR_HEIGHT),
    }, f, indent=2)
print(f"Saved: {CONFIG_DIR}/homography_ground.{{npz,json}}")
print("Done!")
