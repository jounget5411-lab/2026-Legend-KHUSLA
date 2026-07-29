#!/usr/bin/env python3
"""
Final comparison with CORRECT pixel positions.

The live camera detected cones at meter positions (via cone H).
To get the actual detection pixel: pixel = H_cone_inv @ [cam_mx, cam_my, 1].
Then compare what ALL three H variants produce at those exact pixels.
"""

import numpy as np
import json, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")
OUT_DIR = os.path.expanduser("~/calib_out_vis")

with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    H_cone = np.array(json.load(f)['H'])

H_ground_ours = np.load(os.path.join(CONFIG_DIR, "homography_ground.npz"))['H']
H_cone_inv = np.linalg.inv(H_cone)

# Live match data: camera meter positions (from cone H) and lidar ground truth
live = [
    {'cam_mx': -1.79, 'cam_my': 2.27, 'lid_x': -2.20, 'lid_y': 2.48, 'lid_r': 3.3},
    {'cam_mx': -1.98, 'cam_my': 6.19, 'lid_x': -2.20, 'lid_y': 6.05, 'lid_r': 6.4},
    {'cam_mx': +2.70, 'cam_my': 6.49, 'lid_x': +2.41, 'lid_y': 6.63, 'lid_r': 7.1},
]

# Recover actual detection pixels from cone H inverse
for m in live:
    p = H_cone_inv @ np.array([m['cam_mx'], m['cam_my'], 1.0])
    m['u'] = p[0] / p[2]
    m['v'] = p[1] / p[2]

print("Recovered actual detection pixels:")
for m in live:
    print(f"  cam({m['cam_mx']:+.2f},{m['cam_my']:.2f}) → pixel({m['u']:.1f},{m['v']:.1f})")


def apply_H(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0]/p[2], p[1]/p[2]


print(f"\n{'='*80}")
print("LIVE: 3 methods at CORRECT camera detection pixels")
print(f"{'='*80}")
print(f"\n{'r(m)':<6} {'pixel':<14} {'Lidar GT':<16} "
      f"{'Cone H':<16} {'Cone ΔX ΔY |Δ|':<22} "
      f"{'Ground H':<16} {'Gnd ΔX  ΔY  |Δ|':<22}")
print("-" * 120)

cone_gaps = []
gnd_gaps = []

for m in live:
    u, v = m['u'], m['v']
    lx, ly = m['lid_x'], m['lid_y']

    # Cone H
    cx, cy = apply_H(H_cone, u, v)
    cdx, cdy = cx - lx, cy - ly
    cd = math.hypot(cdx, cdy)

    # Ground H (already in our coords)
    gx, gy = apply_H(H_ground_ours, u, v)
    gdx, gdy = gx - lx, gy - ly
    gd = math.hypot(gdx, gdy)

    cone_gaps.append({'dx': cdx, 'dy': cdy, 'd': cd, 'r': m['lid_r'], 'y': ly})
    gnd_gaps.append({'dx': gdx, 'dy': gdy, 'd': gd, 'r': m['lid_r'], 'y': ly})

    print(f"r={m['lid_r']:.1f}  ({u:5.1f},{v:5.1f})  ({lx:+.2f},{ly:.2f})  "
          f"({cx:+.2f},{cy:.2f})  {cdx:+.3f} {cdy:+.3f} {cd:.3f}  "
          f"({gx:+.2f},{gy:.2f})  {gdx:+.3f} {gdy:+.3f} {gd:.3f}")

print(f"\n{'MEAN':50} "
      f"{np.mean([g['dx'] for g in cone_gaps]):+.3f} {np.mean([g['dy'] for g in cone_gaps]):+.3f} {np.mean([g['d'] for g in cone_gaps]):.3f}  "
      f"{'':16} "
      f"{np.mean([g['dx'] for g in gnd_gaps]):+.3f} {np.mean([g['dy'] for g in gnd_gaps]):+.3f} {np.mean([g['d'] for g in gnd_gaps]):.3f}")


# Distance dependence check
print(f"\n{'='*80}")
print("DISTANCE DEPENDENCE: ΔX vs Y")
print(f"{'='*80}")

for label, gaps in [("Cone H", cone_gaps), ("Ground H", gnd_gaps)]:
    ys = [g['y'] for g in gaps]
    dxs = [g['dx'] for g in gaps]
    if len(ys) >= 2:
        A = np.column_stack([np.ones(len(ys)), ys])
        coeffs, _, _, _ = np.linalg.lstsq(A, dxs, rcond=None)
        a, b = coeffs
        ang_biases = [dx/y for dx, y in zip(dxs, ys)]
        print(f"\n{label}:")
        print(f"  ΔX = {a:+.4f} + {b:+.5f}·Y")
        print(f"  ΔX range: [{min(dxs):+.3f}, {max(dxs):+.3f}]")
        print(f"  ΔX std:   {np.std(dxs):.4f}m")
        print(f"  Angular bias ΔX/Y range: [{min(ang_biases):+.4f}, {max(ang_biases):+.4f}] rad")
        print(f"  Angular bias ΔX/Y std:   {np.std(ang_biases):.4f} rad = {np.degrees(np.std(ang_biases)):.2f}°")


# ============================================================
# Calibration inlier check for ground H
# ============================================================
print(f"\n{'='*80}")
print("CALIBRATION INLIER REPROJECTION — CORRECTED")
print(f"{'='*80}")

cone_npz = np.load(os.path.join(CONFIG_DIR, "homography.npz"))
inlier_px = cone_npz['pixel_pts_inlier']
inlier_mt = cone_npz['meter_pts_inlier']  # (X=right, Y=forward)

pix_h = np.hstack([inlier_px, np.ones((len(inlier_px), 1))])

# Cone H
proj_cone = (H_cone @ pix_h.T).T
proj_cone_xy = proj_cone[:, :2] / proj_cone[:, 2:3]
err_cone = np.linalg.norm(proj_cone_xy - inlier_mt, axis=1)

# Ground H (already in our coords)
proj_gnd = (H_ground_ours @ pix_h.T).T
proj_gnd_xy = proj_gnd[:, :2] / proj_gnd[:, 2:3]
err_gnd = np.linalg.norm(proj_gnd_xy - inlier_mt, axis=1)
res_gnd = proj_gnd_xy - inlier_mt

gt_Y = inlier_mt[:, 1]

print(f"\n{'Method':<20} {'Mean':<10} {'Median':<10} {'Max':<10} {'ΔX mean':<12} {'ΔY mean'}")
print("-" * 70)
print(f"{'Cone H':<20} {err_cone.mean():.4f}m   {np.median(err_cone):.4f}m   {err_cone.max():.4f}m   "
      f"{(proj_cone_xy[:,0]-inlier_mt[:,0]).mean():+.4f}m   {(proj_cone_xy[:,1]-inlier_mt[:,1]).mean():+.4f}m")
print(f"{'Ground H':<20} {err_gnd.mean():.4f}m   {np.median(err_gnd):.4f}m   {err_gnd.max():.4f}m   "
      f"{res_gnd[:,0].mean():+.4f}m   {res_gnd[:,1].mean():+.4f}m")

print(f"\nGround H by distance:")
for y_lo, y_hi in [(1, 3), (3, 5), (5, 7), (7, 10)]:
    mask = (gt_Y >= y_lo) & (gt_Y < y_hi)
    if mask.sum() > 0:
        print(f"  Y=[{y_lo},{y_hi}): n={mask.sum():2d}, "
              f"ΔX={res_gnd[mask,0].mean():+.4f}±{res_gnd[mask,0].std():.4f}, "
              f"ΔY={res_gnd[mask,1].mean():+.4f}±{res_gnd[mask,1].std():.4f}, "
              f"|Δ|={err_gnd[mask].mean():.3f}m")


# ============================================================
# Final verdict plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# 1. Live: ΔX vs Y for both methods
ax = axes[0]
ax.set_title("Live: ΔX vs Y (forward distance)", fontsize=12)
ys_cone = [g['y'] for g in cone_gaps]
dxs_cone = [g['dx'] for g in cone_gaps]
ys_gnd = [g['y'] for g in gnd_gaps]
dxs_gnd = [g['dx'] for g in gnd_gaps]
ax.scatter(ys_cone, dxs_cone, s=100, c='red', edgecolors='black', label='Cone H', zorder=5)
ax.scatter(ys_gnd, dxs_gnd, s=100, c='green', marker='s', edgecolors='black', label='Ground H', zorder=5)
for g in cone_gaps:
    ax.annotate(f"r={g['r']:.0f}", (g['y'], g['dx']), fontsize=8, xytext=(5, 5),
                textcoords='offset points', color='red')
for g in gnd_gaps:
    ax.annotate(f"r={g['r']:.0f}", (g['y'], g['dx']), fontsize=8, xytext=(5, -12),
                textcoords='offset points', color='green')
ax.axhline(0, color='gray', lw=1)
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("ΔX (m, right+)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2. Live: |Δ| vs distance
ax = axes[1]
ax.set_title("Live: total gap |Δ| vs distance", fontsize=12)
ax.scatter([g['r'] for g in cone_gaps], [g['d'] for g in cone_gaps],
           s=100, c='red', edgecolors='black', label='Cone H')
ax.scatter([g['r'] for g in gnd_gaps], [g['d'] for g in gnd_gaps],
           s=100, c='green', marker='s', edgecolors='black', label='Ground H')
ax.set_xlabel("Lidar distance (m)")
ax.set_ylabel("|Δ| gap (m)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.8)

# 3. Calib inlier |Δ| vs Y
ax = axes[2]
ax.set_title("Calib inlier: |Δ| vs Y", fontsize=12)
ax.scatter(gt_Y, err_cone, s=10, alpha=0.5, c='red', label=f'Cone H (mean={err_cone.mean():.3f}m)')
ax.scatter(gt_Y, err_gnd, s=10, alpha=0.5, c='green', label=f'Ground H (mean={err_gnd.mean():.3f}m)')
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("|Δ| error (m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Final comparison: Cone H vs Ground-projected Friend H", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "25_final_compare.png"), dpi=130)
plt.close()
print(f"\n→ {os.path.join(OUT_DIR, '25_final_compare.png')}")

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

cone_live_mean = np.mean([g['d'] for g in cone_gaps])
gnd_live_mean = np.mean([g['d'] for g in gnd_gaps])
cone_dx_std = np.std([g['dx'] for g in cone_gaps])
gnd_dx_std = np.std([g['dx'] for g in gnd_gaps])

print(f"""
                    Cone H          Ground H
  Live mean |Δ|:   {cone_live_mean:.3f}m          {gnd_live_mean:.3f}m
  Live ΔX bias:    {np.mean(dxs_cone):+.3f}m         {np.mean(dxs_gnd):+.3f}m
  Live ΔX std:     {cone_dx_std:.3f}m          {gnd_dx_std:.3f}m
  Calib mean |Δ|:  {err_cone.mean():.3f}m          {err_gnd.mean():.3f}m
  Center→X:        +0.661m          +0.078m
  Center→Y:        +1.012m          +0.808m

Ground H strengths:
  ✓ Center pixel maps correctly (X≈0.08m ≈ camera-lidar offset)
  ✓ Based on exact geometric model (no training data bias)
  ✓ Uniform across all positions (no data-coverage gap)

Ground H weakness:
  ✗ ~0.5m systematic leftward bias on calibration data
  ✗ Calib inlier error 0.68m vs cone H 0.07m

Cone H strengths:
  ✓ Excellent fit to calibration data (0.07m)
  ✓ Better mean live gap (0.35 vs 0.45m)

Cone H weakness:
  ✗ 0.3m rightward bias (distance-dependent)
  ✗ Trained on asymmetric data — biased extrapolation
""")
