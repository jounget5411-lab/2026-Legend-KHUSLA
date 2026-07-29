#!/usr/bin/env python3
"""
Bias analysis: Is ΔX constant or distance-dependent?

1. Calibration inliers: check H residual pattern vs distance
2. Live matches: ΔX vs Y (forward distance)
3. The 0.66m vs 0.30m discrepancy explained
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG_DIR = os.path.expanduser("~/xycar_ws/src/track_drive/config")
OUT_DIR = os.path.expanduser("~/calib_out_vis")

# Load H and inlier data
with open(os.path.join(CONFIG_DIR, "homography.json")) as f:
    H = np.array(json.load(f)['H'])

npz = np.load(os.path.join(CONFIG_DIR, "homography.npz"))
inlier_px = npz['pixel_pts_inlier']   # (N, 2) pixel coords
inlier_mt = npz['meter_pts_inlier']   # (N, 2) meter coords (X=right, Y=forward)

# Live match data (from the v2 run, hardcoded since car was stationary)
live_matches = [
    {'lid_x': -2.20, 'lid_y': 2.48, 'cam_x': -1.79, 'cam_y': 2.27, 'lid_r': 3.3},
    {'lid_x': -2.20, 'lid_y': 6.05, 'cam_x': -1.98, 'cam_y': 6.19, 'lid_r': 6.4},
    {'lid_x': +2.41, 'lid_y': 6.63, 'cam_x': +2.70, 'cam_y': 6.49, 'lid_r': 7.1},
]

print("=" * 70)
print("PART 1: Calibration inlier residual analysis")
print("=" * 70)

# Project inlier pixels through H
pix_h = np.hstack([inlier_px, np.ones((len(inlier_px), 1))])
proj = (H @ pix_h.T).T
proj_xy = proj[:, :2] / proj[:, 2:3]

# Residuals: H_projected - ground_truth
res_x = proj_xy[:, 0] - inlier_mt[:, 0]  # ΔX
res_y = proj_xy[:, 1] - inlier_mt[:, 1]  # ΔY
gt_Y = inlier_mt[:, 1]   # forward distance
gt_X = inlier_mt[:, 0]   # lateral position
gt_r = np.sqrt(gt_X**2 + gt_Y**2)

print(f"Inlier points: {len(inlier_px)}")
print(f"Residual ΔX: mean={res_x.mean():+.4f}m  std={res_x.std():.4f}m")
print(f"Residual ΔY: mean={res_y.mean():+.4f}m  std={res_y.std():.4f}m")
print(f"\nResidual by distance band:")
for y_lo, y_hi in [(1, 3), (3, 5), (5, 7), (7, 10)]:
    mask = (gt_Y >= y_lo) & (gt_Y < y_hi)
    if mask.sum() > 0:
        print(f"  Y=[{y_lo},{y_hi})m: n={mask.sum():3d}, "
              f"ΔX={res_x[mask].mean():+.4f}±{res_x[mask].std():.4f}, "
              f"ΔY={res_y[mask].mean():+.4f}±{res_y[mask].std():.4f}")

print(f"\nResidual by lateral side:")
left = gt_X < 0
right = gt_X > 0
print(f"  Left  (X<0): n={left.sum()}, ΔX={res_x[left].mean():+.4f}, ΔY={res_y[left].mean():+.4f}")
print(f"  Right (X>0): n={right.sum()}, ΔX={res_x[right].mean():+.4f}, ΔY={res_y[right].mean():+.4f}")


# ============================================================
print(f"\n{'='*70}")
print("PART 2: H center-pixel offset at different v values")
print("=" * 70)
print("What X offset does H produce at u=320 (center) for different v (depth)?")
print()
for v in [480, 450, 420, 400, 380, 360, 340, 320, 300, 280]:
    p = H @ np.array([320, v, 1.0])
    x, y = p[0]/p[2], p[1]/p[2]
    if 0 < y < 30:
        print(f"  v={v:3d} → X={x:+.3f}m, Y={y:+.3f}m   (X offset at center)")


# ============================================================
print(f"\n{'='*70}")
print("PART 3: Why 0.66m vs 0.30m?")
print("=" * 70)
print("""
The 0.66m is the X output for the exact pixel (320,480) through H.
The 0.30m is the average ΔX between camera-H and lidar for 3 matched cones.

These are DIFFERENT measurements:
- 0.66m = absolute X of where the center-bottom pixel maps to
  (ideally should be ~0 if camera is centered)
- 0.30m = camera-projected X minus lidar-measured X for real cones
  (measures alignment between camera and lidar)

Key: the cones are NOT at pixel (320,480) — they're off-center.
The 0.66m at (320,480) is the maximum bias (at the image center).
At the actual cone pixel positions, the bias is smaller (0.23-0.40m)
because the cones are off to the sides.
""")

# Check: what's the H X-offset at actual cone pixel positions?
print("H X-offset at calibration inlier pixel positions:")
print(f"  u range: [{inlier_px[:,0].min():.0f}, {inlier_px[:,0].max():.0f}]")
print(f"  v range: [{inlier_px[:,1].min():.0f}, {inlier_px[:,1].max():.0f}]")

# For the 3 live cones, what pixel do they correspond to?
# Use H_inv to check
H_inv = np.linalg.inv(H)
for m in live_matches:
    # Where does the lidar position map to in pixel space?
    p = H_inv @ np.array([m['lid_x'], m['lid_y'], 1.0])
    u_expected = p[0]/p[2]
    v_expected = p[1]/p[2]
    print(f"\n  Lidar ({m['lid_x']:+.2f},{m['lid_y']:.2f}): "
          f"expected pixel ≈ ({u_expected:.0f},{v_expected:.0f})")
    # The actual camera detection pixel
    # approximate from camera-projected coords
    p2 = H_inv @ np.array([m['cam_x'], m['cam_y'], 1.0])
    u_actual = p2[0]/p2[2]
    v_actual = p2[1]/p2[2]
    print(f"  Camera det:  actual pixel ≈ ({u_actual:.0f},{v_actual:.0f})")
    print(f"  Pixel difference: Δu={u_actual-u_expected:.0f}")


# ============================================================
print(f"\n{'='*70}")
print("PART 4: ΔX vs Y — linear regression")
print("=" * 70)

# Live data
live_Y = [m['lid_y'] for m in live_matches]
live_dX = [m['cam_x'] - m['lid_x'] for m in live_matches]
live_dY_fwd = [m['cam_y'] - m['lid_y'] for m in live_matches]

# Fit ΔX = a + b*Y
A = np.column_stack([np.ones(len(live_Y)), live_Y])
coeffs_x, res, _, _ = np.linalg.lstsq(A, live_dX, rcond=None)
a_x, b_x = coeffs_x
print(f"\nLive data (3 points):")
print(f"  ΔX = {a_x:+.4f} + {b_x:+.4f} * Y")
print(f"  → At Y=0: ΔX = {a_x:+.3f}m")
print(f"  → At Y=5: ΔX = {a_x + 5*b_x:+.3f}m")
print(f"  → At Y=10: ΔX = {a_x + 10*b_x:+.3f}m")

if abs(b_x) < 0.01:
    print(f"  ▶ Slope ≈ 0: bias is CONSTANT (parallel shift)")
elif abs(b_x) > 0.02:
    print(f"  ▶ Slope significant: bias is DISTANCE-DEPENDENT")

# Also check: is the bias in ANGLE rather than distance?
# Angular bias = ΔX / Y → should be constant if it's a rotation
live_ang = [dx/y for dx, y in zip(live_dX, live_Y)]
print(f"\n  Angular bias ΔX/Y:")
for m, ang in zip(live_matches, live_ang):
    print(f"    Y={m['lid_y']:.2f}m → ΔX/Y = {ang:+.4f} rad = {np.degrees(ang):+.2f}°")
print(f"  Mean angular bias: {np.mean(live_ang):+.4f} rad = {np.degrees(np.mean(live_ang)):+.2f}°")
ang_std = np.std(live_ang)
print(f"  Std: {ang_std:.4f} rad = {np.degrees(ang_std):.2f}°")

if ang_std < 0.01:
    print(f"  ▶ Angular bias is CONSTANT → rotation error (≈{np.degrees(np.mean(live_ang)):+.1f}°)")
else:
    print(f"  ▶ Angular bias varies → mixed error")


# ============================================================
print(f"\n{'='*70}")
print("PART 5: Root cause diagnosis")
print("=" * 70)


# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Calibration residuals vs Y
ax = axes[0, 0]
ax.set_title("Calib inlier: ΔX vs Y (forward)")
ax.scatter(gt_Y, res_x, s=15, c='steelblue', alpha=0.6)
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("ΔX residual (m)")
ax.grid(True, alpha=0.3)

# 2. Calibration residuals vs X
ax = axes[0, 1]
ax.set_title("Calib inlier: ΔX vs X (lateral)")
ax.scatter(gt_X, res_x, s=15, c='steelblue', alpha=0.6)
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel("X lateral (m)")
ax.set_ylabel("ΔX residual (m)")
ax.grid(True, alpha=0.3)

# 3. Calibration residuals 2D
ax = axes[0, 2]
ax.set_title("Calib inlier: residual vectors")
ax.set_aspect('equal')
ax.scatter(inlier_mt[:, 0], inlier_mt[:, 1], s=10, c='gray', alpha=0.3, label='GT')
for i in range(len(inlier_mt)):
    ax.annotate('', xy=(proj_xy[i, 0], proj_xy[i, 1]),
                xytext=(inlier_mt[i, 0], inlier_mt[i, 1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.5))
ax.plot(0, 0, 'ko', ms=8)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.grid(True, alpha=0.3)

# 4. Live: ΔX vs Y with linear fit
ax = axes[1, 0]
ax.set_title("Live: ΔX vs Y (3 matched cones)")
ax.scatter(live_Y, live_dX, s=80, c='red', edgecolors='black', zorder=5)
for m, dx in zip(live_matches, live_dX):
    ax.annotate(f"r={m['lid_r']:.1f}m", (m['lid_y'], dx),
                fontsize=8, xytext=(5, 5), textcoords='offset points')
y_fit = np.linspace(0, 10, 50)
ax.plot(y_fit, a_x + b_x * y_fit, 'b--', alpha=0.5, label=f'ΔX={a_x:.3f}+{b_x:.4f}·Y')
ax.axhline(0, color='gray', lw=0.5)
ax.axhline(np.mean(live_dX), color='red', ls=':', label=f'mean={np.mean(live_dX):.3f}')
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("ΔX (camera-lidar, m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Angular bias ΔX/Y vs Y
ax = axes[1, 1]
ax.set_title("Live: angular bias ΔX/Y vs Y")
ax.scatter(live_Y, [np.degrees(a) for a in live_ang], s=80, c='orange', edgecolors='black')
ax.axhline(np.degrees(np.mean(live_ang)), color='orange', ls=':',
           label=f'mean={np.degrees(np.mean(live_ang)):+.2f}°')
ax.set_xlabel("Y forward (m)")
ax.set_ylabel("ΔX/Y (degrees)")
ax.legend()
ax.grid(True, alpha=0.3)

# 6. H center offset vs v
ax = axes[1, 2]
ax.set_title("H center-pixel X offset vs v (=depth)")
vs = np.arange(480, 250, -2)
xs = []
ys_fwd = []
for v in vs:
    p = H @ np.array([320, v, 1.0])
    x, y = p[0]/p[2], p[1]/p[2]
    if 0 < y < 20:
        xs.append(x)
        ys_fwd.append(y)
    else:
        xs.append(np.nan)
        ys_fwd.append(np.nan)
ax.plot(vs, xs, 'b-', lw=2)
ax.set_xlabel("v (pixel row, 480=bottom)")
ax.set_ylabel("X offset at u=320 (m)")
ax.axhline(0, color='gray', lw=0.5)
ax.grid(True, alpha=0.3)
# Add secondary Y axis showing forward distance
ax2 = ax.twinx()
ax2.plot(vs, ys_fwd, 'g--', alpha=0.5, label='Y forward (m)')
ax2.set_ylabel("Y forward (m)", color='green')
ax2.legend(loc='upper right')

plt.suptitle("Bias Analysis: Is ΔX constant or distance-dependent?", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "23_bias_analysis.png"), dpi=130)
plt.close()
print(f"\n→ {os.path.join(OUT_DIR, '23_bias_analysis.png')}")
print("Done!")
