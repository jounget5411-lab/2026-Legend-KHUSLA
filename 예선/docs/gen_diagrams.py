#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_state(ax, x, y, w, h, label, sub="", color="#4ECDC4"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                          facecolor=color, edgecolor="#2C3E50", linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2+0.15, label, ha="center", va="center", fontsize=10, fontweight="bold")
    if sub:
        ax.text(x+w/2, y+h/2-0.25, sub, ha="center", va="center", fontsize=7, style="italic")

def draw_trans(ax, x1, y1, x2, y2, label, color="#E74C3C"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.15, mid_y+0.2, label, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF3E0", edgecolor="#E67E22", alpha=0.9))

def draw_note(ax, x, y, text, color="#FFF9C4", edge="#F9A825"):
    ax.text(x, y, text, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor=color, edgecolor=edge))

# ============================================================
# 1. Node-Topic Architecture
# ============================================================
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("Track Drive - Node & Topic Architecture", fontsize=16, fontweight="bold", pad=20)

draw_state(ax, 0.3, 9, 2.4, 1, "Unity Sim", "ros_tcp_endpoint", "#E8D5B7")
draw_state(ax, 0.3, 6.5, 2.4, 1, "lane_detect", "OpenCV BEV\nHSV white/yellow", "#FF6B6B")
draw_state(ax, 0.3, 4, 2.4, 1, "yolo_detect", "YOLOv8n detect\n16 classes", "#FF6B6B")
draw_state(ax, 5, 7, 2.4, 1, "integration", "LiDAR cluster\nROI filter", "#45B7D1")
draw_state(ax, 8.5, 4.5, 3.5, 2.5, "path_planner", "State Machine\nIDLE>WAIT>CONE\n>LANE>SHORTCUT\n>OVERTAKE>PED", "#4ECDC4")
draw_state(ax, 13, 7, 2.5, 1, "motion", "Multi-point pursuit\nspeed/angle ctrl", "#96CEB4")
draw_state(ax, 13, 4, 2.5, 1, "fused_viewer", "BEV display", "#DDA0DD")

topics = [
    (3.5, 10, "/usb_cam/image_raw/front"),
    (3.5, 9.3, "/scan (LaserScan)"),
    (3.5, 8.6, "/imu"),
    (4, 6, "/detect/lane (z=cls)"),
    (4, 4.8, "/detect/road_pixels"),
    (4, 4.2, "/detect/events_raw"),
    (8, 8.2, "/fused/lane"),
    (8, 7.5, "/fused/obstacles"),
    (12.5, 8.5, "/center_path"),
    (12.5, 7.8, "/target"),
    (12.5, 3.2, "/emergency_stop"),
    (12.5, 2.5, "/left_turn1,2"),
    (12.5, 1.8, "/child_zone /slow_merge"),
    (15.5, 3.2, "/xycar_motor"),
]
for x, y, t in topics:
    ax.text(x, y, t, fontsize=7, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFE66D", edgecolor="#E67E22", lw=0.8))

arrows = [
    (2.7, 9.5, 1.5, 7.5), (2.7, 9.5, 1.5, 5),
    (2.7, 9.3, 5.5, 8), (2.7, 8.6, 9, 7),
    (2.7, 7, 4, 6.3), (2.7, 4.5, 4, 5.1), (2.7, 4.2, 4, 4.5),
    (7.4, 7.5, 8.5, 6.5), (7.4, 8.2, 8.5, 6.8),
    (5, 5.1, 8.5, 5.5), (5, 4.5, 8.5, 5.2),
    (12, 6.5, 12.5, 8.2), (12, 6.2, 12.5, 7.5),
    (12, 5, 12.5, 3.5), (12, 4.8, 12.5, 2.8), (12, 4.6, 12.5, 2.1),
    (15.5, 7.5, 15.5, 3.5),
]
for x1, y1, x2, y2 in arrows:
    draw_trans(ax, x1, y1, x2, y2, "", "#999")

ax.text(8, 0.8, "cls: 6=WHITE 8=YELLOW (lane_detect) | 0=BLACK_CAR 6=GREEN 10=LEFT 12=POLICE 13=RED 14=STOP (yolo)",
        fontsize=7, ha="center", color="#555")

plt.tight_layout()
plt.savefig("/home/xytron/xycar_ws/docs/01_node_topic_arch.png", dpi=150, bbox_inches="tight")
plt.close()
print("1/4 done")

# ============================================================
# 2. Main State Machine
# ============================================================
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("Main State Machine", fontsize=16, fontweight="bold", pad=20)

draw_state(ax, 0.5, 8.2, 2, 0.9, "IDLE", "wait sim", "#BDC3C7")
draw_state(ax, 3.5, 8.2, 2, 0.9, "WAIT", "GREEN 4s", "#F39C12")
draw_state(ax, 6.5, 8.2, 2, 0.9, "CONE", "cone drive", "#E74C3C")
draw_state(ax, 9.5, 8.2, 2.5, 0.9, "LANE", "yellow-line", "#27AE60")
draw_state(ax, 0.5, 5.5, 2.5, 1.2, "PEDESTRIAN", "person stop\nr:0.1~0.4 x>1.5m\nroad_half:1.2m", "#9B59B6")
draw_state(ax, 4, 5.5, 3, 1.2, "SHORTCUT", "left turn\nWAIT>TURN1>FOLLOW\n>TURN2", "#3498DB")
draw_state(ax, 8, 5.5, 3, 1.2, "OVERTAKE", "overtake\nTO_2ND>PASS>BESIDE\n>RETURN>MERGE", "#E67E22")
draw_state(ax, 12, 5.5, 2.5, 1.2, "CHILD_ZONE", "child protect\nspeed=6\n17s timeout", "#1ABC9C")
draw_state(ax, 9.5, 2.5, 2.5, 1, "RED_STOP", "stopline+noGREEN\nwait for GREEN", "#C0392B")

draw_trans(ax, 2.5, 8.65, 3.5, 8.65, "sim OK")
draw_trans(ax, 5.5, 8.65, 6.5, 8.65, "GREEN 4s")
draw_trans(ax, 8.5, 8.65, 9.5, 8.65, "miss30")
draw_trans(ax, 9.5, 8.2, 1.5, 6.7, "person")
draw_trans(ax, 10, 8.2, 5.5, 6.7, "stop+GREEN\n+noPOLICE")
draw_trans(ax, 11, 8.2, 9.5, 6.7, "car v>=250\nhdg50~170")
draw_trans(ax, 12, 8.5, 13, 6.7, "CHILD_START\nor after SC")
draw_trans(ax, 10.5, 8.2, 10.5, 3.5, "stop+noGREEN")

ax.text(7.5, 1.5, "CHILD_ZONE = speed cap inside LANE (not separate phase)\nRED_STOP = estop inside LANE (not separate phase)\nAll sub-states return to LANE when done",
        fontsize=8, ha="center", style="italic", color="#777")

plt.tight_layout()
plt.savefig("/home/xytron/xycar_ws/docs/02_state_machine.png", dpi=150, bbox_inches="tight")
plt.close()
print("2/4 done")

# ============================================================
# 3. OVERTAKE Flow
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("OVERTAKE Sub-State Flow", fontsize=16, fontweight="bold", pad=20)

ot = [
    (4, 11, 4, 0.7, "LANE", "car v>=250 hdg50~170 noPOLICE", "#27AE60"),
    (4, 9.5, 4, 0.7, "LANE_TO_2ND (6s)", "yellow right 1.8m", "#E67E22"),
    (4, 8, 4, 0.7, "PASSING", "white left 0.8m | left lidar wait", "#E67E22"),
    (4, 6.5, 4, 0.7, "CAR_BESIDE", "white left 5.5m | left gone wait", "#E67E22"),
    (4, 5, 4, 0.7, "RETURN (3s)", "yellow left 2.0m", "#3498DB"),
    (4, 3.5, 4, 0.7, "MERGE (2s)", "yellow right 0.5m", "#3498DB"),
    (4, 2, 4, 0.7, "LANE", "20s cooldown", "#27AE60"),
]
for x, y, w, h, l, s, c in ot:
    draw_state(ax, x, y, w, h, l, s, c)
for i, label in enumerate(["trigger", "6s", "left detect", "left gone", "3s", "2s"]):
    draw_trans(ax, 6, 11-i*1.5, 6, 11-i*1.5-0.8, label)

draw_note(ax, 1.5, 9.9, "path:\nyellow fit\n+ offset")
draw_note(ax, 1.5, 7.5, "path:\nwhite fit\n(min Y per x bin)\n+ outlier reject\n+ EMA a=0.15")
draw_note(ax, 10.5, 7.3, "left lidar:\n+90deg +/-8deg\n0.5~3.0m", "#E3F2FD", "#1565C0")

plt.tight_layout()
plt.savefig("/home/xytron/xycar_ws/docs/03_overtake_flow.png", dpi=150, bbox_inches="tight")
plt.close()
print("3/4 done")

# ============================================================
# 4. SHORTCUT Flow
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("SHORTCUT (Left Turn) Sub-State Flow", fontsize=16, fontweight="bold", pad=20)

sc = [
    (4, 11, 4, 0.7, "RED_STOP", "stopline + no GREEN = stop", "#C0392B"),
    (4, 9.5, 4, 0.7, "WAITING", "estop + wait LEFT (2 tick debounce)", "#3498DB"),
    (4, 8, 4, 0.7, "TURNING_1 (2.4s)", "/left_turn1 -> motion\nangle=-100 speed=10", "#E74C3C"),
    (4, 6.5, 4, 0.7, "FOLLOW", "_tick_lane() follow\nwait CROSSROAD v>=278", "#3498DB"),
    (4, 5, 4, 0.7, "TURNING_2 (2.5s)", "/left_turn2 -> motion\nangle=-100 speed=10", "#E74C3C"),
    (4, 3.5, 4, 0.7, "LANE (return)", "slow 3s + force CHILD_ZONE\n+ SC 10s cooldown", "#27AE60"),
]
for x, y, w, h, l, s, c in sc:
    draw_state(ax, x, y, w, h, l, s, c)
for i, label in enumerate(["GREEN+noPOLICE", "LEFT 2tick", "48 ticks", "CROSS v>=278", "50 ticks"]):
    draw_trans(ax, 6, 11-i*1.5, 6, 11-i*1.5-0.8, label)

draw_note(ax, 1.2, 10.8, "POLICE intersection:\nGREEN+POLICE<0.5s\n= just GO", "#FFEBEE", "#C62828")
draw_note(ax, 10.5, 8.4, "TURNING =\nmotion hardcode\npath_planner sends\n/left_turn1,2", "#E3F2FD", "#1565C0")
draw_note(ax, 10.5, 3.9, "after SHORTCUT:\nauto CHILD_ZONE\n(17s max)", "#E8F5E9", "#2E7D32")

plt.tight_layout()
plt.savefig("/home/xytron/xycar_ws/docs/04_shortcut_flow.png", dpi=150, bbox_inches="tight")
plt.close()
print("4/4 done - all saved to ~/xycar_ws/docs/")
