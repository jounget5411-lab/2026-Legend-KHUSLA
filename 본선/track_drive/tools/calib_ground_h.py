#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""바닥 호모그래피 H 캘리브 — undistort 이미지 픽셀 → 차량좌표(m) (독립 CLI, ROS 불필요).

역할: camera.yaml 의 K/D 로 undistort 한 이미지 위에서 바닥 기준점을 클릭하고,
      각 점의 실측 차량좌표(x전방/y좌, m)를 콘솔로 받아 cv2.findHomography 로
      H(픽셀→차량좌표)를 계산. 재투영 오차 출력 후 camera.yaml 의 H 갱신.
입력: --image 바닥 기준점이 보이는 사진, --camera camera.yaml (K/D 필요),
      비대화 모드는 --points CSV (한 줄에 u,v,x,y — u,v 는 undistort 픽셀).
출력: 콘솔에 H/점별 오차/RMS(m). --save 시 camera.yaml 의 H 갱신 + calibrated 재계산.
예선 대비 변경점: 신규 도구 (예선 H 는 시뮬 tf_static 기반 자동 산출이었음).

사용법:
  # 대화 모드 — 클릭 → 콘솔에 "x y" 입력 반복, q 로 종료 후 계산
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml --save
  # 비대화 모드 — 측정값 CSV 재사용 (u,v,x,y / # 주석 허용)
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml \
      --points pts.csv --save
  --save [경로] : 경로 생략 시 --camera 파일에 저장.

좌표/규약:
  - 차량좌표 = lidar_frame: x 전방+, y 좌+ (단위 m). 예) 앞 1.2m 오른쪽 0.3m → "1.2 -0.3"
  - H 는 undistort(new_K=K, lib/bev.py Undistorter 와 동일 규약) 이미지 픽셀 기준.
    K/D 가 미캘리브(단위행렬)면 undistort 를 건너뛰고 경고 — calib_fisheye.py 먼저.
  - 점 4개 이상 필요. 한 직선 위 점들만 찍으면 H 퇴화 — 전후/좌우로 퍼뜨릴 것.
"""

import argparse
import datetime
import os
import re
import sys

import cv2
import numpy as np
import yaml

WIN = "calib_ground_h"
MIN_POINTS = 4        # findHomography 최소 점 수
RMS_WARN_M = 0.05     # 재투영 RMS 가 이보다 크면 경고 (5cm)


# ======================== camera.yaml 로드/병합 저장 (calib_fisheye.py 와 동일 규약) ========================

def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _flag(data, key):
    """부분 캘리브 플래그. 구버전 파일(플래그 없음)은 calibrated 값으로 간주."""
    if key in data:
        return bool(data[key])
    return bool(data.get("calibrated", False))


def _save_camera_yaml(path, data, tool_name):
    """기존 키 보존 + 지정 순서로 기록. (주의: 기존 파일의 주석은 사라짐)"""
    order = ["calibrated", "intrinsics_calibrated", "h_calibrated",
             "image_width", "image_height", "K", "D", "H"]
    out = {k: data[k] for k in order if k in data}
    for k, v in data.items():
        if k not in out:
            out[k] = v
    header = (
        "# 실차 카메라 캘리브 파일 — track_drive.lib.bev.load_camera_config() 가 읽는다.\n"
        "# %s 가 %s 에 갱신.\n"
        "# calibrated: K/D(어안)와 H(바닥) 둘 다 완료 시에만 true (부분 플래그는 아래 참고)\n"
        "# K/D: fisheye 내부 파라미터.  H: undistort(new_K=K) 이미지 픽셀 → lidar 미터 (x전방/y좌)\n"
        % (tool_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None,
                       allow_unicode=True)
    print("저장: %s" % path)


def _load_kd(path):
    """camera.yaml 에서 K/D 만 읽는다 (calibrated 값과 무관하게 원본 사용)."""
    data = _load_yaml(path)
    if not data:
        sys.exit("camera.yaml 읽기 실패: %s" % path)
    try:
        K = np.asarray(data["K"], np.float64).reshape(3, 3)
        D = np.asarray(data["D"], np.float64).reshape(-1)[:4]
    except (KeyError, TypeError, ValueError):
        sys.exit("camera.yaml 에 K/D 없음/형식 오류: %s" % path)
    return K, D


def _undistort(img, K, D):
    """lib/bev.py Undistorter 와 동일 규약: new_K = K (H 좌표계 일치 필수)."""
    if np.allclose(K, np.eye(3)) or not np.any(D):
        print("[경고] K/D 미캘리브(단위행렬) — undistort 생략, 원본(raw) 픽셀 기준 H 가 됨.")
        print("        calib_fisheye.py 를 먼저 돌리는 것을 강력 권장.")
        return img
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D.reshape(4, 1), np.eye(3), K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ======================== 점 수집 ========================

def _load_points_csv(path):
    """CSV: 한 줄에 u,v,x,y (쉼표/공백 구분, # 주석·빈 줄 허용)."""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = re.split(r"[,\s]+", s)
            if len(vals) != 4:
                sys.exit("%s %d행: u,v,x,y 4개 값 필요: %r" % (path, ln, s))
            try:
                u, v, x, y = (float(t) for t in vals)
            except ValueError:
                sys.exit("%s %d행: 숫자 변환 실패: %r" % (path, ln, s))
            pts.append([u, v, x, y])
    return pts


def _draw_points(base, pts):
    disp = base.copy()
    for i, (u, v, x, y) in enumerate(pts):
        c = (int(round(u)), int(round(v)))
        cv2.drawMarker(disp, c, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(disp, "%d(%.2f,%.2f)" % (i + 1, x, y),
                    (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(disp, "click ground point | [u] undo  [q] done (%d pts)" % len(pts),
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return disp


def _collect_interactive(base):
    """클릭 → 콘솔에 실측 차량좌표 입력. q/ESC 종료, u 마지막 점 취소."""
    print("\n[안내] 이미지에서 바닥 기준점 클릭 → 콘솔에 실측 'x y' (m) 입력.")
    print("  x=전방+, y=좌+ (lidar_frame). 예) 앞1.2m 오른쪽0.3m → 1.2 -0.3")
    print("  키: [u] 마지막 점 취소 / [q]·[ESC] 입력 종료 (%d점 이상)" % MIN_POINTS)
    print("  팁: 한 직선 위 점만 찍으면 H 퇴화 — 전후·좌우로 퍼뜨릴 것.\n")

    pts = []      # 확정 [u, v, x, y]
    pending = []  # 클릭됐지만 좌표 미입력

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((mx, my))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        cv2.imshow(WIN, _draw_points(base, pts))
        key = cv2.waitKey(30) & 0xFF
        while pending:
            u, v = pending.pop(0)
            tmp = _draw_points(base, pts)
            cv2.drawMarker(tmp, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(WIN, tmp)
            cv2.waitKey(1)  # 콘솔 입력 동안 창 갱신 멈춤 — 정상
            s = input("점%d 픽셀(%d,%d) 의 차량좌표 x y (m, 빈 입력=이 점 취소): "
                      % (len(pts) + 1, u, v)).strip()
            if not s:
                print("  → 취소")
                continue
            vals = re.split(r"[,\s]+", s)
            try:
                x, y = float(vals[0]), float(vals[1])
            except (ValueError, IndexError):
                print("  → 형식 오류 (예: 1.2 -0.3) — 이 점 취소")
                continue
            pts.append([float(u), float(v), x, y])
        if key == ord("u") and pts:
            rm = pts.pop()
            print("취소: 점%d (%.0f,%.0f)→(%.2f,%.2f)" % (len(pts) + 1, *rm))
        elif key in (ord("q"), 27):
            break
    return pts


# ======================== H 계산/검증 ========================

def _compute_h(pts):
    arr = np.asarray(pts, np.float64)
    uv, xy = arr[:, :2], arr[:, 2:]
    H, _ = cv2.findHomography(uv, xy, 0)  # 점 수 적음 — 전점 최소자승
    if H is None:
        sys.exit("H 계산 실패 — 점 배치가 퇴화(일직선?)했거나 좌표 오입력")
    proj = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - xy, axis=1)
    print("\n===== 재투영 오차 (픽셀 → H → 차량좌표, 단위 m) =====")
    print(" 점   픽셀(u,v)        실측(x,y)          H투영(x,y)         오차")
    for i in range(len(err)):
        print(" %2d  (%6.1f,%6.1f)  (%6.3f,%6.3f)  (%6.3f,%6.3f)  %.3f"
              % (i + 1, uv[i, 0], uv[i, 1], xy[i, 0], xy[i, 1],
                 proj[i, 0], proj[i, 1], err[i]))
    rms = float(np.sqrt(np.mean(err ** 2)))
    print("RMS %.3f m / 최대 %.3f m" % (rms, float(err.max())))
    if rms > RMS_WARN_M:
        print("[경고] RMS %.0fcm > %.0fcm — 실측/클릭 재확인 권장"
              % (rms * 100, RMS_WARN_M * 100))
    np.set_printoptions(precision=8, suppress=False)
    print("H =\n%s" % H)
    return H, rms


def _show_verify(base, pts, H):
    """실측 좌표를 H 역변환으로 되투영(초록 원) — 클릭점(빨강 십자)과 겹치면 정상."""
    disp = _draw_points(base, pts)
    Hinv = np.linalg.inv(H)
    xy = np.asarray(pts, np.float64)[:, 2:]
    back = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    for u, v in back:
        cv2.circle(disp, (int(round(u)), int(round(v))), 7, (0, 255, 0), 2)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(disp, "green=measured reprojected | any key = quit", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(WIN, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="바닥 H 캘리브 — 픽셀→차량좌표 (자세한 사용법은 파일 헤더)")
    ap.add_argument("--image", required=True, help="바닥 기준점 사진")
    ap.add_argument("--camera", required=True, help="camera.yaml (K/D 로 undistort)")
    ap.add_argument("--points", default=None,
                    help="비대화 모드: u,v,x,y CSV (u,v 는 undistort 픽셀)")
    ap.add_argument("--save", nargs="?", const="@camera", default=None,
                    help="H 저장. 경로 생략 시 --camera 파일에 병합")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit("이미지 읽기 실패: %s" % args.image)
    K, D = _load_kd(args.camera)
    und = _undistort(img, K, D)

    if args.points:
        pts = _load_points_csv(args.points)
        print("CSV 로드: %d점 (%s)" % (len(pts), args.points))
    else:
        pts = _collect_interactive(und)

    if len(pts) < MIN_POINTS:
        sys.exit("점 %d개 — 최소 %d개 필요" % (len(pts), MIN_POINTS))

    H, rms = _compute_h(pts)

    if not args.points:
        _show_verify(und, pts, H)

    if args.save is not None:
        save_path = args.camera if args.save == "@camera" else args.save
        data = _load_yaml(save_path)
        data["H"] = [[float(v) for v in row] for row in H]
        data["h_calibrated"] = True
        data["h_rms_m"] = float(round(rms, 4))
        data["h_points"] = [[float(v) for v in p] for p in pts]  # 기록용 (u,v,x,y)
        intr = _flag(data, "intrinsics_calibrated")
        data["calibrated"] = bool(intr)  # K/D 까지 있어야 true
        _save_camera_yaml(save_path, data, "tools/calib_ground_h.py")
        if not intr:
            print("[경고] K/D 미캘리브 상태 — calibrated 는 false 유지. "
                  "calib_fisheye.py 후 이 도구를 다시 돌릴 것 (undistort 기준이 달라짐).")
        else:
            print("다음 단계: verify_lidar_cam.py 로 라이다-카메라 정합 육안 확인")


if __name__ == "__main__":
    main()
