#!/usr/bin/env python3
"""
YOLO seg 기반 차선 검출 — OpenCV lane_detect 대체용 (차선만).

friend/best.pt (segmentation, 12클래스) 사용.
  6=LANE(외곽선), 8=MID(중앙선) 마스크만 사용.
seg 마스크 → BEV 워프 → 기존 _mask_to_centerline_xy(검증된 ground 변환) → /detect/lane

발행 형식은 OpenCV lane_detect와 동일 (z=cls_id: 8=중앙, 6=외곽).
→ integration → lane_curve / plan() 그대로 소비 가능.

주의: 이 seg 모델엔 차량/경찰/CROSSROAD 클래스가 없음.
      신호/객체 검출은 기존 detection 모델(yolo_detect)이 계속 담당.
"""

import os
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32

from ultralytics import YOLO

# 검증된 BEV 워프 + ground 변환 재사용
from .lane_detect_node import _mask_to_centerline_xy, H_PIX2BEV, CLS_WHITE_LANE, CLS_YELLOW_MID
from .common import BEV_W, BEV_H, LIDAR_FRAME, lane_mask_to_lidar_xy

# seg 모델 클래스 id → 발행 z (lane 관습)
SEG_MID = 8         # 중앙선 → z=8
SEG_LANE = 6        # 외곽 흰 실선 → z=6
SEG_CHILD_LANE = 2  # 어린이보호 노란 실선 → z=2

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive
DEFAULT_MODEL = os.path.join(_PKG_ROOT, "config", "lane.pt")


class YoloLaneNode(Node):
    def __init__(self):
        super().__init__("yolo_lane_node")

        self.declare_parameter("model_path", DEFAULT_MODEL)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("centerline", True)   # 기본 0.18m 요약 centerline. False=raw 마스크 점구름(전체)

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)

        if not os.path.exists(model_path):
            self.get_logger().error(f"NO MODEL: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"loading seg model: {model_path}")
        self._model = YOLO(model_path)
        self.get_logger().info(f"task={self._model.task} names={self._model.names}")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front", self._on_image, qos_profile_sensor_data)
        self._pub_lane = self.create_publisher(PoseArray, "/detect/lane", 10)
        self._pub_quality = self.create_publisher(Float32, "/detect/lane_quality", 10)

        self._log = 0
        self.get_logger().info("yolo_lane_node started (seg lanes → /detect/lane)")

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return
        H_img, W_img = img.shape[:2]

        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz,
            device=self._device, verbose=False, retina_masks=False)
        r = results[0] if results else None

        out = PoseArray()
        out.header = msg.header
        out.header.frame_id = LIDAR_FRAME

        use_centerline = bool(self.get_parameter("centerline").value)
        n_mid = n_lane = 0
        if r is not None and r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy()   # (N, Hm, Wm)

            for seg_cls, z_id in ((SEG_MID, CLS_YELLOW_MID),):
                sel = np.where(cls == seg_cls)[0]
                if sel.size == 0:
                    continue
                binmask = np.zeros((H_img, W_img), dtype=np.uint8)
                for i in sel:
                    m = masks[i]
                    if m.shape != (H_img, W_img):
                        m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
                    binmask[m > 0.5] = 255
                # 카메라 픽셀 마스크 → BEV 워프
                bev = cv2.warpPerspective(
                    binmask, H_PIX2BEV, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST)
                # centerline=True: 0.18m 구간당 1점 / False: raw 마스크 점구름
                if use_centerline:
                    xs, ys = _mask_to_centerline_xy(bev)
                else:
                    xs, ys = lane_mask_to_lidar_xy(bev)
                for x, y in zip(xs, ys):
                    p = Pose()
                    p.position.x = float(x)
                    p.position.y = float(y)
                    p.position.z = float(z_id)
                    out.poses.append(p)
                if seg_cls == SEG_MID:
                    n_mid = len(xs)
                else:
                    n_lane = len(xs)

        self._pub_lane.publish(out)
        self._pub_quality.publish(Float32(data=1.0 if out.poses else 0.0))

        self._log += 1
        if self._log >= 20:
            self._log = 0
            self.get_logger().info(f"mid_pts={n_mid} lane_pts={n_lane}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
