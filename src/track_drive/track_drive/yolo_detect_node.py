#!/usr/bin/env python3
"""
YOLO 검출 노드 — best.pt(YOLOv8n-seg)로 세그멘테이션 검출.

좌표 변환(H)은 하지 않음 — 원본 카메라 픽셀(u,v) + cls_id를 발행.
H 변환은 integration_node가 담당.

구독: /usb_cam/image_raw/front (Image, rgb8 640×480)
발행:
  /detect/road_pixels (PoseArray) — H변환 필요한 클래스: LANE(6), MID(8),
     CHILD_LANE(2), STOP(10), GOAL(4). x=u, y=v, z=cls_id
  /detect/events_raw (PoseArray) — 플래그용 클래스: RED(9), YELLOW(11),
     GREEN(5), LEFT(7), CHILD_START(3), CHILD_END(1). z=cls_id만 의미 있음.
     frontview(0)는 버림.

클래스 매핑:
  0=frontview 1=CHILD_END 2=CHILD_LANE 3=CHILD_START 4=GOAL
  5=GREEN 6=LANE 7=LEFT 8=MID 9=RED 10=STOP 11=YELLOW
"""

import os
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray

from ultralytics import YOLO

# ======================== 상수 ========================

CAM_FRONT_FRAME = "usb_cam_front"

# H변환이 필요한 클래스 (도로 위 위치 의미 있음)
ROAD_CLASS_IDS = {2, 4, 6, 8, 10}  # CHILD_LANE, GOAL, LANE, MID, STOP

# 플래그용 클래스 (감지 여부만 의미 있음)
EVENT_CLASS_IDS = {1, 3, 5, 7, 9, 11}  # CHILD_END, CHILD_START, GREEN, LEFT, RED, YELLOW

# 마스크 픽셀 서브샘플 — N마다 1개만 발행
PIXEL_SUBSAMPLE = 3

# 기본 모델 경로 (워크스페이스 기준)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), "xycar_ws", "src", "track_drive", "config", "best.pt")

# ======================== ROS 노드 ========================

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cpu")

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"Loading YOLO: {model_path}")
        self._model = YOLO(model_path)
        self.get_logger().info(f"Classes: {self._model.names}")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)

        self._pub_road = self.create_publisher(PoseArray, "/detect/road_pixels", 10)
        self._pub_events = self.create_publisher(PoseArray, "/detect/events_raw", 10)

        self.get_logger().info("yolo_detect_node started")

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return

        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz,
            verbose=False, device=self._device)

        road_out = PoseArray()
        road_out.header = msg.header
        road_out.header.frame_id = CAM_FRONT_FRAME

        events_out = PoseArray()
        events_out.header = msg.header
        events_out.header.frame_id = CAM_FRONT_FRAME

        r = results[0] if results else None
        if r is not None and r.masks is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            img_h, img_w = img.shape[:2]

            for mask, cls_id in zip(masks, classes):
                cls_id = int(cls_id)
                if cls_id == 0:  # frontview → 버림
                    continue

                if cls_id in ROAD_CLASS_IDS:
                    if mask.shape != (img_h, img_w):
                        mask = cv2.resize(mask, (img_w, img_h),
                                          interpolation=cv2.INTER_NEAREST)
                    vs, us = np.where(mask > 0.5)
                    if us.size == 0:
                        continue
                    vs = vs[::PIXEL_SUBSAMPLE]
                    us = us[::PIXEL_SUBSAMPLE]
                    cls_f = float(cls_id)
                    for u, v in zip(us, vs):
                        p = Pose()
                        p.position.x = float(u)
                        p.position.y = float(v)
                        p.position.z = cls_f
                        road_out.poses.append(p)

                elif cls_id in EVENT_CLASS_IDS:
                    p = Pose()
                    p.position.z = float(cls_id)
                    events_out.poses.append(p)

        self._pub_road.publish(road_out)
        self._pub_events.publish(events_out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
