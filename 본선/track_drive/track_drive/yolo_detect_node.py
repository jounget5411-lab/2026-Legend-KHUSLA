#!/usr/bin/env python3
"""
YOLO 검출 노드 — best.pt(YOLOv8n-detect)로 객체 검출.

좌표 변환(H)은 하지 않음 — 원본 카메라 픽셀(u,v) + cls_id를 발행.
거리 판단은 path_planner가 v 픽셀 임계값(bbox 아래 끝)으로 처리.

구독: /usb_cam/image_raw/front (Image, rgb8 640×480)
발행:
  /detect/road_pixels (PoseArray) — 위치가 의미 있는 객체. 한 detection당 1점.
     x = u_center (bbox 좌우 중심)
     y = v_bottom (bbox 아래 끝 — 카메라에서 가까울수록 큰 값, 거리 proxy)
     z = cls_id
     포함 클래스: STOP(14), CROSSROAD_OUT(4), GOAL(5),
                 HUMAN(8), BLACK_CAR(0), GREEN_CAR(7)
  /detect/events_raw (PoseArray) — 존재 여부만 의미 있는 플래그.
     z = cls_id (x, y는 의미 없음)
     포함 클래스: GREEN(6), RED(13), YELLOW(15), LEFT(10),
                 CHILD_START(3), CHILD_END(1), POLICE(12)

차선 클래스(LANE=9, MID=11, CHILD_LANE=2)는 OpenCV(lane_detect_node)가 담당
하므로 무시.

클래스 매핑 (best.pt 학습 시점 기준):
  0=BLACK_CAR  1=CHILD_END  2=CHILD_LANE  3=CHILD_START
  4=CROSSROAD_OUT  5=GOAL  6=GREEN  7=GREEN_CAR  8=HUMAN
  9=LANE  10=LEFT  11=MID  12=POLICE  13=RED  14=STOP  15=YELLOW
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray

from ultralytics import YOLO

# ======================== 상수 ========================

CAM_FRONT_FRAME = "usb_cam_front"

# 위치(픽셀) 정보가 의미 있는 클래스 — bbox 1점 발행
# STOP/CROSSROAD_OUT/GOAL: 도로 위 마킹, bbox 아래 끝(y2) = 거리 proxy
# HUMAN/BLACK_CAR/GREEN_CAR: 객체, bbox 아래 끝 = 발치 = 거리 proxy
DETECT_CLASS_IDS = {0, 1, 3, 4, 5, 7, 8, 14}  # CHILD_START(3), CHILD_END(1) 추가 — v 거리 필요

# 존재 여부만 의미 있는 플래그 클래스 — cls_id만 발행
EVENT_CLASS_IDS = {6, 10, 12, 13, 15}

# 명시적으로 무시할 클래스 (OpenCV 차선 검출이 담당)
IGNORE_CLASS_IDS = {2, 9, 11}  # CHILD_LANE, LANE, MID

# 기본 모델 경로 — 워크스페이스 src 절대경로
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive
DEFAULT_MODEL_PATH = os.path.join(_PKG_ROOT, "config", "best.pt")

# ======================== ROS 노드 ========================

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cuda")

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)

        if not os.path.exists(model_path):
            self.get_logger().error("NO MODEL")
            raise FileNotFoundError(model_path)

        self.get_logger().info("YOLO LOAD")
        self._model = YOLO(model_path)
        self.get_logger().info("YOLO CLS")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)

        self._pub_road = self.create_publisher(PoseArray, "/detect/road_pixels", 10)
        self._pub_events = self.create_publisher(PoseArray, "/detect/events_raw", 10)

        self.get_logger().info("YOLO")

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
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()       # (N, 4): x1, y1, x2, y2
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for bbox, cls_id in zip(boxes, classes):
                cls_id = int(cls_id)

                if cls_id in IGNORE_CLASS_IDS:
                    continue

                if cls_id in DETECT_CLASS_IDS:
                    x1, y1, x2, y2 = bbox
                    u_center = float((x1 + x2) * 0.5)
                    v_bottom = float(y2)
                    p = Pose()
                    p.position.x = u_center
                    p.position.y = v_bottom
                    p.position.z = float(cls_id)
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
