#!/usr/bin/env python3
"""
YOLO 차선 검출 노드 — best.pt segmentation으로 차선 픽셀 추출.

lane_detect_node를 대체. 좌표 변환은 하지 않고, 차선 segmentation 마스크의
원본 이미지 픽셀 좌표 (u, v)를 그대로 발행. 호모그래피 → lidar_frame 미터
변환은 integration_node가 담당.

구독:
  /usb_cam/image_raw/front (Image, rgb8 640×480)

발행:
  /detect/lane_pixels (PoseArray, frame_id=usb_cam_front)
    LANE/CHILD_LANE 클래스만 필터링. integration_node가 구독해서
    lidar_frame 미터로 변환.
    position.x = u, position.y = v, position.z = cls_id

  /detect/yolo_pixels (PoseArray, frame_id=usb_cam_front)
    모든 감지 인스턴스의 segmentation 픽셀을 한 토픽에 통합.
    다른 노드(신호등 판단, 결승선, 점선 인식 등)가 cls_id로
    필터링해서 사용. 메시지 형식은 lane_pixels와 동일.

모델: best.pt (YOLOv8n-seg). 클래스 매핑:
  0=frontview 1=CHILD_END 2=CHILD_LANE 3=CHILD_START 4=GOAL
  5=GREEN 6=LANE 7=LEFT 8=MID 9=RED 10=STOP 11=YELLOW
차선으로 취급하는 클래스: 2=CHILD_LANE, 6=LANE.

품질 토픽은 발행하지 않음 — integration_node가 좌표 변환 후
점 개수/스팬 기준으로 /fused/lane_quality를 직접 계산.
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

from .common import CAM_FRONT_FRAME


# best.pt 클래스 매핑 (참고)
# 0=frontview 1=CHILD_END 2=CHILD_LANE 3=CHILD_START 4=GOAL 5=GREEN
# 6=LANE 7=LEFT 8=MID 9=RED 10=STOP 11=YELLOW
LANE_CLASS_IDS = {2, 6}

# 마스크 픽셀 서브샘플 — N마다 1개만 토픽에 실음
PIXEL_SUBSAMPLE = 3


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")

        self.declare_parameter("model_path", "/Users/daehyeon/Downloads/kookmin2/best.pt")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cpu")

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)

        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLO model not found: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"loading YOLO model: {model_path}")
        self._model = YOLO(model_path)
        self.get_logger().info(f"classes: {self._model.names}")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front",
            self._on_image, qos_profile_sensor_data)

        self._pub_lane_pixels = self.create_publisher(
            PoseArray, "/detect/lane_pixels", 10)
        self._pub_all_pixels = self.create_publisher(
            PoseArray, "/detect/yolo_pixels", 10)

        self.get_logger().info("yolo_detector_node started")

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return

        results = self._model.predict(
            img,
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device,
        )

        lane_out = PoseArray()
        lane_out.header = msg.header
        lane_out.header.frame_id = CAM_FRONT_FRAME

        all_out = PoseArray()
        all_out.header = msg.header
        all_out.header.frame_id = CAM_FRONT_FRAME

        r = results[0] if results else None
        if r is not None and r.masks is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()                  # (N, Hm, Wm)
            classes = r.boxes.cls.cpu().numpy().astype(int)     # (N,)
            H, W = img.shape[:2]

            for mask, cls_id in zip(masks, classes):
                if mask.shape != (H, W):
                    mask = cv2.resize(
                        mask, (W, H), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(mask > 0.5)
                if xs.size == 0:
                    continue
                ys = ys[::PIXEL_SUBSAMPLE]
                xs = xs[::PIXEL_SUBSAMPLE]

                cls_f = float(cls_id)
                is_lane = int(cls_id) in LANE_CLASS_IDS
                for u, v in zip(xs, ys):
                    p = Pose()
                    p.position.x = float(u)
                    p.position.y = float(v)
                    p.position.z = cls_f
                    all_out.poses.append(p)
                    if is_lane:
                        q = Pose()
                        q.position.x = p.position.x
                        q.position.y = p.position.y
                        q.position.z = p.position.z
                        lane_out.poses.append(q)

        self._pub_lane_pixels.publish(lane_out)
        self._pub_all_pixels.publish(all_out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
