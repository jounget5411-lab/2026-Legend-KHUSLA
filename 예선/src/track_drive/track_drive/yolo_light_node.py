#!/usr/bin/env python3
"""
신호등 전용 YOLO 노드 — friend/legend_best.pt (segment, {0:green, 1:left, 2:red}).
주황불을 red로 학습한 모델이라 기존 detection 모델의 GREEN 오인(주황->green) 문제 해결용.
신호등 상태만 /detect/traffic_light (String: GREEN/RED/NONE)로 발행.
좌회전 화살표/기타 객체는 기존 yolo_detect(config/best.pt)가 계속 담당.
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

GREEN_CLS = 0
LEFT_CLS = 1
RED_CLS = 2

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive
DEFAULT_MODEL_PATH = os.path.join(_PKG_ROOT, "config", "light.pt")


class YoloLightNode(Node):
    def __init__(self):
        super().__init__("yolo_light_node")
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("swap_rb", True)   # RGB->BGR 변환 (legend 모델 색순서 대응)

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)
        self._swap_rb = bool(self.get_parameter("swap_rb").value)

        if not os.path.exists(model_path):
            self.get_logger().error(f"NO MODEL: {model_path}")
            raise FileNotFoundError(model_path)

        self._model = YOLO(model_path)
        self.get_logger().info(f"light model task={self._model.task} names={self._model.names}")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front", self._on_image, qos_profile_sensor_data)
        self._pub = self.create_publisher(String, "/detect/traffic_light", 10)
        self._log = 0
        self.get_logger().info("yolo_light_node started")

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except ValueError:
            return

        # legend_best.pt는 표준 색순서로 학습됨. ultralytics는 numpy를 BGR로 가정하므로
        # rgb8을 그대로 넣으면 R<->B가 뒤바뀌어 빨강<->파랑 오인 → RGB->BGR 변환.
        # (친구의 다른 모델들은 RGB 그대로 학습돼 변환 안 함. 방향 반대면 swap_rb:=false)
        if self._swap_rb:
            img = np.ascontiguousarray(img[:, :, ::-1])

        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False)
        r = results[0] if results else None

        state = "NONE"
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            green_conf = float(confs[cls == GREEN_CLS].max()) if (cls == GREEN_CLS).any() else 0.0
            red_conf = float(confs[cls == RED_CLS].max()) if (cls == RED_CLS).any() else 0.0
            if green_conf > 0.0 and green_conf >= red_conf:
                state = "GREEN"
            elif red_conf > 0.0:
                state = "RED"

        self._pub.publish(String(data=state))
        self._log += 1
        if self._log >= 20:
            self._log = 0
            self.get_logger().info(f"light={state}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
