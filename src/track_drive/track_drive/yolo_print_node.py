#!/usr/bin/env python3
"""
best_3.pt 검출 결과 실시간 로그 출력 노드 (디버그용).

시뮬에서 키보드로 주행하면서 best_3가 무엇을 어떤 confidence로 잡는지 확인.
프레임마다 검출된 모든 객체를 한 줄씩 출력 (여러 개면 여러 줄).
차량(GREEN_CAR/BLACK_CAR), 신호등(노란불 등) 인식 여부 점검용.
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from ultralytics import YOLO

DEFAULT_MODEL = os.path.join(
    os.path.expanduser("~"), "xycar_ws", "src", "track_drive", "friend", "best_3.pt")


class YoloPrintNode(Node):
    def __init__(self):
        super().__init__("yolo_print_node")

        self.declare_parameter("model_path", DEFAULT_MODEL)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cuda")

        model_path = str(self.get_parameter("model_path").value)
        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = str(self.get_parameter("device").value)

        if not os.path.exists(model_path):
            self.get_logger().error(f"NO MODEL: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"loading model: {model_path}")
        self._model = YOLO(model_path)
        self._names = self._model.names
        self.get_logger().info(f"task={self._model.task} names={self._names}")

        self.create_subscription(
            Image, "/usb_cam/image_raw/front", self._on_image, qos_profile_sensor_data)

        self._frame = 0
        self.get_logger().info("yolo_print_node started")

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            return

        self._frame += 1
        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz,
            device=self._device, verbose=False)
        r = results[0] if results else None

        if r is None or r.boxes is None or len(r.boxes) == 0:
            return

        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()

        order = np.argsort(-confs)
        lines = []
        for i in order:
            c = int(cls[i])
            if isinstance(self._names, dict):
                name = self._names.get(c, str(c))
            else:
                name = self._names[c]
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            lines.append(f"  {c:2d} {name:<12} conf={confs[i]:.2f} "
                         f"box=({cx:.0f},{cy:.0f}) area={area:.0f}")

        self.get_logger().info(
            f"[f{self._frame}] {len(lines)} det:\n" + "\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = YoloPrintNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
