#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# H 캘리브레이션용 데이터 수집 노드 (임시 도구)
# 카메라 front + 라이다를 동시에 보고,
# 키로 차를 미세 제어하면서 원하는 순간 둘을 저장한다.
#=============================================
import rclpy, os, time, threading
import numpy as np
import cv2
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge


class CalibCollectNode(Node):

    def __init__(self):
        super().__init__('calib_collect')

        # 저장 경로 준비
        self.save_dir = os.path.expanduser('~/calib_data')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_idx = 0

        # 센서 데이터 보관함
        self.image = None
        self.scan_ranges = None      # 거리 배열
        self.scan_angle_min = None   # 시작 각도(rad)
        self.scan_angle_inc = None   # 각도 증가량(rad)
        self.bridge = CvBridge()

        # 모터 메시지
        self.motor_msg = XycarMotor()

        # Publisher / Subscriber
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        self.get_logger().info(f'calib_collect started. save_dir={self.save_dir}')

    #=============================================
    # 카메라 콜백: 최신 프레임을 OpenCV 이미지로 보관
    #=============================================
    def cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    #=============================================
    # 라이다 콜백: 거리 + 각도 정보까지 보관 (캘리브레이션엔 각도 필수)
    #=============================================
    def lidar_callback(self, msg):
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    #=============================================
    # 모터 발행 (한 번 쏘기)
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #=============================================
    # "한 틱" 미세 이동: 잠깐 가고 바로 정지
    #=============================================
    def tick_move(self, angle, speed, dur=0.3):
        self.drive(angle, speed)
        time.sleep(dur)
        self.drive(0, 0)

    #=============================================
    # 현재 카메라+라이다 한 쌍 저장
    #=============================================
    def capture(self):
        if self.image is None or self.scan_ranges is None:
            self.get_logger().warn('아직 센서 안 들어옴. 저장 취소.')
            return
        idx = self.save_idx
        img_path = os.path.join(self.save_dir, f'img_{idx:03d}.png')
        scan_path = os.path.join(self.save_dir, f'scan_{idx:03d}.npz')

        cv2.imwrite(img_path, self.image)
        # 라이다는 거리+각도정보를 함께 묶어서 저장
        np.savez(scan_path,
                 ranges=self.scan_ranges,
                 angle_min=self.scan_angle_min,
                 angle_increment=self.scan_angle_inc)

        self.get_logger().info(f'saved #{idx}  ->  {img_path} / {scan_path}')
        self.save_idx += 1

    #=============================================
    # 메인 루프: 화면 표시 + 키 입력 처리
    #=============================================
    def main_loop(self):
        self.get_logger().info("===== 키 안내 =====")
        self.get_logger().info(" w/s: 전진/후진,  a/d: 좌/우조향 전진")
        self.get_logger().info(" c: 저장,  q: 종료")

        SPEED = 3.0    # 미세 이동 속도 (느리게)
        ANGLE = 20.0   # 조향각

        while rclpy.ok():
            if self.image is not None:
                view = self.image.copy()
                # 화면에 현재 저장 개수 + 라이다 점 개수 표시
                npts = 0 if self.scan_ranges is None else int(np.sum(np.isfinite(self.scan_ranges)))
                txt = f'saved={self.save_idx}  lidar_pts={npts}'
                cv2.putText(view, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.imshow('calib_collect (front)', view)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('w'):
                self.tick_move(0, SPEED)
            elif key == ord('s'):
                self.tick_move(0, -SPEED)
            elif key == ord('a'):
                self.tick_move(-ANGLE, SPEED)
            elif key == ord('d'):
                self.tick_move(ANGLE, SPEED)
            elif key == ord('c'):
                self.capture()
            elif key == ord('q'):
                break


def main(args=None):
    rclpy.init(args=args)
    node = CalibCollectNode()

    # spin을 별도 스레드에서 돌려 콜백이 계속 최신값 갱신하게 함
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0, 0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()