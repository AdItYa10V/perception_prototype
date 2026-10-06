#!/usr/bin/env python3
"""YOLO11n TensorRT detector. Publishes boxes (vision_msgs) and, for pose models,
gate keypoints as PolygonStamped (x, y in enhanced-image pixels, z = keypoint confidence)."""
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

PKG = 'auv_perception_pixel'


class GateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector')
        for n, v in [('image_topic', '/perception/image_enhanced'), ('model_path', ''),
                     ('imgsz', [384, 640]), ('conf', 0.4), ('iou', 0.5), ('max_det', 3),
                     ('device', 0), ('half', True), ('use_keypoints', False),
                     ('publish_debug', False)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        path = g('model_path') or os.path.join(
            get_package_share_directory(PKG), 'models', 'gate_model.engine')
        from ultralytics import YOLO  # heavy import, keep local
        self.kp = bool(g('use_keypoints'))
        self.model = YOLO(path, task='pose' if self.kp else 'detect')
        self.kw = dict(imgsz=list(g('imgsz')), conf=float(g('conf')), iou=float(g('iou')),
                       max_det=int(g('max_det')), device=int(g('device')),
                       half=bool(g('half')), verbose=False)
        self.model.predict(np.zeros((540, 960, 3), np.uint8), **self.kw)  # warm-up
        self.bridge, self.debug, self.t_log = CvBridge(), bool(g('publish_debug')), 0.0
        self.pub_det = self.create_publisher(Detection2DArray, 'gate/detections', 10)
        self.pub_kp = self.create_publisher(PolygonStamped, 'gate/keypoints', 10)
        self.pub_dbg = self.create_publisher(Image, 'gate/debug', 1)
        self.create_subscription(Image, g('image_topic'), self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        t0 = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        r = self.model.predict(img, **self.kw)[0]
        arr = Detection2DArray()
        arr.header = msg.header
        n = 0 if r.boxes is None else len(r.boxes)
        if n:
            xywh = r.boxes.xywh.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            for i in range(n):
                d = Detection2D()
                d.header, d.id = msg.header, str(i)
                d.bbox.center.position.x, d.bbox.center.position.y = float(xywh[i, 0]), float(xywh[i, 1])
                d.bbox.size_x, d.bbox.size_y = float(xywh[i, 2]), float(xywh[i, 3])
                h = ObjectHypothesisWithPose()
                h.hypothesis.class_id, h.hypothesis.score = str(int(cls[i])), float(conf[i])
                d.results.append(h)
                arr.detections.append(d)
            if self.kp and r.keypoints is not None:
                b = int(np.argmax(conf))
                xy = r.keypoints.xy[b].cpu().numpy()
                kc = r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None \
                    else np.ones(len(xy))
                poly = PolygonStamped()
                poly.header = msg.header
                poly.polygon.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(c))
                                       for p, c in zip(xy, kc)]
                self.pub_kp.publish(poly)
        self.pub_det.publish(arr)
        if self.debug:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(r.plot(), 'bgr8'))
        now = time.perf_counter()
        if now - self.t_log > 5.0:
            self.t_log = now
            self.get_logger().info(f'inference+pub {1000 * (now - t0):.1f} ms, {n} det')


def main():
    rclpy.init()
    rclpy.spin(GateDetector())


if __name__ == '__main__':
    main()
