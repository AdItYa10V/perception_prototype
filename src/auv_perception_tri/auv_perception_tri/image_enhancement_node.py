#!/usr/bin/env python3
"""Downscale + underwater colour correction + CLAHE; republishes CameraInfo for the scaled stream.

1080p30 in -> 960x540 out. Re-stamps images by -camera_latency_s so TF lookups in the
pose/triangulation nodes hit the vehicle pose at the moment of exposure, not arrival.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from auv_perception_tri.utils.camera_model import CameraModel


class ImageEnhancement(Node):
    def __init__(self):
        super().__init__('image_enhancement')
        for n, v in [('input_topic', '/camera/image_raw'),
                     ('output_topic', '/perception/image_enhanced'),
                     ('camera_info_topic', '/perception/camera_info'),
                     ('calib_file', ''), ('camera_frame', 'camera_optical_frame'),
                     ('scale', 0.5), ('camera_latency_s', 0.0),
                     ('red_compensation', True), ('red_alpha', 1.0),
                     ('clahe_clip', 2.0), ('clahe_tile', 8), ('gamma', 1.0),
                     ('skip_n', 1), ('cv_threads', 2)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        cv2.setNumThreads(int(g('cv_threads')))
        self.scale, self.latency = float(g('scale')), float(g('camera_latency_s'))
        self.red, self.alpha = bool(g('red_compensation')), float(g('red_alpha'))
        self.skip, self.frame = max(1, int(g('skip_n'))), g('camera_frame')
        self.clahe = cv2.createCLAHE(float(g('clahe_clip')), (int(g('clahe_tile')),) * 2)
        gm = float(g('gamma'))
        self.lut = None if abs(gm - 1.0) < 1e-3 else \
            (255.0 * (np.arange(256) / 255.0) ** (1.0 / gm)).astype(np.uint8)
        self.cam = CameraModel.from_yaml(g('calib_file')) if g('calib_file') else None
        self.bridge, self.count = CvBridge(), 0
        self.pub = self.create_publisher(Image, g('output_topic'), qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, g('camera_info_topic'), 10)
        self.create_subscription(Image, g('input_topic'), self.cb, qos_profile_sensor_data)

    def enhance(self, bgr):
        if self.red:  # underwater red-channel compensation (Ancuti-style, cheap)
            b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
            r = r + self.alpha * (g[::4, ::4].mean() - r[::4, ::4].mean()) \
                * (1.0 - r / 255.0) * (g / 255.0)
            bgr = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.LUT(out, self.lut) if self.lut is not None else out

    def cb(self, msg):
        self.count += 1
        if self.count % self.skip:
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        w2, h2 = int(w * self.scale), int(h * self.scale)
        if self.scale != 1.0:
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        out = self.bridge.cv2_to_imgmsg(self.enhance(img), 'bgr8')
        stamp = (Time.from_msg(msg.header.stamp) - Duration(seconds=self.latency)).to_msg()
        out.header.stamp, out.header.frame_id = stamp, self.frame
        self.pub.publish(out)
        if self.cam is not None:
            self.cam.set_image_size(w2, h2)
            ci = CameraInfo()
            ci.header = out.header
            ci.width, ci.height = w2, h2
            ci.k = self.cam.K.ravel().tolist()
            ci.d = self.cam.D.ravel().tolist()
            ci.distortion_model = 'equidistant' if self.cam.model == 'fisheye' else 'plumb_bob'
            self.pub_info.publish(ci)


def main():
    rclpy.init()
    rclpy.spin(ImageEnhancement())


if __name__ == '__main__':
    main()
