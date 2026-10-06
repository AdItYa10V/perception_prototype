#!/usr/bin/env python3
"""LEVEL 1 - pixel steering. No 3D anywhere.

gate/steering (Vector3Stamped):
  x = horizontal error, -1..1 (+ = gate right of image centre -> yaw right)
  y = vertical error,   -1..1 (+ = gate below centre          -> descend)
  z = bbox width / image width (closeness; grows as you approach)
gate/range_rough (Float32): fx * gate_width_m / bbox_width_px  (very rough, metres)
"""
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray


class GatePixel(Node):
    def __init__(self):
        super().__init__('gate_pose')
        for n, v in [('min_score', 0.4), ('select', 'score'), ('gate_width_m', 3.05),
                     ('aim_offset_x', 0.0), ('aim_offset_y', 0.0)]:
            self.declare_parameter(n, v)
        self.W = self.H = self.fx = None
        self.pub = self.create_publisher(Vector3Stamped, 'gate/steering', 10)
        self.pub_r = self.create_publisher(Float32, 'gate/range_rough', 10)
        self.create_subscription(CameraInfo, '/perception/camera_info', self.cb_info, 10)
        self.create_subscription(Detection2DArray, 'gate/detections', self.cb, 10)

    def cb_info(self, m):
        self.W, self.H, self.fx = m.width, m.height, m.k[0]

    def cb(self, msg):
        g = lambda n: self.get_parameter(n).value
        if self.W is None:
            return
        ds = [d for d in msg.detections if d.results and d.results[0].hypothesis.score >= g('min_score')]
        if not ds:
            return
        key = (lambda d: d.bbox.size_x) if g('select') == 'largest' else \
              (lambda d: d.results[0].hypothesis.score)
        d = max(ds, key=key)
        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = (d.bbox.center.position.x - self.W / 2) / (self.W / 2) - g('aim_offset_x')
        out.vector.y = (d.bbox.center.position.y - self.H / 2) / (self.H / 2) - g('aim_offset_y')
        out.vector.z = d.bbox.size_x / self.W
        self.pub.publish(out)
        self.pub_r.publish(Float32(data=float(self.fx * g('gate_width_m') / max(d.bbox.size_x, 1.0))))


def main():
    rclpy.init()
    rclpy.spin(GatePixel())


if __name__ == '__main__':
    main()
