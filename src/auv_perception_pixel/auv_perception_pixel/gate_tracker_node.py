#!/usr/bin/env python3
"""Constant-velocity Kalman filter on each steering channel (x, y, size).
Smooths detector jitter and coasts through short dropouts (bubbles, glare) at a fixed
publish rate. gate/lock goes False after coast_s without a detection."""
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Bool


class CV1D:
    def __init__(self, x0, q, r):
        self.x, self.P, self.q, self.r = np.array([x0, 0.0]), np.diag([1.0, 1.0]), q, r

    def predict(self, dt):
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q

    def update(self, z):
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0])
        self.P = self.P - np.outer(K, self.P[0, :])


class GateTrackerPixel(Node):
    def __init__(self):
        super().__init__('gate_tracker')
        for n, v in [('coast_s', 0.5), ('q', 2.0), ('r', 0.0025), ('publish_rate', 30.0)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        self.coast, self.q, self.r = g('coast_s'), g('q'), g('r')
        self.f, self.t, self.t_meas = None, None, None
        self.pub = self.create_publisher(Vector3Stamped, 'gate/steering_filtered', 10)
        self.pub_lock = self.create_publisher(Bool, 'gate/lock', 10)
        self.create_subscription(Vector3Stamped, 'gate/steering', self.cb, 10)
        self.create_timer(1.0 / g('publish_rate'), self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def predict_to(self, t):
        dt = t - self.t
        if dt > 0:
            for f in self.f:
                f.predict(dt)
            self.t = t

    def cb(self, m):
        t, z = self.now(), [m.vector.x, m.vector.y, m.vector.z]
        if self.f is None or t - self.t_meas > self.coast:
            self.f = [CV1D(v, self.q, self.r) for v in z]
            self.t = t
        else:
            self.predict_to(t)
        for f, v in zip(self.f, z):
            f.update(v)
        self.t_meas = t

    def tick(self):
        if self.f is None:
            return
        t = self.now()
        lock = (t - self.t_meas) <= self.coast
        self.pub_lock.publish(Bool(data=bool(lock)))
        if not lock:
            return
        self.predict_to(t)
        out = Vector3Stamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.vector.x, out.vector.y, out.vector.z = (float(f.x[0]) for f in self.f)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GateTrackerPixel())


if __name__ == '__main__':
    main()
