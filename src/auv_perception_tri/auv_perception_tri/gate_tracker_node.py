#!/usr/bin/env python3
"""Static-landmark filter in the odom frame (shared by pnp + tri packages).

The gate does not move, so its odom position is a static KF state: each measurement
(from any frame; transformed to odom via TF) is fused with its own covariance, gated by
Mahalanobis distance, and the result is re-expressed in base_link at publish_rate using the
newest EKF TF -> the controller sees the gate move correctly relative to the AUV even when
detections drop out.
meas_cov_scale > 1 inflates measurement noise because successive measurements are
correlated (PnP systematic bias / overlapping triangulation windows)."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from auv_perception_tri.utils.geometry import nlerp, tf_to_Rt, transform_pose


class GateTracker(Node):
    def __init__(self):
        super().__init__('gate_tracker')
        for n, v in [('odom_frame', 'odom'), ('base_frame', 'base_link'), ('publish_rate', 30.0),
                     ('gate_chi2', 11.34), ('max_rejects', 10), ('process_noise', 1e-4),
                     ('orient_alpha', 0.2), ('use_orientation', True), ('lost_timeout_s', 3.0),
                     ('tf_timeout_s', 0.05), ('meas_cov_scale', 1.0)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        self.g = g
        self.p = self.P = self.q = self.t = None
        self.rejects = 0
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.pub_odom = self.create_publisher(PoseWithCovarianceStamped, 'gate/pose_odom', 10)
        self.pub_base = self.create_publisher(PoseStamped, 'gate/pose_base', 10)
        self.pub_lock = self.create_publisher(Bool, 'gate/lock', 10)
        self.create_subscription(PoseWithCovarianceStamped, 'gate/pose_meas', self.cb, 10)
        self.create_timer(1.0 / g('publish_rate'), self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cb(self, m):
        g = self.g
        pp, oo = m.pose.pose.position, m.pose.pose.orientation
        p, q = np.array([pp.x, pp.y, pp.z]), np.array([oo.x, oo.y, oo.z, oo.w])
        C = np.array(m.pose.covariance).reshape(6, 6)[:3, :3] * g('meas_cov_scale')
        if m.header.frame_id != g('odom_frame'):
            try:
                tf = self.tfb.lookup_transform(g('odom_frame'), m.header.frame_id, m.header.stamp,
                                               Duration(seconds=g('tf_timeout_s')))
            except TransformException as e:
                self.get_logger().warn(f'TF: {e}', throttle_duration_sec=2.0)
                return
            R, t = tf_to_Rt(tf.transform)
            p, q = transform_pose(R, t, p, q)
            C = R @ C @ R.T
        now = self.now()
        if self.p is None:
            self.p, self.P, self.q = p, C, q
        else:
            self.P = self.P + np.eye(3) * g('process_noise') * max(now - self.t, 0.0)
            y, S = p - self.p, self.P + C
            if y @ np.linalg.solve(S, y) > g('gate_chi2'):
                self.rejects += 1
                if self.rejects > g('max_rejects'):   # EKF jump or bad initial lock: restart
                    self.p, self.P, self.q, self.rejects = p, C, q, 0
                return
            K = self.P @ np.linalg.inv(S)
            self.p, self.P, self.rejects = self.p + K @ y, (np.eye(3) - K) @ self.P, 0
            if g('use_orientation'):
                self.q = nlerp(self.q, q, g('orient_alpha'))
        self.t = now
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = m.header.stamp, g('odom_frame')
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = (float(v) for v in self.p)
        o = out.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in self.q)
        c = np.zeros((6, 6))
        c[:3, :3] = self.P
        c[3, 3] = c[4, 4] = c[5, 5] = 0.05
        out.pose.covariance = c.ravel().tolist()
        self.pub_odom.publish(out)

    def tick(self):
        if self.p is None:
            return
        fresh = (self.now() - self.t) < self.g('lost_timeout_s')
        self.pub_lock.publish(Bool(data=bool(fresh)))
        if not fresh:
            return
        try:
            tf = self.tfb.lookup_transform(self.g('base_frame'), self.g('odom_frame'), Time())
        except TransformException:
            return
        R, t = tf_to_Rt(tf.transform)
        p, q = transform_pose(R, t, self.p, self.q)
        out = PoseStamped()
        out.header.stamp, out.header.frame_id = self.get_clock().now().to_msg(), self.g('base_frame')
        out.pose.position.x, out.pose.position.y, out.pose.position.z = (float(v) for v in p)
        o = out.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        self.pub_base.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GateTracker())


if __name__ == '__main__':
    main()
