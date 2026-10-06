#!/usr/bin/env python3
"""LEVEL 3 - multi-frame triangulation using the EKF pose (VN100 + BAR30 + ORB-SLAM3 fused).

Each frame: undistort gate keypoints -> unit rays, rotate/translate into `odom` with the TF at
the (latency-corrected) image stamp. A keyframe is stored only after the vehicle moved
min_baseline_m (pure rotation carries no depth). At solve_rate_hz every keypoint is
triangulated robustly; with >=3 keypoints a Kabsch fit to the FBX model gives the full gate
pose, and its residual is a free sanity check on the gate dimensions.
use_keypoints=false -> triangulates only the bbox centre (position only).
Publishes gate/pose_meas in the odom frame."""
import numpy as np
import rclpy
from collections import deque
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from auv_perception_tri.utils.camera_model import CameraModel
from auv_perception_tri.utils.geometry import kabsch, robust_triangulate, tf_to_Rt


class GateTriangulation(Node):
    def __init__(self):
        super().__init__('gate_pose')
        for n, v in [('calib_file', ''), ('camera_frame', 'camera_optical_frame'),
                     ('odom_frame', 'odom'), ('use_keypoints', True),
                     ('model_points', [-1.525, -0.76, 0.0, 1.525, -0.76, 0.0,
                                       1.525, 0.76, 0.0, -1.525, 0.76, 0.0]),
                     ('keypoint_ids', [0, 1, 2, 3]), ('min_kpt_conf', 0.3), ('min_score', 0.4),
                     ('min_baseline_m', 0.05), ('window', 80), ('max_age_s', 15.0),
                     ('min_obs', 6), ('min_parallax_deg', 4.0), ('resid_px', 6.0),
                     ('solve_rate_hz', 5.0), ('tf_timeout_s', 0.05), ('max_range', 20.0),
                     ('fit_tol_m', 0.25), ('orient_sigma', 0.15)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        self.g = g
        self.cam = CameraModel.from_yaml(g('calib_file'))
        self.use_kp = bool(g('use_keypoints'))
        self.ids = [int(i) for i in g('keypoint_ids')]
        self.model = np.array(g('model_points'), float).reshape(-1, 3)[self.ids]
        self.obs, self.dirty, self.last_stamp = deque(maxlen=int(g('window'))), False, None
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, 'gate/pose_meas', 10)
        self.create_subscription(CameraInfo, '/perception/camera_info', self.cb_info, 10)
        if self.use_kp:
            self.create_subscription(PolygonStamped, 'gate/keypoints', self.cb_kp, 10)
        else:
            self.create_subscription(Detection2DArray, 'gate/detections', self.cb_det, 10)
        self.create_timer(1.0 / g('solve_rate_hz'), self.solve)

    def cb_info(self, m):
        self.cam.set_image_size(m.width, m.height)

    def cb_kp(self, msg):
        P = msg.polygon.points
        if len(P) < max(self.ids) + 1:
            return
        xy = np.array([[P[i].x, P[i].y] for i in self.ids])
        ok = np.array([P[i].z for i in self.ids]) >= self.g('min_kpt_conf')
        self.ingest(msg.header.stamp, xy, ok)

    def cb_det(self, msg):
        ds = [d for d in msg.detections if d.results and d.results[0].hypothesis.score >= self.g('min_score')]
        if ds:
            d = max(ds, key=lambda d: d.results[0].hypothesis.score)
            self.ingest(msg.header.stamp, np.array([[d.bbox.center.position.x, d.bbox.center.position.y]]),
                        np.array([True]))

    def ingest(self, stamp, xy, ok):
        try:
            tf = self.tfb.lookup_transform(self.g('odom_frame'), self.g('camera_frame'), stamp,
                                           Duration(seconds=self.g('tf_timeout_s')))
        except TransformException as e:
            self.get_logger().warn(f'TF: {e}', throttle_duration_sec=2.0)
            return
        R, o = tf_to_Rt(tf.transform)
        if self.obs and np.linalg.norm(o - self.obs[-1]['o']) < self.g('min_baseline_m'):
            return
        self.obs.append(dict(t=stamp.sec + stamp.nanosec * 1e-9, o=o,
                             rays=self.cam.rays(xy) @ R.T, ok=ok))
        self.last_stamp, self.dirty = stamp, True

    def solve(self):
        g = self.g
        if not self.dirty or len(self.obs) < g('min_obs'):
            return
        self.dirty = False
        t_new = self.obs[-1]['t']
        obs = [o for o in self.obs if t_new - o['t'] <= g('max_age_s')]
        ang_thr = g('resid_px') / self.cam.fx
        pts, covs = [], []
        for k in range(len(obs[-1]['rays'])):
            sel = [o for o in obs if o['ok'][k]]
            if len(sel) < g('min_obs'):
                return
            res = robust_triangulate(np.array([o['o'] for o in sel]),
                                     np.array([o['rays'][k] for o in sel]),
                                     ang_thr, g('min_parallax_deg'), g('min_obs'), sigma=ang_thr / 3)
            if res is None:
                return
            pts.append(res[0])
            covs.append(res[1])
        pts = np.array(pts)
        if self.use_kp and len(pts) >= 3:
            R, t, rms = kabsch(self.model, pts)
            if rms > g('fit_tol_m'):
                self.get_logger().warn(f'model fit rms {rms:.2f} m > tol (bad keypoints / wrong model_points)',
                                       throttle_duration_sec=2.0)
                return
            q = Rot.from_matrix(R).as_quat()
        else:
            t, q = pts.mean(0), np.array([0.0, 0.0, 0.0, 1.0])
        if np.linalg.norm(t - obs[-1]['o']) > g('max_range'):
            return
        C = sum(covs) / len(covs) ** 2
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = self.last_stamp, g('odom_frame')
        p = out.pose.pose
        p.position.x, p.position.y, p.position.z = (float(v) for v in t)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
        c = np.zeros((6, 6))
        c[:3, :3] = C
        c[3, 3] = c[4, 4] = c[5, 5] = g('orient_sigma') ** 2 if (self.use_kp and len(pts) >= 3) else 1e3
        out.pose.covariance = c.ravel().tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GateTriangulation())


if __name__ == '__main__':
    main()
