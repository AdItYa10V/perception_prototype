#!/usr/bin/env python3
"""LEVEL 2 - single-frame PnP from gate keypoints (model_points = FBX dimensions).

Undistorts keypoints to the normalized plane and solves with K=I (works for fisheye too).
IPPE for planar gates returns both pose solutions; if they are near-ambiguous, the one
closest to the previous orientation wins (kills the classic flip at long range).
Publishes gate/pose_meas (PoseWithCovarianceStamped, camera optical frame)."""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo

from auv_perception_pnp.utils.camera_model import CameraModel
from auv_perception_pnp.utils.geometry import rot_angle


class GatePnP(Node):
    def __init__(self):
        super().__init__('gate_pose')
        for n, v in [('calib_file', ''), ('camera_frame', 'camera_optical_frame'),
                     ('model_points', [-1.525, -0.76, 0.0, 1.525, -0.76, 0.0,
                                       1.525, 0.76, 0.0, -1.525, 0.76, 0.0]),
                     ('keypoint_ids', [0, 1, 2, 3]), ('min_kpt_conf', 0.3), ('min_points', 4),
                     ('max_reproj_px', 6.0), ('min_range', 0.5), ('max_range', 15.0),
                     ('pos_sigma_frac', 0.05), ('orient_sigma', 0.2), ('ambiguity_ratio', 1.5)]:
            self.declare_parameter(n, v)
        g = lambda n: self.get_parameter(n).value
        self.cam = CameraModel.from_yaml(g('calib_file'))
        self.model = np.array(g('model_points'), float).reshape(-1, 3)
        self.ids = [int(i) for i in g('keypoint_ids')]
        self.last_R = None
        self.pub = self.create_publisher(PoseWithCovarianceStamped, 'gate/pose_meas', 10)
        self.create_subscription(CameraInfo, '/perception/camera_info', self.cb_info, 10)
        self.create_subscription(PolygonStamped, 'gate/keypoints', self.cb, 10)

    def cb_info(self, m):
        self.cam.set_image_size(m.width, m.height)

    def cb(self, msg):
        g = lambda n: self.get_parameter(n).value
        P = msg.polygon.points
        if len(P) < max(self.ids) + 1:
            return
        xy = np.array([[p.x, p.y] for p in P])
        cf = np.array([p.z for p in P])
        sel = [i for i in self.ids if cf[i] >= g('min_kpt_conf')]
        if len(sel) < g('min_points'):
            return
        obj, nrm = self.model[sel], self.cam.undistort_points(xy[sel]).reshape(-1, 1, 2)
        I3 = np.eye(3)
        planar = np.ptp(obj[:, 2]) < 1e-6
        n, rvs, tvs, errs = cv2.solvePnPGeneric(
            obj, nrm, I3, None, flags=cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_EPNP)
        if not n:
            return
        sols = sorted(zip(errs.ravel(), rvs, tvs), key=lambda s: s[0])
        best = sols[0]
        if len(sols) > 1 and self.last_R is not None and \
                sols[1][0] < g('ambiguity_ratio') * max(sols[0][0], 1e-9):
            best = min(sols[:2], key=lambda s: rot_angle(cv2.Rodrigues(s[1])[0], self.last_R))
        rv, tv = cv2.solvePnPRefineLM(obj, nrm, I3, None, best[1].copy(), best[2].copy())
        proj, _ = cv2.projectPoints(obj, rv, tv, I3, None)
        reproj_px = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - nrm.reshape(-1, 2)) ** 2, 1)))) * self.cam.fx
        rng = float(np.linalg.norm(tv))
        if tv[2] <= 0 or reproj_px > g('max_reproj_px') or not g('min_range') <= rng <= g('max_range'):
            return
        R = cv2.Rodrigues(rv)[0]
        self.last_R = R
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = msg.header.stamp, g('camera_frame')
        p = out.pose.pose
        p.position.x, p.position.y, p.position.z = (float(v) for v in tv.ravel())
        q = Rot.from_matrix(R).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
        s = g('pos_sigma_frac') * rng      # depth error grows ~ range; depth axis is the worst
        c = np.zeros((6, 6))
        c[0, 0], c[1, 1], c[2, 2] = s ** 2, s ** 2, (2 * s) ** 2
        c[3, 3] = c[4, 4] = c[5, 5] = g('orient_sigma') ** 2
        out.pose.covariance = c.ravel().tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GatePnP())


if __name__ == '__main__':
    main()
