"""Camera model: calibrated at one resolution, usable at any scaled resolution.

Points are undistorted to the *normalized image plane* (x/z, y/z), so every
downstream solver (PnP, triangulation) works with K = I and no distortion.
This handles both pinhole+radial (action cam in 'Linear' FOV mode) and
equidistant fisheye (action cam in 'Wide'/'SuperView').
"""
import cv2
import numpy as np
import yaml


class CameraModel:
    def __init__(self, K, D, size, model='pinhole'):
        self.K0 = np.array(K, float).reshape(3, 3)
        self.model = model
        D = np.array(D, float).reshape(-1)
        self.D = D[:4].reshape(4, 1) if model == 'fisheye' else D.reshape(-1, 1)
        self.base_size = (int(size[0]), int(size[1]))
        self.set_image_size(*self.base_size)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(d['K'], d['D'], (d['width'], d['height']), d.get('model', 'pinhole'))

    def set_image_size(self, w, h):
        """Rescale K when the stream is downscaled (enhancement node does 0.5x)."""
        K = self.K0.copy()
        K[0, :] *= w / self.base_size[0]
        K[1, :] *= h / self.base_size[1]
        self.K, self.size = K, (int(w), int(h))

    @property
    def fx(self):
        return float(self.K[0, 0])

    def undistort_points(self, pts):
        """Nx2 pixels -> Nx2 normalized coordinates."""
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if self.model == 'fisheye':
            out = cv2.fisheye.undistortPoints(p, self.K, self.D)
        else:
            out = cv2.undistortPoints(p, self.K, self.D)
        return out.reshape(-1, 2)

    def rays(self, pts):
        """Nx2 pixels -> Nx3 unit rays in the camera optical frame (z fwd)."""
        n = self.undistort_points(pts)
        r = np.hstack([n, np.ones((len(n), 1))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)
