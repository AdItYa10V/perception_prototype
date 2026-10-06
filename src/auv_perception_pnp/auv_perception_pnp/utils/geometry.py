import numpy as np
from scipy.spatial.transform import Rotation as Rot


def triangulate_rays(O, D, sigma=1e-3):
    """Least-squares intersection of rays (origins O Nx3, unit dirs D Nx3).
    The point is an unweighted solve (stays robust to a biased start); the returned 3x3
    covariance weights each ray by its lateral noise (range*sigma)^2.
    Returns (None, None) if the rays are ~parallel."""
    P = np.eye(3)[None] - D[:, :, None] * D[:, None, :]
    A = P.sum(0)
    if np.linalg.cond(A) > 1e8:
        return None, None
    x = np.linalg.solve(A, np.einsum('ijk,ik->j', P, O))
    r = np.maximum(np.linalg.norm(x - O, axis=1), 0.3)
    Aw = np.einsum('i,ijk->jk', 1.0 / (r * sigma) ** 2, P)
    return x, np.linalg.inv(Aw)


def _two_ray(o1, d1, o2, d2):
    """Midpoint of closest approach of two rays; None if parallel or behind a camera."""
    w = o1 - o2
    b, d, e = d1 @ d2, d1 @ w, d2 @ w
    den = 1.0 - b * b
    if den < 1e-7:
        return None
    s = (b * e - d) / den
    t = e + s * b
    if s <= 0 or t <= 0:
        return None
    return 0.5 * ((o1 + s * d1) + (o2 + t * d2))


def robust_triangulate(O, D, ang_thresh, min_parallax_deg, min_inliers, sigma=1e-3,
                       iters=40, seed=0):
    """RANSAC (two-ray seeds, angular residual) + least-squares refinement on the inliers.
    Plain LS is NOT robust here: any point near the camera path has small point-to-line
    distance to every ray, so a few gross outliers drag the estimate home.
    Returns (x, cov, inlier_mask) or None."""
    n = len(O)
    if n < max(min_inliers, 2):
        return None
    rng = np.random.default_rng(seed)

    def resid(x):
        v = x - O
        r = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        return np.arccos(np.clip(np.einsum('ij,ij->i', v, D) / r, -1, 1))

    best, cnt = None, 0
    half = n // 2
    for _ in range(iters):
        i, j = int(rng.integers(0, max(half, 1))), int(rng.integers(half, n))   # favour baseline
        x = _two_ray(O[i], D[i], O[j], D[j])
        if x is None:
            continue
        m = resid(x) < ang_thresh
        if m.sum() > cnt:
            best, cnt = m, int(m.sum())
    if best is None or cnt < min_inliers:
        return None
    mask = best
    for _ in range(3):
        x, _ = triangulate_rays(O[mask], D[mask], sigma)
        if x is None:
            return None
        new = resid(x) < ang_thresh
        if new.sum() < min_inliers:
            return None
        if np.array_equal(new, mask):
            break
        mask = new
    x, cov = triangulate_rays(O[mask], D[mask], sigma)
    if x is None:
        return None
    Dm = D[mask]
    if np.degrees(np.arccos(np.clip(np.min(Dm @ Dm.T), -1, 1))) < min_parallax_deg:
        return None
    return x, cov, mask


def kabsch(model, pts):
    """Rigid fit pts ~= R @ model + t (no scale). Returns R, t, rms residual."""
    mc, pc = model.mean(0), pts.mean(0)
    H = (model - mc).T @ (pts - pc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = pc - R @ mc
    rms = float(np.sqrt(np.mean(np.sum((pts - (model @ R.T + t)) ** 2, axis=1))))
    return R, t, rms


def rot_angle(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def tf_to_Rt(transform):
    """geometry_msgs/Transform -> (3x3 R, 3 t)."""
    q, t = transform.rotation, transform.translation
    return Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([t.x, t.y, t.z])


def transform_pose(R, t, p, q):
    """Apply rigid transform (R,t) to pose (p, quat xyzw)."""
    return R @ p + t, (Rot.from_matrix(R) * Rot.from_quat(q)).as_quat()


def nlerp(q0, q1, a):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)
