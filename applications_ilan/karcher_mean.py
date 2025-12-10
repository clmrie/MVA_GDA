import os
import sys
import numpy as np
from mesh import Mesh
from applications.logmap import compute_logmap
from heat_method import heat_geodesic_from_sources

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def karcher_mean(mesh: Mesh, targets: np.ndarray, tau: float = 0.5, max_iters: int = 20, tol: float = 1e-4):
    targets = np.asarray(targets, dtype=int)
    if targets.size == 0:
        raise ValueError("targets is empty")
    m = int(targets[0])
    V = mesh.V
    bbox_diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    ang_weight = 0.1

    for _ in range(max_iters):
        logmap = compute_logmap(mesh, source=m, ref_dir=None, t=None, t_mult=1.0)
        avg = np.mean(logmap[targets])
        if abs(avg) < tol:
            break
        step_len = tau * abs(avg)
        step_dir = np.angle(avg)

        dist, _ = heat_geodesic_from_sources(mesh, m, t=None, t_mult=1.0)
        angles = np.angle(logmap)
        ang_diff = _wrap_angle(angles - step_dir)
        obj = (dist - step_len) ** 2 + ang_weight * ang_diff ** 2 * (bbox_diag ** 0)
        m = int(np.argmin(obj))
    return m
