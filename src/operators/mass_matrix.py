

import numpy as np
import scipy.sparse as sp

def lumped_mass_barycentric(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """
    Lumped (barycentric) mass matrix:
      M_ii = sum over incident faces of (area(face)/3).
    """
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)

    n = V.shape[0]
    m = np.zeros(n, dtype=np.float64)
    np.add.at(m, F[:, 0], area / 3.0)
    np.add.at(m, F[:, 1], area / 3.0)
    np.add.at(m, F[:, 2], area / 3.0)

    m = np.maximum(m, 1e-16) 
    return sp.diags(m, format="csr")

