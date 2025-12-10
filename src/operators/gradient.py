
import numpy as np

def per_face_grad_barycentric(V: np.ndarray, F: np.ndarray):
    """
    Returns:
      G : (m, 3, 3) where G[f, a, :] = ∇φ_a on face f (a in {0,1,2})
          expressed in R^3 (lies in the triangle plane).
      area : (m,) triangle areas
      dblA : (m,) = 2 * area = ||N|| where N is unnormalized normal
    Formula:
      ∇φ_a = (N × e_a) / ||N||^2
      where e_a is the edge opposite vertex a and N = (vj-vi) × (vk-vi).
    """
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    vi, vj, vk = V[i], V[j], V[k]

    N = np.cross(vj - vi, vk - vi)                 # (m,3)
    dblA = np.linalg.norm(N, axis=1)               # = 2*area
    denom = np.maximum(dblA, 1e-16)
    inv_denom2 = 1.0 / (denom ** 2)                # = 1/(||N||^2)

    # opposite edges
    e0 = vk - vj   # opposite vertex i
    e1 = vi - vk   # opposite vertex j
    e2 = vj - vi   # opposite vertex k

    g0 = np.cross(N, e0) * inv_denom2[:, None]     # ∇φ_i
    g1 = np.cross(N, e1) * inv_denom2[:, None]     # ∇φ_j
    g2 = np.cross(N, e2) * inv_denom2[:, None]     # ∇φ_k

    G = np.stack([g0, g1, g2], axis=1)             # (m,3,3)
    area = 0.5 * dblA
    return G, area, dblA

def gradient_scalar_per_face(V: np.ndarray, F: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute per-face gradient of a scalar vertex function u.
    Returns:
      grad_u : (m,3) vector constant on each face.
    """
    G, _, _ = per_face_grad_barycentric(V, F)      # (m,3,3)
    ui = u[F[:, 0]][:, None]
    uj = u[F[:, 1]][:, None]
    uk = u[F[:, 2]][:, None]
    grad_u = G[:, 0, :] * ui + G[:, 1, :] * uj + G[:, 2, :] * uk
    return grad_u
