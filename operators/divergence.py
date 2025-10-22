
# operators/divergence.py
import numpy as np
from operators.gradient import per_face_grad_barycentric

def divergence_vertex_from_face_field(V: np.ndarray, F: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Discrete divergence of a piecewise-constant vector field X (one 3D vector per face),
    returned as a scalar per vertex (FEM weak form).
      (div X)_i = - sum_{f incident to i} ∫_f X · ∇φ_i dA
                 = - sum_f area_f * (X_f · ∇φ_i|_f)
    """
    G, area, _ = per_face_grad_barycentric(V, F)  
    IG = G * area[:, None, None]                  
    contrib = -np.einsum("mab,mb->ma", IG, X)      

    n = V.shape[0]
    div = np.zeros(n, dtype=np.float64)
    np.add.at(div, F[:, 0], contrib[:, 0])
    np.add.at(div, F[:, 1], contrib[:, 1])
    np.add.at(div, F[:, 2], contrib[:, 2])
    return div

