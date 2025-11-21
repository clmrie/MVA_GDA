import numpy as np
import scipy.sparse as sp

def cotangent_laplacian(V: np.ndarray, F: np.ndarray, clamp: bool = True) -> sp.csr_matrix:
    """
    Symmetric cotangent Laplacian. 
    If clamp=True, negative weights are set to zero (Intrinsic Delaunay approximation).
    """
    n = V.shape[0]
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    vi, vj, vk = V[i], V[j], V[k]

    def cot(a, b):
        cross = np.cross(a, b)
        denom = np.linalg.norm(cross, axis=1)
        denom = np.maximum(denom, 1e-16)
        return (a * b).sum(axis=1) / denom

    cot0 = cot(vj - vi, vk - vi)
    cot1 = cot(vk - vj, vi - vj)
    cot2 = cot(vi - vk, vj - vk)

    w_jk = 0.5 * cot0
    w_ki = 0.5 * cot1
    w_ij = 0.5 * cot2

    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    data = np.concatenate([w_jk, w_jk, w_ki, w_ki, w_ij, w_ij])

    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    
    if clamp:
        W.data = np.maximum(W.data, 0.0)

    d = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(d, format="csr") - W
    return (L + L.T) * 0.5