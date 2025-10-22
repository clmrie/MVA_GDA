
import numpy as np
import scipy.sparse as sp

def cotangent_laplacian(V: np.ndarray, F: np.ndarray, clamp: bool = True, eps: float = 1e-12) -> sp.csr_matrix:
    """
    Symmetric cotangent Laplacian (negative semidefinite).
    Off-diagonal: L_ij = -w_ij,  Diagonal: L_ii = sum_j w_ij,
    with w_ij = 0.5*(cot alpha + cot beta) accumulated over incident faces.
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
        W.data[W.data < 0] = np.maximum(W.data[W.data < 0], -eps)

    d = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(d, format="csr") - W
    return (L + L.T) * 0.5
