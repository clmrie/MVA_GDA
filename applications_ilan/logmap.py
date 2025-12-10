import os
import sys
import numpy as np
import scipy.sparse.linalg as spla
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mesh import Mesh
from operators.mass_matrix import lumped_mass_barycentric
from operators.laplacian import cotangent_laplacian
from vector_heat import build_connection_laplacian, ambient_to_complex_per_vertex, vertex_frames_from_normals
from applications.r0 import initialize_R0

EPS = 1e-12


def _mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    return float(np.mean(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    return np.asarray(m.vertex_normals, dtype=float)


def _solve_vector_heat(Lc, M, rhs, t):
    A = M.astype(np.complex128) + t * Lc
    return spla.spsolve(A, rhs.astype(np.complex128))


def _normalize(u: np.ndarray) -> np.ndarray:
    mag = np.abs(u)
    mag = np.maximum(mag, EPS)
    return u / mag


def _transport_reference(V: np.ndarray, F: np.ndarray, t1: np.ndarray, t2: np.ndarray, source: int, ref_dir: np.ndarray, Lc, M, t):
    w0 = np.zeros((V.shape[0], 3), dtype=float)
    w0[source] = ref_dir
    u0 = ambient_to_complex_per_vertex(w0, t1, t2)
    rhs = M.dot(u0.astype(np.complex128))
    u = _solve_vector_heat(Lc, M, rhs, t)
    return _normalize(u)


def _radial_field(V: np.ndarray, F: np.ndarray, t1: np.ndarray, t2: np.ndarray, source: int, Lc, M, t):
    R0 = initialize_R0(V, F, source, t1, t2)
    u = _solve_vector_heat(Lc, M, R0, t)
    return _normalize(u)


def _complex_to_ambient(u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    xr = u.real.reshape(-1, 1)
    xi = u.imag.reshape(-1, 1)
    return xr * t1 + xi * t2


def _divergence_from_radial(V: np.ndarray, normals: np.ndarray, R: np.ndarray, t1: np.ndarray, t2: np.ndarray, L) -> np.ndarray:
    X = _complex_to_ambient(R, t1, t2)
    L = L.tocsr()
    indptr = L.indptr
    indices = L.indices
    data = L.data
    n = V.shape[0]
    div = np.zeros(n, dtype=float)
    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]
        vi = V[i]
        ni = normals[i]
        Xi = X[i]
        for p in range(start, end):
            j = indices[p]
            if j == i:
                continue
            w_ij = -float(data[p])
            vj = V[j]
            nj = normals[j]
            Xj = X[j]
            e = vj - vi
            e_i = e - np.dot(e, ni) * ni
            e_j = -e - np.dot(-e, nj) * nj
            Rij = 0.5 * (float(np.dot(e_i, Xi)) + float(np.dot(e_j, Xj)))
            div[i] += w_ij * Rij
    return div


def _pin_solve(L, rhs: np.ndarray, pin: int) -> np.ndarray:
    L = L.tolil()
    n = rhs.shape[0]
    L[pin, :] = 0.0
    L[:, pin] = 0.0
    L[pin, pin] = 1.0
    b = rhs.copy()
    b[pin] = 0.0
    L = L.tocsr()
    x = spla.spsolve(L, b)
    return np.asarray(x, dtype=float)


def compute_logmap(mesh: Mesh, source: int, ref_dir: np.ndarray | None = None, t: float | None = None, t_mult: float = 1.0):
    V, F = mesh.V, mesh.F

    normals = _vertex_normals(V, F)
    t1, t2 = vertex_frames_from_normals(V, normals)

    if ref_dir is None:
        ref_dir = t1[source]

    if t is None:
        h = _mean_edge_length(V, F)
        t = h * h
    t = float(t) * float(t_mult)

    Lc = build_connection_laplacian(V, F, normals, t1, t2)
    M = lumped_mass_barycentric(V, F)

    H = _transport_reference(V, F, t1, t2, source, ref_dir, Lc, M, t)
    R = _radial_field(V, F, t1, t2, source, Lc, M, t)
    R[source] = 0.0 + 0.0j

    phi = np.angle(R) - np.angle(H)
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    L_scalar = cotangent_laplacian(V, F)
    divR = _divergence_from_radial(V, normals, R, t1, t2, L_scalar)
    rhs = -divR
    r = _pin_solve(L_scalar, rhs, pin=source)
    r = r - r[source]
    r = np.maximum(r, 0.0)

    logmap = r * np.exp(1j * phi)
    logmap[source] = 0.0 + 0.0j

    return logmap
