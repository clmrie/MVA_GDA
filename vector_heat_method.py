"""Vector Heat Method implementation (patched).

Implements the algorithm of
Nicholas Sharp, Yousuf Soliman, and Keenan Crane (TOG 2019)
with:
  • minimal-rotation transport between tangent planes (connection Laplacian)
  • Appendix-A initial conditions for the radial field
  • optional clamping of negative cotan weights for consistency on non-Delaunay meshes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Project-local operators (expected in your repo)
from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from operators.gradient import per_face_grad_barycentric  # kept for parity; not required here
from operators.divergence import divergence_vertex_from_face_field


# -----------------------------
# Small geometry / linear algebra helpers
# -----------------------------

def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    nV = V.shape[0]
    nF = F.shape[0]
    tri_n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    tri_n = _normalize(tri_n)
    N = np.zeros((nV, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(N, F[:, k], tri_n)
    return _normalize(N)


def _build_tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangent frames (t1, t2) at each vertex from its normal."""
    nV = normals.shape[0]
    t1 = np.zeros_like(normals)
    t2 = np.zeros_like(normals)
    for i in range(nV):
        n = normals[i]
        # pick a helper axis far from normal
        axis = np.array([0.0, 0.0, 1.0])
        if abs(n[2]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, axis)
        ln = np.linalg.norm(e1)
        if ln < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
            e1 = np.cross(n, axis)
            ln = np.linalg.norm(e1)
        e1 = e1 / max(ln, 1e-15)
        e2 = np.cross(n, e1)
        t1[i] = e1
        t2[i] = e2
    return t1, t2


def _complex_from_vector(v: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> complex:
    """Project 3D vector v into the (t1,t2) basis and return as complex a+ib."""
    return complex(np.dot(v, t1), np.dot(v, t2))


def _vector_from_complex(z: complex, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Lift complex z = a+ib living in tangent coords into 3D vector a*t1 + b*t2."""
    return (z.real * t1) + (z.imag * t2)


def _rodrigues_rotation(n_src: np.ndarray, n_dst: np.ndarray) -> np.ndarray:
    """
    Smallest 3×3 rotation taking unit vector n_src to n_dst.
    Robust to the parallel/antiparallel cases.
    """
    c = float(np.clip(np.dot(n_src, n_dst), -1.0, 1.0))
    if c > 1.0 - 1e-15:
        return np.eye(3)
    axis = np.cross(n_src, n_dst)
    s = float(np.linalg.norm(axis))
    if s < 1e-15:
        # opposite normals: rotate 180° about any axis orthogonal to n_src
        a = np.array([1.0, 0.0, 0.0]) if abs(n_src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(n_src, a)
        axis = axis / max(np.linalg.norm(axis), 1e-15)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)  # since sin(pi)=0, (1-cos(pi))=2
    axis = axis / s
    angle = np.arctan2(s, c)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rot2_from_frames(t1_i, t2_i, n_i, t1_j, t2_j, n_j) -> np.ndarray:
    """
    2×2 block mapping tangent coeffs at i to tangent coeffs at j
    via smallest-rotation parallel transport between normals.
    """
    R = _rodrigues_rotation(n_i, n_j)  # 3x3
    Rt1 = R @ t1_i
    Rt2 = R @ t2_i
    a11 = float(np.dot(Rt1, t1_j)); a12 = float(np.dot(Rt1, t2_j))
    a21 = float(np.dot(Rt2, t1_j)); a22 = float(np.dot(Rt2, t2_j))
    return np.array([[a11, a12],
                     [a21, a22]], dtype=np.float64)


# -----------------------------
# Discrete operators
# -----------------------------

@dataclass
class VectorHeatOperators:
    """Cached geometry and operators for the vector heat method."""
    frames_t1: np.ndarray            # (n,3)
    frames_t2: np.ndarray            # (n,3)
    vertex_normals: np.ndarray       # (n,3)
    L_conn: sp.csr_matrix            # (2n,2n) connection Laplacian (block)
    M_vec: sp.csr_matrix             # (2n,2n) vector mass = kron(M_scalar, I2)


def _clamped_cotan(L: sp.spmatrix) -> sp.csr_matrix:
    """
    Clamp negative cot weights to zero and rebuild a symmetric PSD cotan Laplacian.
    Ensures consistency between off-diagonals and diagonals.
    """
    C = L.tocoo()
    n = L.shape[0]
    rows, cols, data = [], [], []
    accum = np.zeros(n, dtype=np.float64)
    for i, j, v in zip(C.row, C.col, C.data):
        if i == j:
            continue
        w = max(0.0, -float(v))   # note: v is typically <=0 for off-diagonals
        if w == 0.0:
            continue
        rows.append(i); cols.append(j); data.append(-w)
        accum[i] += w
    # diagonals
    rows.extend(range(n)); cols.extend(range(n)); data.extend(accum.tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def precompute_vector_heat_operators(
    V: np.ndarray, F: np.ndarray, M_scalar: sp.spmatrix, L_scalar: sp.spmatrix, *,
    use_clamped: bool = True
) -> VectorHeatOperators:
    """
    Build tangent frames, the connection Laplacian (2n×2n block matrix), and the
    vector mass matrix (kron of scalar mass with I2). Transport uses minimal
    rotation between tangent planes.
    """
    n = V.shape[0]
    normals = _vertex_normals(V, F)
    t1, t2 = _build_tangent_frames(normals)

    # Off-diagonal weights from (clamped) scalar cotan matrix to keep operators consistent
    L_used = _clamped_cotan(L_scalar) if use_clamped else L_scalar.tocsr()
    Lcoo = L_used.tocoo()

    # Build blocks
    diag_blocks = np.zeros((n, 2, 2), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(ii: int, jj: int, block: np.ndarray) -> None:
        base_i = 2 * ii
        base_j = 2 * jj
        rows.extend([base_i, base_i, base_i + 1, base_i + 1])
        cols.extend([base_j, base_j + 1, base_j, base_j + 1])
        data.extend(block.reshape(-1).tolist())

    for i, j, v in zip(Lcoo.row, Lcoo.col, Lcoo.data):
        if i >= j:
            continue
        # We interpret L_used as -Δ, so off-diagonals are ≤ 0 and weights are w = -v ≥ 0
        w = max(0.0, -float(v))
        if w == 0.0:
            continue

        rot_ij = _rot2_from_frames(t1[i], t2[i], normals[i], t1[j], t2[j], normals[j])
        rot_ji = rot_ij.T  # inverse to numerical tolerance

        diag_blocks[i] += w * np.eye(2)
        diag_blocks[j] += w * np.eye(2)
        add_block(i, j, -w * rot_ij)
        add_block(j, i, -w * rot_ji)

    # Add diagonal blocks
    for i in range(n):
        add_block(i, i, diag_blocks[i])

    L_conn = sp.csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))

    # Vector mass: kron(M_scalar, I2)
    M_vec = sp.kron(M_scalar, sp.identity(2, format="csr"), format="csr")

    return VectorHeatOperators(
        frames_t1=t1, frames_t2=t2, vertex_normals=normals, L_conn=L_conn, M_vec=M_vec
    )


# -----------------------------
# Appendix-A initializer (radial field RHS)
# -----------------------------

def _edge_adjacent_vertices(F: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map directed edge (i,j) to list of opposite vertices [k] or [k,l] (one or two triangles)."""
    adj: dict[tuple[int, int], list[int]] = {}
    for a, b, c in F.astype(int):
        adj.setdefault((a, b), []).append(c)
        adj.setdefault((b, c), []).append(a)
        adj.setdefault((c, a), []).append(b)
        adj.setdefault((b, a), []).append(c)
        adj.setdefault((c, b), []).append(a)
        adj.setdefault((a, c), []).append(b)
    return adj


def _corner_angle(a, b, c) -> float:
    """Angle at a in triangle (a,b,c)."""
    u = b - a; v = c - a
    nu = u / max(np.linalg.norm(u), 1e-15)
    nv = v / max(np.linalg.norm(v), 1e-15)
    d = float(np.clip(np.dot(nu, nv), -1.0, 1.0))
    return float(np.arccos(d))


def _face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)
    return np.maximum(A, 1e-16)


def _initial_vector_field(
    V: np.ndarray,
    F: np.ndarray,
    ops: VectorHeatOperators,
    L_scalar: sp.spmatrix,     # unused, kept for signature parity
    sources: Sequence[int],
) -> np.ndarray:
    """
    Appendix A initial conditions for the radial field:
    Build x in C^{|V|} via per-edge/triangle formulas, rotate to vertex frames,
    and return as R^2 per-vertex array (n,2).
    """
    n = V.shape[0]
    if not sources:
        return np.zeros((n, 2), dtype=np.float64)

    A = _face_areas(V, F)
    adj = _edge_adjacent_vertices(F)

    # Helper: locate triangle index by its (unordered) vertex set
    tri_key = {tuple(sorted(map(int, F[k]))): k for k in range(F.shape[0])}

    x_complex = np.zeros(n, dtype=np.complex128)

    for i in sources:
        # neighbors of i: any vertex j that co-appears in a face with i
        neigh = np.unique(F[np.any(F == i, axis=1)])
        for j in neigh:
            j = int(j)
            if j == i:
                continue
            opp = adj.get((i, j), [])
            if len(opp) == 0:
                continue

            contrib = 0.0 + 0.0j

            # Triangle i-j-k
            k = int(opp[0])
            t_idx = tri_key.get(tuple(sorted((i, j, k))))
            if t_idx is not None:
                alpha = _corner_angle(V[i], V[j], V[k])  # θ^i_{jk}
                lik = np.linalg.norm(V[k] - V[i])
                term1 = (lik / (4.0 * A[t_idx])) * np.array([
                    alpha * np.sin(alpha),
                    np.sin(alpha) - alpha * np.cos(alpha)
                ])

                # rotate -e^{i φ_ji}
                edge = V[i] - V[j]
                edge -= np.dot(edge, ops.vertex_normals[j]) * ops.vertex_normals[j]
                z = _complex_from_vector(edge, ops.frames_t1[j], ops.frames_t2[j])
                rot = -z / abs(z) if abs(z) > 1e-16 else 1.0 + 0.0j
                contrib += rot * complex(term1[0], term1[1])

            # Triangle j-i-l (second incident tri, if any)
            if len(opp) >= 2:
                l = int(opp[1])
                t2_idx = tri_key.get(tuple(sorted((j, i, l))))
                if t2_idx is not None:
                    beta = _corner_angle(V[i], V[l], V[j])  # θ^i_{lj}
                    lil = np.linalg.norm(V[l] - V[i])
                    term2 = (lil / (4.0 * A[t2_idx])) * np.array([
                        beta * np.sin(beta),
                        beta * np.cos(beta) - np.sin(beta)
                    ])

                    edge = V[i] - V[j]
                    edge -= np.dot(edge, ops.vertex_normals[j]) * ops.vertex_normals[j]
                    z = _complex_from_vector(edge, ops.frames_t1[j], ops.frames_t2[j])
                    rot = -z / abs(z) if abs(z) > 1e-16 else 1.0 + 0.0j
                    contrib += rot * complex(term2[0], term2[1])

            x_complex[j] += contrib

        # Source value x_i (sum over incident triangles) — Eq. (17)
        acc = 0.0 + 0.0j
        t_ids = np.where(np.any(F == i, axis=1))[0]
        for t in t_ids:
            a, b, c = map(int, F[t])
            if a == i:
                j, k = b, c
            elif b == i:
                j, k = c, a
            else:
                j, k = a, b
            alpha = _corner_angle(V[i], V[j], V[k])  # θ^i_{jk}
            lij = np.linalg.norm(V[j] - V[i])
            lik = np.linalg.norm(V[k] - V[i])

            vec = np.array([
                -np.sin(alpha) * (lik * alpha + lij * np.sin(alpha)),
                lij * (np.cos(alpha) * np.sin(alpha) - alpha)
                + lik * (alpha * np.cos(alpha) - np.sin(alpha))
            ]) / (4.0 * A[t])

            edge = V[j] - V[i]
            edge -= np.dot(edge, ops.vertex_normals[i]) * ops.vertex_normals[i]
            z = _complex_from_vector(edge, ops.frames_t1[i], ops.frames_t2[i])
            rot = (z / abs(z)) if abs(z) > 1e-16 else (1.0 + 0.0j)
            acc += rot * complex(vec[0], vec[1])

        x_complex[i] += acc

    X0 = np.zeros((n, 2), dtype=np.float64)
    X0[:, 0] = x_complex.real
    X0[:, 1] = x_complex.imag
    return X0


# -----------------------------
# Solvers
# -----------------------------

def _pin_solve(A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b when A can have a nullspace of one constant per connected component.
    We detect components from A's sparsity (off-diagonal structure) and pin one vertex
    in each component (Dirichlet x[p]=0). Works for 1D or multi-RHS b.
    """
    A = A.tocsr()
    n = A.shape[0]

    # Build an undirected adjacency from off-diagonals
    Acoo = A.tocoo()
    mask = Acoo.row != Acoo.col
    rows = Acoo.row[mask]
    cols = Acoo.col[mask]
    data = np.ones_like(rows, dtype=np.float64)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)

    # Connected components
    from scipy.sparse.csgraph import connected_components
    n_comp, labels = connected_components(G, directed=False)

    # Choose one pin per component
    pins = np.zeros(n_comp, dtype=int)
    for c in range(n_comp):
        pins[c] = int(np.where(labels == c)[0][0])

    # Build mask of rows/cols to keep
    keep = np.ones(n, dtype=bool)
    keep[pins] = False

    A_red = A[keep][:, keep]

    if b.ndim == 1:
        b_red = b[keep]
        x_red = spla.spsolve(A_red.tocsc(), b_red)
        x = np.zeros(n, dtype=b.dtype)
        x[keep] = x_red
        # pins remain 0
    else:
        b_red = b[keep, :]
        x_red = spla.spsolve(A_red.tocsc(), b_red)
        x = np.zeros((n, b.shape[1]), dtype=b.dtype)
        x[keep, :] = x_red
        # pins rows stay 0
    return x



def _per_face_from_vertex_field(
    Y_vert: np.ndarray,
    t1: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    """
    Convert per-vertex tangent vectors (n,2) into a constant per-face 3D vector (m,3)
    by averaging lifted vectors from the three vertices.
    """
    n = t1.shape[0]
    Y3 = (Y_vert[:, :1] * t1) + (Y_vert[:, 1:2] * t2)  # (n,3)
    return Y3


# -----------------------------
# Public API
# -----------------------------

def vector_heat_geodesic(
    mesh_or_VF,
    sources: Iterable[int] | int,
    *,
    t: float | None = None,
    t_mult: float = 1.0,
    rhs_sign: float = 1.0,
    use_clamped: bool = True,
):
    """
    Compute geodesic distances from 'sources' using the Vector Heat Method.

    Parameters
    ----------
    mesh_or_VF : Mesh-like or tuple(V,F)
        Either an object with attributes .V (n,3), .F (m,3), or a (V,F) tuple.
    sources : int or Iterable[int]
        Source vertex index or list/array of indices.
    t : float or None
        Heat time. If None, use mean edge length squared times t_mult.
    t_mult : float
        Multiplier for the default t heuristic.
    rhs_sign : float
        +1 for Δφ = div(R); with L ≈ -Δ this means L φ = -div(R) * rhs_sign.
    use_clamped : bool
        If True, clamp negative cot weights for both scalar and vector operators.

    Returns
    -------
    phi : (n,) ndarray
        Geodesic distance (shifted so min=0).
    info : dict
        Diagnostic information: t, operators, M_scalar, L_scalar, rhs_sign, sources.
    """
    # Unpack mesh
    if isinstance(mesh_or_VF, tuple):
        V, F = mesh_or_VF
    else:
        V, F = mesh_or_VF.V, mesh_or_VF.F
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)

    n = V.shape[0]

    # Scalar operators
    L_scalar_raw = cotangent_laplacian(V, F)          # ≈ -Δ
    L_scalar = _clamped_cotan(L_scalar_raw) if use_clamped else L_scalar_raw.tocsr()
    M_scalar = lumped_mass_barycentric(V, F).tocsr()

    # Default heat time if not provided: mean edge length^2 * t_mult
    if t is None:
        # Fast rough mean edge length
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = V[edges[:, 0]] - V[edges[:, 1]]
        h2 = np.mean(np.sum(e * e, axis=1))
        t = float(t_mult * h2)

    # Vector operators (connection Laplacian + vector mass)
    ops = precompute_vector_heat_operators(V, F, M_scalar, L_scalar, use_clamped=use_clamped)

    # Build Appendix-A RHS (per-vertex tangent vectors)
    if isinstance(sources, int):
        sources = [int(sources)]
    else:
        sources = [int(s) for s in sources]

    X0 = _initial_vector_field(V, F, ops, L_scalar, sources)  # (n,2)

    # Backward Euler: (M + t L_conn) Y = M X0
    M_vec = ops.M_vec
    A = (M_vec + t * ops.L_conn).tocsr()

    # RHS
    rhs_vec = M_vec @ np.hstack([X0[:, 0], X0[:, 1]])

    # Solve block system
    Y = spla.spsolve(A, rhs_vec)
    Y = Y.reshape(-1, 2)

    # Normalize to get unit radial field in tangent coords -----------------------
    Y_norm = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-15)
    # enforce outward orientation once (your current distances are inverted)
    R_vert = -Y / Y_norm

    # Per-face constant 3D field (average), projected to triangle planes ---------
    R3_vert = (R_vert[:, :1] * ops.frames_t1) + (R_vert[:, 1:2] * ops.frames_t2)  # (n,3)
    X_face = (R3_vert[F[:, 0]] + R3_vert[F[:, 1]] + R3_vert[F[:, 2]]) / 3.0
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    Nf = np.cross(vj - vi, vk - vi)
    nf = Nf / np.maximum(np.linalg.norm(Nf, axis=1, keepdims=True), 1e-16)
    X_face = X_face - (np.sum(X_face * nf, axis=1, keepdims=True)) * nf

    # Weak-form divergence (minus & area already included) -----------------------
    div = divergence_vertex_from_face_field(V, F, X_face)  # = - G^T A X  :contentReference[oaicite:3]{index=3}

    # Poisson with L ≈ -Δ :  L φ = -div   (NO mass multiply) --------------------
    rhs = -div
    phi = _pin_solve(L_scalar, rhs)

    # Final shift to start at 0; do NOT clamp -----------------------------------
    phi -= float(phi.min())

    # Optional sanity to see what we return
    print("[VHM] internal phi range:", float(phi.min()), float(phi.max()))











    return phi, {
        "t": float(t),
        "operators": ops,
        "M_scalar": M_scalar,
        "L_scalar": L_scalar,
        "rhs_sign": rhs_sign,
        "sources": sources,
    }
