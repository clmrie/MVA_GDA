# heat_method.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from operators.gradient import gradient_scalar_per_face
from operators.divergence import divergence_vertex_from_face_field


def _unique_edges(F: np.ndarray) -> np.ndarray:
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    return np.unique(E, axis=0)


def _typical_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    E = _unique_edges(F)
    return float(np.mean(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def _pin_solve(L: sp.spmatrix, b: np.ndarray, pin: int = 0) -> np.ndarray:
    Lp = L.tolil(copy=True)
    bp = b.copy()
    Lp[pin, :] = 0.0
    Lp[:, pin] = 0.0
    Lp[pin, pin] = 1.0
    bp[pin] = 0.0
    return spla.spsolve(Lp.tocsr(), bp)


def heat_geodesic_from_sources(
    mesh,
    source_ids,
    t: float | None = None,
    *,
    t_mult: float = 1.0,
    rhs_sign: int = -1,
    heat_rhs: str = "Mdelta",
    boundary_condition: str = "neumann"
):
    V, F = mesh.V, mesh.F
    n = V.shape[0]

    L = cotangent_laplacian(V, F)
    M = lumped_mass_barycentric(V, F)

    if t is None:
        h = _typical_edge_length(V, F)
        t = h * h
    t = float(t_mult) * float(t)

    if np.isscalar(source_ids):
        source_ids = [int(source_ids)]
    else:
        source_ids = [int(s) for s in source_ids]

    delta = np.zeros(n, dtype=np.float64)
    delta[np.array(source_ids, dtype=int)] = 1.0

    A_heat = (M + t * L).tocsr()

    if heat_rhs == "delta":
        b_heat = delta.copy()
    elif heat_rhs == "Mdelta":
        b_heat = M @ delta
    else:
        raise ValueError("heat_rhs must be 'delta' or 'Mdelta'")

    if boundary_condition == "dirichlet":
        edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
        boundary_edges = unique_edges[counts == 1]
        b_verts = np.unique(boundary_edges.flatten())

        if b_verts.size > 0:
            A_heat = A_heat.tolil()
            A_heat[b_verts, :] = 0.0
            A_heat[:, b_verts] = 0.0
            A_heat[b_verts, b_verts] = 1.0
            A_heat = A_heat.tocsr()
            b_heat[b_verts] = 0.0

    elif boundary_condition != "neumann":
        raise ValueError("boundary_condition must be 'neumann' or 'dirichlet'")

    u = spla.spsolve(A_heat, b_heat)

    grad_u = gradient_scalar_per_face(V, F, u)
    norm = np.maximum(np.linalg.norm(grad_u, axis=1, keepdims=True), 1e-16)
    X = -grad_u / norm

    div = divergence_vertex_from_face_field(V, F, X)
    rhs = rhs_sign * div

    pin = source_ids[0] if len(source_ids) > 0 else 0
    phi = _pin_solve(L, rhs, pin=pin)

    phi -= phi.min()
    phi = np.maximum(phi, 0.0)

    return phi, {
        "t": float(t),
        "L": L.tocsr(),
        "M": M.tocsr(),
        "heat_rhs": heat_rhs,
        "boundary_condition": boundary_condition,
        "t_mult": t_mult,
        "sources": source_ids,
        "u": u,
        "X": X,
        "grad_u": grad_u,
        "delta": delta,
        "div": div,
    }