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
    rhs_variant: str = "weak",
    rhs_sign: int = -1,         
    heat_rhs: str = "Mdelta",   
):
    """
    Heat Method for geodesic distance computation.

    Parameters
    ----------
    mesh: has .V (n,3) and .F (m,3)
    source_ids: int or list[int]
    t: base heat time; if None, uses h^2 with h = mean edge length
    t_mult: multiply chosen t by this factor
    rhs_variant: "weak" (integrated divergence) or "mass" (pointwise×mass)
    rhs_sign: typically -1 with our divergence convention
    heat_rhs:
        "delta"  -> solve (M + tL) u = δ
        "Mdelta" -> solve (M + tL) u = M δ   (recommended with stiffness L)

    Returns
    -------
    phi: (n,) geodesic distances
    info: dictionary with extra metadata
    """
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
        b_heat = delta
    elif heat_rhs == "Mdelta":
        b_heat = M @ delta
    else:
        raise ValueError("heat_rhs must be 'delta' or 'Mdelta'")
    u = spla.spsolve(A_heat, b_heat)

    grad_u = gradient_scalar_per_face(V, F, u) 
    norm = np.maximum(np.linalg.norm(grad_u, axis=1, keepdims=True), 1e-16)
    X = -grad_u / norm  

    div_int = divergence_vertex_from_face_field(V, F, X)
    if rhs_variant.lower() == "weak":
        rhs = div_int
    elif rhs_variant.lower() == "mass":
        A_vert = M.diagonal()
        div_point = div_int / np.maximum(A_vert, 1e-16)
        rhs = A_vert * div_point
    else:
        raise ValueError("rhs_variant must be 'weak' or 'mass'")
    rhs = rhs_sign * rhs

    phi = _pin_solve(L, rhs, pin=0)

    phi -= phi.min()
    phi = np.maximum(phi, 0.0)

    return phi, {
        "t": float(t),
        "L": L.tocsr(),
        "M": M.tocsr(),
        "rhs_variant": rhs_variant,
        "rhs_sign": rhs_sign,
        "t_mult": t_mult,
        "heat_rhs": heat_rhs,
    }
