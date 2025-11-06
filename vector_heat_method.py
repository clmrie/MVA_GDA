"""Vector Heat Method implementation.

This module implements the algorithm of Nicholas Sharp, Yousuf Soliman,
and Keenan Crane, "The Vector Heat Method" (ACM TOG 2019), to recover
geodesic distances on triangle meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from operators.gradient import per_face_grad_barycentric
from operators.divergence import divergence_vertex_from_face_field


@dataclass
class VectorHeatOperators:
    """Precomputed operators required by the vector heat method."""

    connection_laplacian: sp.csr_matrix
    mass: sp.csr_matrix
    frames_t1: np.ndarray  # (n, 3)
    frames_t2: np.ndarray  # (n, 3)


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    face_normals = np.cross(vj - vi, vk - vi)
    normals = np.zeros_like(V)
    np.add.at(normals, F[:, 0], face_normals)
    np.add.at(normals, F[:, 1], face_normals)
    np.add.at(normals, F[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-16)
    return normals / norms


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Choose helper axis far from the normal to avoid degeneracy.
    axis = np.array([0.0, 0.0, 1.0])
    if abs(n[2]) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, axis)
    norm = np.linalg.norm(t1)
    if norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n, axis)
        norm = np.linalg.norm(t1)
        if norm < 1e-8:
            raise ValueError("Failed to build tangent frame for vertex normal.")
    t1 /= norm
    t2 = np.cross(n, t1)
    return t1, t2


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    ax = axis / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = ax
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    outer = np.outer(ax, ax)
    return c * np.eye(3) + s * K + (1.0 - c) * outer


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
        axis_norm = 1.0
    axis = axis / axis_norm
    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    sin_theta = np.dot(axis, np.cross(a, b))
    return math.atan2(sin_theta, cos_theta)


def _transport_matrix(
    i: int,
    j: int,
    V: np.ndarray,
    normals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    edge = V[j] - V[i]
    edge_norm = np.linalg.norm(edge)
    if edge_norm < 1e-12:
        return np.eye(2)
    axis = edge / edge_norm
    angle = _signed_angle(normals[j], normals[i], axis)
    R = _rotation_matrix(axis, angle)

    basis_j = np.column_stack((t1[j], t2[j]))
    basis_i = np.column_stack((t1[i], t2[i]))

    transported = R @ basis_j
    # Project transported basis onto face i frame
    proj = np.empty((2, 2))
    proj[:, 0] = np.array([np.dot(basis_i[:, 0], transported[:, 0]), np.dot(basis_i[:, 1], transported[:, 0])])
    proj[:, 1] = np.array([np.dot(basis_i[:, 0], transported[:, 1]), np.dot(basis_i[:, 1], transported[:, 1])])
    return proj


def precompute_vector_heat_operators(V: np.ndarray, F: np.ndarray) -> VectorHeatOperators:
    n = V.shape[0]
    normals = _vertex_normals(V, F)
    frames_t1 = np.zeros_like(V)
    frames_t2 = np.zeros_like(V)
    for idx, nrm in enumerate(normals):
        t1, t2 = _orthonormal_basis(nrm)
        frames_t1[idx] = t1
        frames_t2[idx] = t2

    L = cotangent_laplacian(V, F)
    Lcoo = L.tocoo()

    diag_blocks = np.zeros((n, 2, 2), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(ii: int, jj: int, block: np.ndarray) -> None:
        base_i = 2 * ii
        base_j = 2 * jj
        for a in range(2):
            for b in range(2):
                rows.append(base_i + a)
                cols.append(base_j + b)
                data.append(block[a, b])

    for i, j, val in zip(Lcoo.row, Lcoo.col, Lcoo.data):
        if i >= j:
            continue
        if val >= 0:
            continue
        weight = -val
        P_ij = _transport_matrix(i, j, V, normals, frames_t1, frames_t2)
        P_ji = _transport_matrix(j, i, V, normals, frames_t1, frames_t2)
        diag_blocks[i] += weight * np.eye(2)
        diag_blocks[j] += weight * np.eye(2)
        add_block(i, j, -weight * P_ij)
        add_block(j, i, -weight * P_ji)

    for idx in range(n):
        add_block(idx, idx, diag_blocks[idx])

    connection_L = sp.coo_matrix((data, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()

    M = lumped_mass_barycentric(V, F)
    mass = sp.kron(M, sp.eye(2), format="csr")

    return VectorHeatOperators(
        connection_laplacian=connection_L,
        mass=mass,
        frames_t1=frames_t1,
        frames_t2=frames_t2,
    )


def _pin_solve(L: sp.spmatrix, b: np.ndarray, pin: int = 0) -> np.ndarray:
    Lp = L.tolil(copy=True)
    bp = b.copy()
    Lp[pin, :] = 0.0
    Lp[:, pin] = 0.0
    Lp[pin, pin] = 1.0
    bp[pin] = 0.0
    return spla.spsolve(Lp.tocsr(), bp)


def _initial_vector_field(
    V: np.ndarray,
    F: np.ndarray,
    sources: list[int],
    frames_t1: np.ndarray,
    frames_t2: np.ndarray,
) -> np.ndarray:
    n = V.shape[0]
    G, area, _ = per_face_grad_barycentric(V, F)
    X0 = np.zeros((n, 3), dtype=np.float64)
    for s in sources:
        mask = np.where(F == s)
        faces = mask[0]
        local = mask[1]
        if faces.size == 0:
            continue
        grad = (G[faces, local, :] * area[faces][:, None]).sum(axis=0)
        X0[s] += grad

    X0_local = np.zeros((n, 2), dtype=np.float64)
    X0_local[:, 0] = np.sum(X0 * frames_t1, axis=1)
    X0_local[:, 1] = np.sum(X0 * frames_t2, axis=1)
    return X0_local


def vector_heat_geodesic(
    mesh,
    source_ids,
    t: float | None = None,
    *,
    t_mult: float = 1.0,
    rhs_sign: int = -1,
) -> tuple[np.ndarray, dict]:
    """Compute geodesic distance using the Vector Heat Method."""

    V, F = mesh.V, mesh.F
    n = V.shape[0]

    if np.isscalar(source_ids):
        sources = [int(source_ids)]
    else:
        sources = [int(s) for s in source_ids]

    L_scalar = cotangent_laplacian(V, F)
    M_scalar = lumped_mass_barycentric(V, F)

    if t is None:
        # use mean edge length squared
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique_edges = np.unique(edges, axis=0)
        h = np.mean(np.linalg.norm(V[unique_edges[:, 0]] - V[unique_edges[:, 1]], axis=1))
        t = h * h
    t = float(t_mult) * float(t)

    ops = precompute_vector_heat_operators(V, F)
    X0_local = _initial_vector_field(V, F, sources, ops.frames_t1, ops.frames_t2)
    b_vec = ops.mass @ X0_local.reshape(-1)
    A_vec = (ops.mass + t * ops.connection_laplacian).tocsr()
    X_local = spla.spsolve(A_vec, b_vec)

    X_local = X_local.reshape(n, 2)
    X_vertex = (
        X_local[:, 0][:, None] * ops.frames_t1 + X_local[:, 1][:, None] * ops.frames_t2
    )

    X_face = (
        X_vertex[F[:, 0]] + X_vertex[F[:, 1]] + X_vertex[F[:, 2]]
    ) / 3.0
    norms = np.linalg.norm(X_face, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-16)
    X_face /= norms

    div = divergence_vertex_from_face_field(V, F, X_face)
    rhs = rhs_sign * div

    phi = _pin_solve(L_scalar, rhs, pin=0)
    phi -= phi.min()
    phi = np.maximum(phi, 0.0)

    return phi, {
        "t": float(t),
        "operators": ops,
        "M_scalar": M_scalar.tocsr(),
        "L_scalar": L_scalar.tocsr(),
        "rhs_sign": rhs_sign,
        "sources": sources,
    }

