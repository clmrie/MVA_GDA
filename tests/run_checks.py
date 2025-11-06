"""Sanity checks for the heat and vector heat implementations."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from heat_method import heat_geodesic_from_sources
from mesh import Mesh
from operators.divergence import divergence_vertex_from_face_field
from operators.gradient import per_face_grad_barycentric
from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from vector_heat_method import vector_heat_geodesic

BUNNY = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")


def main() -> None:
    mesh = Mesh.load(BUNNY)
    V, F = mesh.V, mesh.F
    n = V.shape[0]
    print("Mesh:", V.shape, F.shape)

    L = cotangent_laplacian(V, F)
    sym = (L - L.T).nnz == 0 or np.allclose((L - L.T).data, 0.0, atol=1e-9)
    print("L symmetric:", sym)
    print("||L·1||:", float(np.linalg.norm(L @ np.ones(n))))

    M = lumped_mass_barycentric(V, F)
    print("M diag min/max:", float(M.diagonal().min()), float(M.diagonal().max()))

    G, _, _ = per_face_grad_barycentric(V, F)
    print("max ||∑_a ∇φ_a|| per face:", float(np.linalg.norm(G.sum(axis=1), ord=np.inf)))

    div0 = divergence_vertex_from_face_field(V, F, np.zeros((F.shape[0], 3)))
    print("||div(0)||:", float(np.linalg.norm(div0)))

    source = int(np.argmax(np.linalg.norm(V - V.mean(axis=0), axis=1)))

    t0 = time.perf_counter()
    phi_heat, info_heat = heat_geodesic_from_sources(mesh, source)
    t1 = time.perf_counter()
    print(
        "HeatMethod OK: "
        f"t={info_heat['t']:.3e}, "
        f"phi range=({phi_heat.min():.3g},{phi_heat.max():.3g}), "
        f"runtime={t1 - t0:.3f}s"
    )

    t2 = time.perf_counter()
    phi_vec, info_vec = vector_heat_geodesic(mesh, source, t=info_heat["t"])
    t3 = time.perf_counter()
    rel_err = np.linalg.norm(phi_vec - phi_heat) / max(np.linalg.norm(phi_heat), 1e-12)
    print(
        "VectorHeatMethod OK: "
        f"t={info_vec['t']:.3e}, "
        f"phi range=({phi_vec.min():.3g},{phi_vec.max():.3g}), "
        f"runtime={t3 - t2:.3f}s, "
        f"rel err vs heat={rel_err:.3e}"
    )


if __name__ == "__main__":
    main()
