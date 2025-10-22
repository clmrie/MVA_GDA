
# tests/run_checks.py
import os, sys, time, numpy as np
sys.path.insert(0, os.path.abspath("."))

from mesh import Mesh
from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from operators.gradient import per_face_grad_barycentric
from operators.divergence import divergence_vertex_from_face_field
from heat_method import heat_geodesic_from_sources

BUNNY = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")

M = Mesh.load(BUNNY)
V, F = M.V, M.F
n = V.shape[0]

print("Mesh:", V.shape, F.shape)

L = cotangent_laplacian(V, F)
sym = (L - L.T).nnz == 0 or np.allclose((L - L.T).data, 0, atol=1e-9)
print("L symmetric:", sym)
print("||L·1||:", np.linalg.norm(L @ np.ones(n)))

Mmat = lumped_mass_barycentric(V, F)
print("M diag min/max:", float(Mmat.diagonal().min()), float(Mmat.diagonal().max()))

G, area, _ = per_face_grad_barycentric(V, F)
print("max ||∑_a ∇φ_a|| per face:", np.linalg.norm(G.sum(axis=1), ord=np.inf))

div0 = divergence_vertex_from_face_field(V, F, np.zeros((F.shape[0],3)))
print("||div(0)||:", np.linalg.norm(div0))

src = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))
t0 = time.perf_counter()
phi, info = heat_geodesic_from_sources(M, src)
t1 = time.perf_counter()
print(f"HeatMethod OK: t={info['t']:.3e}, phi range=({phi.min():.3g},{phi.max():.3g}), runtime={t1-t0:.3f}s")

