# tests/run_checks.py
import os, sys, time, numpy as np
sys.path.insert(0, os.path.abspath("."))

import trimesh

from mesh import Mesh
from operators.laplacian import cotangent_laplacian
from operators.mass_matrix import lumped_mass_barycentric
from operators.gradient import per_face_grad_barycentric
from operators.divergence import divergence_vertex_from_face_field
from heat_method import heat_geodesic_from_sources

BUNNY = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")

def load_any_mesh():
    if os.path.exists(BUNNY):
        print("[info] Loading bunny:", BUNNY)
        return Mesh.load(BUNNY)
    print("[warn] Bunny not found; creating a unit icosphere instead.")
    ico = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Use Mesh constructor directly to avoid Mesh.load rescaling
    return Mesh(V=ico.vertices.astype(np.float64), F=ico.faces.astype(np.int32))

def face_areas(V, F):
    vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    return 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)

def max_abs_sparse(M):
    coo = M.tocoo()
    return float(np.abs(coo.data).max()) if coo.nnz > 0 else 0.0

def random_rotation_matrix(rng):
    # Sample a random 3x3 orthonormal matrix via QR
    A = rng.normal(size=(3,3))
    Q, R = np.linalg.qr(A)
    # Ensure right-handed
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q

def main():
    Msh = load_any_mesh()
    V, F = Msh.V, Msh.F
    n, m = V.shape[0], F.shape[0]
    print(f"Mesh: V={V.shape}, F={F.shape}")

    # --- Operators
    L = cotangent_laplacian(V, F)
    Mmat = lumped_mass_barycentric(V, F)

    # 1) Symmetry & nullspace checks
    diff = L - L.T
    sym_ok = (diff.nnz == 0) or (max_abs_sparse(diff) < 1e-12)
    print("L symmetric:", sym_ok, f"(||L-L^T||_max≈{max_abs_sparse(diff):.2e})")

    ones = np.ones(n)
    L1 = L @ ones
    print("||L·1||_2:", float(np.linalg.norm(L1)))
    print("sum(L·1):", float(L1.sum()))

    # 2) Mass matrix sanity: positive diagonal & total area
    mdiag = Mmat.diagonal()
    print("M diag min/max:", float(mdiag.min()), float(mdiag.max()))
    A_faces = face_areas(V, F)
    area_from_M = float(mdiag.sum())
    area_faces = float(A_faces.sum())
    print("Total area from M vs faces:", area_from_M, area_faces,
          "| rel. err =", abs(area_from_M - area_faces) / max(1.0, area_faces))

    # 3) Partition of unity: sum_a ∇φ_a = 0 per-face
    G, area, _ = per_face_grad_barycentric(V, F)
    # G.shape = (m, 3, 3); sum over vertices (axis=1) -> (m,3)
    Gsum = G.sum(axis=1)
    # Max per-face vector 2-norm:
    pu_max = float(np.max(np.linalg.norm(Gsum, axis=1)))
    print("max ||∑_a ∇φ_a|| per face:", pu_max)

    # 4) Divergence of zero field is zero
    div0 = divergence_vertex_from_face_field(V, F, np.zeros((m,3)))
    print("||div(0)||_2:", float(np.linalg.norm(div0)))

    # 5) Heat distances on this mesh
    src = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))  # a vertex far from center
    t0 = time.perf_counter()
    phi, info = heat_geodesic_from_sources(Msh, src)
    t1 = time.perf_counter()
    print(f"HeatMethod: t={info['t']:.3e}, phi range=({phi.min():.3g},{phi.max():.3g}), runtime={t1-t0:.3f}s")
    print("phi[src] (should be near 0 after min-shift):", float(phi[src]))

    # 6) Rotation invariance (rigid transform should not change distances)
    rng = np.random.default_rng(0)
    R = random_rotation_matrix(rng)
    V_rot = (V @ R.T).astype(np.float64)
    Msh_rot = Mesh(V=V_rot, F=F)
    phi_rot, _ = heat_geodesic_from_sources(Msh_rot, src)
    # Distances are defined up to an additive constant; both are min-shifted to 0 already.
    rot_diff = float(np.max(np.abs(phi - phi_rot)))
    print("max |phi - phi_rot|:", rot_diff)

    # 7) Optional: analytic check on a clean sphere
    #    If current mesh is "sphere-like" (detected crudely) run a quick great-circle error.
    #    Otherwise, build a sphere and test there.
    def sphere_check():
        ico = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        S = Mesh(V=ico.vertices.astype(np.float64), F=ico.faces.astype(np.int32))
        V_s, F_s = S.V, S.F
        base = 0
        phi_s, _ = heat_geodesic_from_sources(S, base)
        # Great-circle distance on unit sphere: arccos(<x, x0>)
        x0 = V_s[base]
        dots = (V_s @ x0).clip(-1.0, 1.0)
        d_true = np.arccos(dots)
        l2 = float(np.linalg.norm(phi_s - d_true) / np.sqrt(len(d_true)))
        linf = float(np.max(np.abs(phi_s - d_true)))
        print(f"[sphere] RMSE={l2:.3e}, Linf={linf:.3e}, phi range=({phi_s.min():.3g},{phi_s.max():.3g})")

    try:
        sphere_check()
    except Exception as e:
        print("[warn] sphere check skipped:", repr(e))

if __name__ == "__main__":
    main()
