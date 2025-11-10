# experiments/visualize_all.py
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from mesh import Mesh
from heat_method import heat_geodesic_from_sources
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

# -------------------------
# Mesh dictionary
# -------------------------
base_dir = "data"
mesh_paths = {
    "armadillo": os.path.join(base_dir, "Armadillo.ply"),
    "bunny": os.path.join(base_dir, "bunny/reconstruction/bun_zipper.ply"),
    "drill": os.path.join(base_dir, "drill/reconstruction/drill_shaft_vrip.ply"),
    "torus": os.path.join(base_dir, "torus.obj"),
}

# -------------------------
# Helpers
# -------------------------
def edge_graph(V, F) -> csr_matrix:
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    n = V.shape[0]
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    data = np.concatenate([w, w])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

def robust_scale(dist: np.ndarray) -> float:
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, 90.0))

def random_source(V: np.ndarray) -> int:
    """Random vertex index (seed must be set outside for reproducibility)."""
    return int(np.random.randint(0, len(V)))

def _pca_project(V: np.ndarray) -> np.ndarray:
    """Project 3D vertices to 2D via PCA (good for elongated meshes)."""
    X = V - V.mean(0, keepdims=True)
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    B = VT[:2]  # top-2 principal directions (2x3)
    return X @ B.T

def plot_isolines(
    M: Mesh,
    phi: np.ndarray,
    src: int,
    title: str,
    save_path: str,
    *,
    projection: str = "pca",
    vmax_percentile: float = 90.0,
    crop_to_cap: bool = True,
):
    V, F = M.V, M.F
    XY = V[:, :2] if projection == "xy" else _pca_project(V)

    vmin = float(phi.min())
    vmax = float(np.percentile(phi, vmax_percentile))
    tri = Triangulation(XY[:, 0], XY[:, 1], triangles=F)

    if crop_to_cap:
        tri_mask = (phi[F].max(axis=1) > vmax)
        tri.set_mask(tri_mask)
        kept_verts = np.unique(F[~tri_mask].ravel())
        if kept_verts.size > 0:
            xk, yk = XY[kept_verts, 0], XY[kept_verts, 1]
        else:
            xk, yk = XY[:, 0], XY[:, 1]
    else:
        xk, yk = XY[:, 0], XY[:, 1]

    levels = np.linspace(vmin, vmax, 40)
    fig, ax = plt.subplots(figsize=(7, 6))
    tpc = ax.tricontourf(tri, phi, levels=levels, cmap="viridis")
    ax.tricontour(tri, phi, levels=15, colors="k", linewidths=0.45, alpha=0.8)
    ax.plot([XY[src, 0]], [XY[src, 1]], "ro", markersize=4)
    ax.set_aspect("equal", adjustable="box")

    margin_x = 0.05 * (xk.max() - xk.min() + 1e-12)
    margin_y = 0.05 * (yk.max() - yk.min() + 1e-12)
    ax.set_xlim(xk.min() - margin_x, xk.max() + margin_x)
    ax.set_ylim(yk.min() - margin_y, yk.max() + margin_y)

    ax.set_title(title)
    fig.colorbar(tpc, ax=ax, shrink=0.8, label="distance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def sweep_t_and_plot(mesh_name: str, mesh_path: str, src: int):
    M = Mesh.load(mesh_path)
    V, F = M.V, M.F
    G = edge_graph(V, F)
    t_mults = (0.25, 1, 4, 16, 64)
    rel_means, rel_maxes = [], []

    for tm in t_mults:
        phi, _ = heat_geodesic_from_sources(
            M, src, t=None, t_mult=tm, heat_rhs="Mdelta"
        )
        phi = np.maximum(phi - phi[src], 0.0)
        d = dijkstra(G, directed=False, indices=src)
        scale = max(robust_scale(d), 1e-12)
        err = np.abs(phi - d)
        rel_means.append(float(err.mean() / scale))
        rel_maxes.append(float(err.max() / scale))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(t_mults, rel_means, "o-", label="mean error")
    ax.plot(t_mults, rel_maxes, "s--", label="max error", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("t / h²")
    ax.set_ylabel("relative error")
    ax.set_title(f"{mesh_name}: Error vs t/h²")
    ax.grid(True, ls=":")
    ax.legend()
    out_dir = os.path.join("results", "plots")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{mesh_name}_t_sweep.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved error plot: {save_path}")

# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    out_dir = os.path.join("results", "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Fixed random seed for reproducible sources across meshes
    np.random.seed(42)

    t_mult_map = {
        "armadillo": 32.0,
        "drill": 32.0,
        "bunny": 16.0,
        "torus": 4.0,
    }

    for name, path in mesh_paths.items():
        print(f"\n=== Processing {name} ===")
        M = Mesh.load(path)
        V, _ = M.V, M.F
        src = random_source(V)  # reproducible due to global seed
        t_mult = t_mult_map.get(name.lower(), 16.0)

        phi, _ = heat_geodesic_from_sources(
            M, src, t=None, t_mult=t_mult, heat_rhs="Mdelta"
        )
        phi = np.maximum(phi - phi[src], 0.0)

        isolines_png = os.path.join(out_dir, f"{name}_isolines_t{int(t_mult)}.png")
        plot_isolines(
            M, phi, src,
            f"{name} — Isolines (t={t_mult}h², random src)",
            isolines_png,
            projection="pca",
            vmax_percentile=90.0,
            crop_to_cap=True,
        )
        print(f"  → Saved isolines: {isolines_png}")

        sweep_t_and_plot(name, path, src)

    print("\nAll meshes processed. Plots saved in results/plots/")
