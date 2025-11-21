import os
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.abspath("."))

from mesh import Mesh
from heat_method import heat_geodesic_from_sources


def edge_graph(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """
    Undirected edge-length graph from triangle mesh.
    NOTE: This computes Graph Distance, which overestimates true Geodesic Distance.
    """
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
    """Robust scale (90th percentile) to normalize errors."""
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, 90.0))


def run_one(mesh_path: str, t_mult: float = 1.0, heat_rhs: str = "Mdelta"):
    """
    Run Heat vs Graph Dijkstra on a single mesh.
    Returns dict with sizes, errors, and timings.
    """
    if not os.path.exists(mesh_path):
        return None

    M = Mesh.load(mesh_path)
    V, F = M.V, M.F

    # 1. Build Graph (Baseline)
    t0 = time.perf_counter()
    G = edge_graph(V, F)
    t_graph = time.perf_counter() - t0

    # Source: farthest from centroid
    src = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))

    # 2. Run Heat Method
    t0 = time.perf_counter()
    phi, info = heat_geodesic_from_sources(
        M, src, t=None, t_mult=t_mult, heat_rhs=heat_rhs
    )
    t_heat_total = time.perf_counter() - t0

    phi = np.maximum(phi - phi[src], 0.0)

    # 3. Run Graph Dijkstra (Solve)
    t1 = time.perf_counter()
    d = dijkstra(G, directed=False, indices=src)
    t_graph_solve = time.perf_counter() - t1
    t_graph_total = t_graph + t_graph_solve

    # 4. Compare
    scale = max(robust_scale(d), 1e-12)
    err = np.abs(phi - d)

    return {
        "nV": int(V.shape[0]),
        "nF": int(F.shape[0]),
        "t_heat_total": float(t_heat_total),
        "t_graph_solve": float(t_graph_solve),
        "t_graph_total": float(t_graph_total),
        "mean_err": float(err.mean()),
        "max_err": float(err.max()),
        "rel_mean_err": float(err.mean() / scale),
        "rel_max_err": float(err.max() / scale),
        "t_used": float(info["t"]),
        "t_mult": float(t_mult),
        "heat_rhs": heat_rhs,
        "src": int(src),
    }


if __name__ == "__main__":
    bunny = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    torus = os.path.join("data", "torus.obj")

    configs = [
        (bunny, [0.25, 1.0, 4.0, 16.0, 64.0]),
        (torus, [1.0, 4.0]),
    ]

    print("Comparing Heat Method vs Graph Distance (Approximate Geodesic)")
    
    for path, t_mults in configs:
        name = os.path.basename(path)
        print(f"\n=== {name} ===")
        
        for tm in t_mults:
            out = run_one(path, t_mult=tm, heat_rhs="Mdelta")
            if out is None:
                print(f"File not found: {path}")
                break

            print(
                f"t_mult={tm:>5}  "
                f"rel_mean={out['rel_mean_err']:.3f}, rel_max={out['rel_max_err']:.3f},  "
                f"mean={out['mean_err']:.3f}, max={out['max_err']:.3f},  "
                f"t_heat={out['t_heat_total']:.3f}s, "
                f"t_graph={out['t_graph_total']:.3f}s"
            )