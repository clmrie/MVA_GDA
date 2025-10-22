# experiments/compare_dijkstra.py
import os, sys, time, numpy as np
sys.path.insert(0, os.path.abspath("."))

from mesh import Mesh
from heat_method import heat_geodesic_from_sources
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra


def edge_graph(V, F) -> csr_matrix:
    """Undirected edge-length graph from triangle mesh (for Dijkstra baseline)."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    n = V.shape[0]
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    data = np.concatenate([w, w])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def run_one(mesh_path: str, k_sources: int = 3,
            t_mult: float = 1.0, rhs_variant: str = "weak",
            rhs_sign: int = -1, heat_rhs: str = "Mdelta"):
    """Convenience runner on a single mesh (aligns φ to the source like Dijkstra)."""
    M = Mesh.load(mesh_path)
    V, F = M.V, M.F
    G = edge_graph(V, F)

    cand = np.argsort(np.linalg.norm(V - V.mean(0), axis=1))[-k_sources:]
    src = int(cand[0])

    t0 = time.perf_counter()
    phi, info = heat_geodesic_from_sources(
        M, src, t=None, t_mult=t_mult,
        rhs_variant=rhs_variant, rhs_sign=rhs_sign, heat_rhs=heat_rhs
    )
    th = time.perf_counter() - t0

    phi = phi - phi[src]
    phi = np.maximum(phi, 0.0)

    t1 = time.perf_counter()
    d = dijkstra(G, directed=False, indices=src)
    td = time.perf_counter() - t1

    # compare
    scale = max(np.median(d[d < np.inf]), 1e-12)
    err = np.abs(phi - d)
    return {
        "nV": V.shape[0], "nF": F.shape[0],
        "t_heat": th, "t_dijk": td,
        "mean_err": float(err.mean()), "max_err": float(err.max()),
        "rel_mean_err": float(err.mean() / scale), "rel_max_err": float(err.max() / scale),
        "t_used": info["t"], "rhs_variant": rhs_variant,
        "rhs_sign": rhs_sign, "t_mult": t_mult, "heat_rhs": heat_rhs,
    }


if __name__ == "__main__":
    bunny = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    torus = os.path.join("data", "torus.obj")

    
    bunny_configs = [
        (0.25, "weak", -1, "Mdelta"),
        (1.00, "weak", -1, "Mdelta"),
        (4.00, "weak", -1, "Mdelta"),
        (16.0, "weak", -1, "Mdelta"),
        (64.0, "weak", -1, "Mdelta"),
    ]
    torus_configs = [
        (1.0, "weak", -1, "Mdelta"),
        (4.0, "weak", -1, "Mdelta"),
    ]

    for path, cfgs in [(bunny, bunny_configs), (torus, torus_configs)]:
        print(f"\n=== {os.path.basename(path)} ===")
        for (tm, rv, rs, h_rhs) in cfgs:
            M = Mesh.load(path)
            V, F = M.V, M.F
            G = edge_graph(V, F)

            src = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))

            t0 = time.perf_counter()
            phi, info = heat_geodesic_from_sources(
                M, src,
                t=None,
                t_mult=tm,
                rhs_variant=rv,
                rhs_sign=rs,
                heat_rhs=h_rhs,
            )
            th = time.perf_counter() - t0

            phi = phi - phi[src]
            phi = np.maximum(phi, 0.0)

            t1 = time.perf_counter()
            d = dijkstra(G, directed=False, indices=src)
            td = time.perf_counter() - t1

            scale = max(np.median(d[d < np.inf]), 1e-12)
            err = np.abs(phi - d)

         
            print(
                f"t_mult={tm:>5}, rhs={rv:>4}, sign={rs:+d}, heat_rhs={h_rhs:<6}  "
                f"rel_mean={err.mean() / scale:.3f}, rel_max={err.max() / scale:.3f}, "
                f"mean={err.mean():.3f}, max={err.max():.3f},  "
                f"t_heat={th:.3f}s, t_dijk={td:.3f}s"
            )
