import os
import sys
import time
import numpy as np
import scipy.sparse.linalg

sys.path.insert(0, os.path.abspath("."))
from mesh import Mesh
from heat_method import compute_laplacian_matrices

def get_mean_edge_length(V, F):
    """
    Computes the mean edge length of the mesh locally.
    This avoids adding a helper method to the Mesh class.
    """
    # 1. Get all edges (3 per face)
    all_edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    
    # 2. Sort to ensure undirected edges (i, j) == (j, i)
    all_edges.sort(axis=1)
    
    # 3. Unique edges only
    unique_edges = np.unique(all_edges, axis=0)
    
    # 4. Compute lengths
    edge_vectors = V[unique_edges[:, 0]] - V[unique_edges[:, 1]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    
    return np.mean(lengths)

def measure_breakdown(mesh_path):
    print(f"\n=== Timing Breakdown: {os.path.basename(mesh_path)} ===")
    
    if not os.path.exists(mesh_path):
        print(f"File not found: {mesh_path}")
        return

    M = Mesh.load(mesh_path)
    
    # Calculate h locally instead of calling M.edge_lengths()
    h = get_mean_edge_length(M.V, M.F)
    t = h**2

    # 1. Update/Build Matrix (Step 1 of Dynamic)
    t0 = time.perf_counter()
    Lc, Mass = compute_laplacian_matrices(M)
    A = Mass - t * Lc
    t_build = (time.perf_counter() - t0) * 1000  # ms

    # 2. Factorization (The Bottleneck)
    t1 = time.perf_counter()
    solve_heat = scipy.sparse.linalg.factorized(A)
    t_factor = (time.perf_counter() - t1) * 1000 # ms

    # 3. Solve (The "Fast" part)
    # create a dummy rhs
    rhs = np.zeros(M.V.shape[0])
    rhs[0] = 1.0
    
    t2 = time.perf_counter()
    _ = solve_heat(rhs)
    t_solve = (time.perf_counter() - t2) * 1000  # ms

    total = t_build + t_factor + t_solve

    print(f"1. Build Matrix:   {t_build:.2f} ms ({t_build/total:.1%})")
    print(f"2. Factorization:  {t_factor:.2f} ms ({t_factor/total:.1%})")
    print(f"3. Back-Sub Solve: {t_solve:.2f}  ms ({t_solve/total:.1%})")
    print(f"--------------------------------")
    print(f"TOTAL TIME:        {total:.2f} ms")

if __name__ == "__main__":
    # Run on a medium-sized mesh (e.g., Bunny or Armadillo)
    mesh_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    measure_breakdown(mesh_path)