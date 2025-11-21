# experiments/boundary_test.py
import os
import sys
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from mesh import Mesh
from heat_method import heat_geodesic_from_sources
from experiments.visualize import plot_isolines

def create_square_mesh(resolution=30):
    """Creates a simple 2D square grid mesh (-1 to 1)."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)
    V = np.column_stack([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())])
    
    # Triangulate a regular grid
    tri = scipy.spatial.Delaunay(V[:, :2])
    return Mesh(V=V, F=tri.simplices)

def run_boundary_test():
    print("Running Boundary Condition Test (Square Mesh)...")
    
    # 1. Use a Square Mesh
    # Neumann -> Concentric Circles (True Distance)
    # Dirichlet -> Rounded Squares (Heat shaped by boundary)
    M = create_square_mesh(resolution=40)
    
    # Source exactly in center
    src = int(np.argmin(np.linalg.norm(M.V, axis=1)))
    
    # Large time step to ensure heat hits the boundary hard
    t_factor = 10.0
    
    out_dir = os.path.join("results", "boundary")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Neumann (Natural)
    print("  Solving Neumann...")
    phi_n, _ = heat_geodesic_from_sources(
        M, src, boundary_condition="neumann", t_mult=t_factor
    )
    phi_n = np.maximum(phi_n - phi_n[src], 0.0)
    
    plot_isolines(
        M, phi_n, src, 
        "Neumann (Circles)", 
        os.path.join(out_dir, "square_neumann.png"),
        projection="xy",
        crop_to_cap=False
    )

    # 3. Dirichlet
    print("  Solving Dirichlet...")
    phi_d, _ = heat_geodesic_from_sources(
        M, src, boundary_condition="dirichlet", t_mult=t_factor
    )
    phi_d = np.maximum(phi_d - phi_d[src], 0.0)
    
    plot_isolines(
        M, phi_d, src, 
        "Dirichlet (Squares)", 
        os.path.join(out_dir, "square_dirichlet.png"),
        projection="xy",
        crop_to_cap=False
    )
    
    print(f"Done. Check {out_dir}/square_*.png")

if __name__ == "__main__":
    run_boundary_test()