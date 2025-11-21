
# experiments/robustness_noise.py
import os
import sys
import numpy as np

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath("."))

from mesh import Mesh
from heat_method import heat_geodesic_from_sources
from experiments.visualize import plot_isolines

def add_noise(mesh: Mesh, noise_level: float = 0.01) -> Mesh:
    """
    Creates a new Mesh with Gaussian noise added to the vertices.
    
    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    noise_level : float
        Standard deviation of the Gaussian noise relative to the mesh bounding box diagonal.
    
    Returns
    -------
    Mesh
        A new Mesh object with noisy vertices.
    """
    V = mesh.V
    # Calculate bounding box diagonal to scale noise appropriately
    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    
    sigma = noise_level * diag
    noise = np.random.normal(scale=sigma, size=V.shape)
    
    # Create new mesh with perturbed vertices; faces remain the same
    return Mesh(V=V + noise, F=mesh.F)

def run_noise_experiment(mesh_name: str, mesh_path: str, noise_level: float = 0.02):
    """
    Runs the heat method on a clean mesh and a noisy version, compares the results,
    and saves visualizations.
    """
    print(f"\n=== Running Robustness Test on {mesh_name} ===")
    
    # 1. Load Clean Mesh
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return

    M_clean = Mesh.load(mesh_path)
    
    # Pick a deterministic source vertex (e.g., 0 or argmax distance from centroid)
    src = 0
    
    # 2. Run Heat Method on Clean Mesh
    print("Running Heat Method on Clean Mesh...")
    phi_clean, _ = heat_geodesic_from_sources(M_clean, src, heat_rhs="Mdelta")
    # Shift so source is at 0
    phi_clean = np.maximum(phi_clean - phi_clean[src], 0.0)
    
    # 3. Generate Noisy Mesh
    print(f"Generating Noisy Mesh (noise_level={noise_level})...")
    M_noisy = add_noise(M_clean, noise_level=noise_level)
    
    # 4. Run Heat Method on Noisy Mesh
    print("Running Heat Method on Noisy Mesh...")
    phi_noisy, _ = heat_geodesic_from_sources(M_noisy, src, heat_rhs="Mdelta")
    phi_noisy = np.maximum(phi_noisy - phi_noisy[src], 0.0)
    
    # 5. Compare Results (L2 and L-inf diff)
    diff = np.abs(phi_clean - phi_noisy)
    max_dist = np.max(phi_clean) if np.max(phi_clean) > 0 else 1.0
    
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    rel_mean_diff = mean_diff / max_dist
    
    print("-" * 40)
    print(f"Mean Absolute Diff:     {mean_diff:.6f}")
    print(f"Max Absolute Diff:      {max_diff:.6f}")
    print(f"Relative Mean Error:    {rel_mean_diff:.6%}")
    print("-" * 40)
    
    # 6. Save Visualizations
    out_dir = os.path.join("results", "robustness")
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot Clean
    clean_img_path = os.path.join(out_dir, f"{mesh_name}_clean.png")
    plot_isolines(
        M_clean, 
        phi_clean, 
        src, 
        f"{mesh_name} - Clean", 
        clean_img_path
    )
    print(f"Saved clean plot: {clean_img_path}")
    
    # Plot Noisy
    noisy_img_path = os.path.join(out_dir, f"{mesh_name}_noisy.png")
    plot_isolines(
        M_noisy, 
        phi_noisy, 
        src, 
        f"{mesh_name} - Noisy (lvl={noise_level})", 
        noisy_img_path
    )
    print(f"Saved noisy plot: {noisy_img_path}")

if __name__ == "__main__":
    # Ensure reproducible noise
    np.random.seed(42)
    
    # Define test cases
    # Adjust paths as necessary for your environment
    bunny_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    
    # Run experiment
    # You can add more meshes here if available
    run_noise_experiment("bunny", bunny_path, noise_level=0.005)