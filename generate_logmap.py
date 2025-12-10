# generate_logmap.py
import sys
import os
import argparse
import numpy as np
import trimesh

# --- CRITICAL FIX: Add 'src' to the Python path ---
# This tells Python to look inside the 'src' folder for your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)
# --------------------------------------------------

# Now we can safely import from src/
try:
    from heat_method import heat_geodesic_from_sources
    from vector_method import vector_heat_transport
    from operators.gradient import gradient_scalar_per_face
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Debug: Checked in {src_dir}")
    sys.exit(1)

def check_nan(name, array):
    """Helper to check if an array contains NaNs or Infs."""
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        print(f"⚠️  WARNING: {name} contains NaNs or Infs! Fixing automatically...")
        return True
    return False

def average_face_field_to_vertices(mesh, face_field):
    """
    Moves a vector field defined on faces (like gradient) to vertices 
    by averaging, weighted by face area.
    """
    V = mesh.vertices
    F = mesh.faces
    n_verts = V.shape[0]
    
    # Compute face areas
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    # Weighted average accumulator
    vertex_vectors = np.zeros((n_verts, 3))
    vertex_weight_sum = np.zeros(n_verts)
    
    for i in range(3):
        # Add contribution of face to each of its vertices
        np.add.at(vertex_vectors, F[:, i], face_field * face_areas[:, None])
        np.add.at(vertex_weight_sum, F[:, i], face_areas)
        
    # Normalize by total weight (area)
    mask = vertex_weight_sum > 1e-12
    vertex_vectors[mask] /= vertex_weight_sum[mask][:, None]
    
    # Clean up vertices that had no area attached (isolated vertices)
    vertex_vectors[~mask] = 0.0
    
    return vertex_vectors

def compute_polar_coordinates(mesh, source_idx):
    """
    Computes Log Map coordinates (r, theta) and vector fields.
    Includes error handling for robust execution.
    """
    # Simple wrapper class if heat_method expects an object with V/F
    class SimpleMesh:
        def __init__(self, v, f):
            self.V = v
            self.F = f
    sm = SimpleMesh(mesh.vertices, mesh.faces)
    
    # --- 1. SCALAR HEAT: Geodesic Distance (r) ---
    print("--- Step 1: Scalar Heat (Distance) ---")
    try:
        phi, _ = heat_geodesic_from_sources(sm, [source_idx])
    except Exception as e:
        print(f"❌ Critical Error in Scalar Heat: {e}")
        return None, None, None, None

    # Handle NaNs in distance immediately
    if check_nan("Phi (Heat)", phi):
        phi = np.nan_to_num(phi)

    # Convert heat solution to distance
    r = phi - phi.min()
    
    # --- 2. VECTOR HEAT: Parallel Transport (X) ---
    print("--- Step 2: Vector Heat (Parallel Transport) ---")
    start_vec = np.array([1.0, 0.0, 0.0]) 
    try:
        X = vector_heat_transport(mesh, source_idx, start_vec)
    except Exception as e:
         print(f"❌ Error in Vector Heat: {e}")
         X = np.zeros((len(mesh.vertices), 3))

    if check_nan("Vectors X", X):
        X = np.nan_to_num(X)

    # --- 3. GRADIENT: Radial Direction (G) ---
    print("--- Step 3: Gradient (Radial Direction) ---")
    grad_per_face = gradient_scalar_per_face(mesh.vertices, mesh.faces, r)
    G = average_face_field_to_vertices(mesh, grad_per_face)
    
    # Secure Normalization
    norm_G = np.linalg.norm(G, axis=1, keepdims=True)
    G_normalized = np.divide(G, norm_G, out=np.zeros_like(G), where=norm_G > 1e-12)
    
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = np.divide(X, norm_X, out=np.zeros_like(X), where=norm_X > 1e-12)
    
    # --- 4. ANGLE: Polar Theta ---
    print("--- Step 4: Polar Angle ---")
    N = mesh.vertex_normals
    
    # Angle calculation using dot product and cross product
    dot_prod = np.sum(X_normalized * G_normalized, axis=1)
    cross_prod = np.cross(X_normalized, G_normalized)
    det = np.sum(cross_prod * N, axis=1)
    
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    theta = np.arctan2(det, dot_prod)
    
    # Clean final result
    check_nan("Theta", theta)
    theta = np.nan_to_num(theta)
    
    return r, theta, X, G

def main():
    # Default path based on your tree structure
    default_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")

    parser = argparse.ArgumentParser(description="Generate Log Map Data (.npz)")
    parser.add_argument('mesh_path', nargs='?', default=default_path, 
                        help=f"Path to input mesh. Defaults to {default_path}")
    parser.add_argument('--out', default='logmap_data.npz', help="Output NPZ file")
    args = parser.parse_args()
    
    # Verify file existence
    if not os.path.exists(args.mesh_path):
        print(f"Error: Mesh file not found at: {args.mesh_path}")
        print("Please check the path or provide a valid .ply/.obj file.")
        return

    print(f"Loading mesh: {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path, process=False)
    
    # --- Robust Source Selection ---
    # Find the vertex closest to the center of mass (safe bet)
    centroid = mesh.vertices.mean(axis=0)
    dists = np.linalg.norm(mesh.vertices - centroid, axis=1)
    source_idx = np.argmin(dists) 
    
    print(f"Source Index (Centroid-closest): {source_idx}")
    
    # Run Calculation
    r, theta, X, G = compute_polar_coordinates(mesh, source_idx)
    
    if r is None:
        print("Calculation failed.")
        return

    # Compute UV coordinates (Flattening)
    u = r * np.cos(theta)
    v = r * np.sin(theta)
    logmap_uv = np.stack([u, v], axis=1)

    print(f"Saving data to {args.out}...")
    np.savez(args.out, 
             vertices=mesh.vertices,
             faces=mesh.faces,
             r=r,
             theta=theta,
             logmap_uv=logmap_uv,
             vectors_X=X)
    print("Done! You can now download the .npz file and view it locally.")

if __name__ == "__main__":
    main()