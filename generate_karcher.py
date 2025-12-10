

import sys
import os
import argparse
import numpy as np
import trimesh

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

try:
    from heat_method import heat_geodesic_from_sources
    from vector_method import vector_heat_transport
    from operators.gradient import gradient_scalar_per_face
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def average_face_field_to_vertices(mesh, face_field):
    V = mesh.vertices
    F = mesh.faces
    n_verts = V.shape[0]
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    vertex_vectors = np.zeros((n_verts, 3))
    vertex_weight_sum = np.zeros(n_verts)
    for i in range(3):
        np.add.at(vertex_vectors, F[:, i], face_field * face_areas[:, None])
        np.add.at(vertex_weight_sum, F[:, i], face_areas)
    
    mask = vertex_weight_sum > 1e-12
    vertex_vectors[mask] /= vertex_weight_sum[mask][:, None]
    return vertex_vectors

def get_logmap_complex(mesh, source_idx):
    """
    Returns the Log Map as complex numbers z = r * exp(i*theta) for all vertices.
    """
    class SimpleMesh:
        def __init__(self, v, f):
            self.V, self.F = v, f
    sm = SimpleMesh(mesh.vertices, mesh.faces)

    # Scalar Heat (r)
    try:
        phi, _ = heat_geodesic_from_sources(sm, [source_idx])
        phi = np.nan_to_num(phi)
        r = phi - phi.min()
    except:
        return np.zeros(len(mesh.vertices), dtype=complex)

    # Vector Heat (X)
    start_vec = np.array([1.0, 0.0, 0.0])
    try:
        X = vector_heat_transport(mesh, source_idx, start_vec)
        X = np.nan_to_num(X)
    except:
        X = np.zeros((len(mesh.vertices), 3))

    # Gradient (G) and Angle (theta)
    grad_per_face = gradient_scalar_per_face(mesh.vertices, mesh.faces, r)
    G = average_face_field_to_vertices(mesh, grad_per_face)
    
    N = mesh.vertex_normals
    norm_G = np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    norm_X = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    G_n, X_n = G / norm_G, X / norm_X
    
    dot = np.clip(np.sum(X_n * G_n, axis=1), -1.0, 1.0)
    det = np.sum(np.cross(X_n, G_n) * N, axis=1)
    theta = np.arctan2(det, dot)
    
    return r * np.exp(1j * theta)


def compute_karcher_mean(mesh, targets, max_iters=15):
    """
    Finds the geometric center of 'targets' indices.
    """
    # Initialize at the vertex closest to the Euclidean centroid
    target_coords = mesh.vertices[targets]
    euclidean_mean = np.mean(target_coords, axis=0)
    dists = np.linalg.norm(mesh.vertices - euclidean_mean, axis=1)
    current_mean_idx = np.argmin(dists)
    
    print(f"Start: Vertex {current_mean_idx}")

    for i in range(max_iters):
        # 1. Compute Log Map centered at current mean
        z_log = get_logmap_complex(mesh, current_mean_idx)
        
        # 2. Get vectors pointing to the targets
        log_vectors = z_log[targets]
        
        # 3. Average them (this is the direction to the true mean)
        avg_vector = np.mean(log_vectors)
        step_dist = np.abs(avg_vector)
        
        # Convergence check
        if step_dist < 1e-4 * np.max(np.abs(z_log)):
            print(f"Converged at iter {i}")
            break
            
        # 4. Move mean: Find vertex whose log coordinate is closest to avg_vector
        dist_in_tangent_plane = np.abs(z_log - avg_vector)
        next_mean_idx = np.argmin(dist_in_tangent_plane)
        
        if next_mean_idx == current_mean_idx:
            print("Stalled (cannot move closer). Stopping.")
            break
            
        current_mean_idx = next_mean_idx
        print(f"Iter {i+1}: Moved to Vertex {current_mean_idx} (Error: {step_dist:.4f})")

    return current_mean_idx


def main():
    default_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path', nargs='?', default=default_path)
    parser.add_argument('--out', default='karcher_data.npz')
    args = parser.parse_args()

    if not os.path.exists(args.mesh_path):
        print("Mesh file not found.")
        return

    print(f"Loading mesh: {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path, process=False)
    
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: len(m.vertices))
    print(f"Cleaned Mesh: {len(mesh.vertices)} verts")
    
   
    np.random.seed(99)
    targets = np.random.choice(len(mesh.vertices), 3, replace=False)
    print(f"Targets indices: {targets}")

    mean_idx = compute_karcher_mean(mesh, targets)
    print(f"Result: Karcher Mean is Vertex {mean_idx}")

    print(f"Saving to {args.out}...")
    np.savez(args.out, 
             vertices=mesh.vertices,
             faces=mesh.faces,
             target_indices=targets,
             mean_index=mean_idx)
    print("Done! Download the file and run view_karcher.py.")

if __name__ == "__main__":
    main()
    