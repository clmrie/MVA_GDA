# generate_logmap.py
import numpy as np
import trimesh
import os
import argparse

# Import your existing modules
from heat_method import heat_geodesic_from_sources
from vector_method import vector_heat_transport
from operators.gradient import gradient_scalar_per_face

def average_face_field_to_vertices(mesh, face_field):
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
        np.add.at(vertex_vectors, F[:, i], face_field * face_areas[:, None])
        np.add.at(vertex_weight_sum, F[:, i], face_areas)
        
    mask = vertex_weight_sum > 1e-12
    vertex_vectors[mask] /= vertex_weight_sum[mask][:, None]
    
    return vertex_vectors

def compute_polar_coordinates(mesh, source_idx):
    # 1. SCALAR HEAT: Distance (r)
    class SimpleMesh:
        def __init__(self, v, f):
            self.V = v
            self.F = f
    
    sm = SimpleMesh(mesh.vertices, mesh.faces)
    
    print("Computing geodesic distance (r)...")
    phi, _ = heat_geodesic_from_sources(sm, [source_idx])
    r = phi - phi.min() 
    
    # 2. VECTOR HEAT: Parallel Transport (X)
    print("Computing parallel transport (X)...")
    # Start with an arbitrary vector at the source
    start_vec = np.array([1.0, 0.0, 0.0]) 
    X = vector_heat_transport(mesh, source_idx, start_vec)
    
    # 3. GRADIENT: Radial Direction (G)
    print("Computing radial gradient (nabla r)...")
    grad_per_face = gradient_scalar_per_face(mesh.vertices, mesh.faces, r)
    G = average_face_field_to_vertices(mesh, grad_per_face)
    
    # Normalize for angle calculation
    norm_G = np.linalg.norm(G, axis=1, keepdims=True)
    G_normalized = np.divide(G, norm_G, out=np.zeros_like(G), where=norm_G > 1e-12)
    
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = np.divide(X, norm_X, out=np.zeros_like(X), where=norm_X > 1e-12)
    
    # 4. ANGLE (theta)
    print("Computing polar angle (theta)...")
    N = mesh.vertex_normals
    
    dot_prod = np.sum(X_normalized * G_normalized, axis=1)
    cross_prod = np.cross(X_normalized, G_normalized)
    det = np.sum(cross_prod * N, axis=1)
    
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    theta = np.arctan2(det, dot_prod)
    
    return r, theta, X, G

def main():
    default_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")

    parser = argparse.ArgumentParser(description="Generate Log Map Data")
    parser.add_argument('mesh_path', nargs='?', default=default_path, help="Path to input mesh")
    parser.add_argument('--out', default='logmap_data.npz', help="Output NPZ file")
    args = parser.parse_args()
    
    if not os.path.exists(args.mesh_path):
        print(f"Error: Mesh file not found at: {args.mesh_path}")
        return

    print(f"Loading mesh: {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path, process=False)
    
    # Find source
    centroid = mesh.vertices.mean(axis=0)
    dists = np.linalg.norm(mesh.vertices - centroid, axis=1)
    source_idx = np.argmax(dists) 
    print(f"Source Index: {source_idx}")
    
    # Compute fields
    r, theta, X, G = compute_polar_coordinates(mesh, source_idx)
    
    # Compute the actual 2D Log Map coordinates (u, v)
    # This flattens the manifold onto the tangent plane
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
             vectors_X=X,  # Parallel transported vectors
             vectors_G=G   # Radial gradient vectors
    )
    print("Done! Download this .npz file and run 'view_logmap.py' on your laptop.")

if __name__ == "__main__":
    main()