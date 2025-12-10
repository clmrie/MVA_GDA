# generate_logmap.py (Version Robuste & Debug)
import numpy as np
import trimesh
import os
import argparse
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Import your existing modules
from heat_method import heat_geodesic_from_sources
from vector_method import vector_heat_transport
from operators.gradient import gradient_scalar_per_face

def check_nan(name, array):
    """Petit utilitaire pour vérifier si un tableau est cassé."""
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        print(f"⚠️  ALERTE: {name} contient des NaNs ou Infs !")
        return True
    return False

def average_face_field_to_vertices(mesh, face_field):
    V = mesh.vertices
    F = mesh.faces
    n_verts = V.shape[0]
    
    # Calcul des aires
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    vertex_vectors = np.zeros((n_verts, 3))
    vertex_weight_sum = np.zeros(n_verts)
    
    for i in range(3):
        np.add.at(vertex_vectors, F[:, i], face_field * face_areas[:, None])
        np.add.at(vertex_weight_sum, F[:, i], face_areas)
        
    # Éviter la division par zéro
    mask = vertex_weight_sum > 1e-12
    vertex_vectors[mask] /= vertex_weight_sum[mask][:, None]
    
    # Remplacer les valeurs non touchées par 0 pour éviter les NaN résiduels
    vertex_vectors[~mask] = 0.0
    
    return vertex_vectors

def compute_polar_coordinates(mesh, source_idx):
    # Wrapper simple pour heat_method
    class SimpleMesh:
        def __init__(self, v, f):
            self.V = v
            self.F = f
    sm = SimpleMesh(mesh.vertices, mesh.faces)
    
    # 1. SCALAR HEAT : Distance (r)
    print("--- Étape 1 : Distance Géodésique (r) ---")
    try:
        # On tente le calcul normal
        phi, _ = heat_geodesic_from_sources(sm, [source_idx])
    except Exception as e:
        print(f"❌ Erreur critique dans Scalar Heat: {e}")
        return None, None, None, None

    if check_nan("Phi (Heat)", phi):
        # Si ça plante, on renvoie des zéros pour ne pas crasher le viewer
        return np.zeros(len(mesh.vertices)), np.zeros(len(mesh.vertices)), np.zeros((len(mesh.vertices), 3)), np.zeros((len(mesh.vertices), 3))

    r = phi - phi.min()
    
    # 2. VECTOR HEAT : Transport (X)
    print("--- Étape 2 : Transport Parallèle (X) ---")
    start_vec = np.array([1.0, 0.0, 0.0]) 
    try:
        X = vector_heat_transport(mesh, source_idx, start_vec)
    except Exception as e:
         print(f"❌ Erreur dans Vector Heat: {e}")
         X = np.zeros((len(mesh.vertices), 3))

    if check_nan("Vecteurs X", X):
        X = np.nan_to_num(X)

    # 3. GRADIENT : Direction Radiale (G)
    print("--- Étape 3 : Gradient Radial (G) ---")
    grad_per_face = gradient_scalar_per_face(mesh.vertices, mesh.faces, r)
    G = average_face_field_to_vertices(mesh, grad_per_face)
    
    # Normalisation Sécurisée
    norm_G = np.linalg.norm(G, axis=1, keepdims=True)
    G_normalized = np.divide(G, norm_G, out=np.zeros_like(G), where=norm_G > 1e-12)
    
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = np.divide(X, norm_X, out=np.zeros_like(X), where=norm_X > 1e-12)
    
    # 4. ANGLE (theta)
    print("--- Étape 4 : Angle Polaire (theta) ---")
    N = mesh.vertex_normals
    
    dot_prod = np.sum(X_normalized * G_normalized, axis=1)
    cross_prod = np.cross(X_normalized, G_normalized)
    det = np.sum(cross_prod * N, axis=1)
    
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    theta = np.arctan2(det, dot_prod)
    
    check_nan("Theta", theta)
    
    return r, theta, X, G

def main():
    default_path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")

    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path', nargs='?', default=default_path)
    parser.add_argument('--out', default='logmap_data.npz')
    args = parser.parse_args()
    
    if not os.path.exists(args.mesh_path):
        print(f"Fichier introuvable : {args.mesh_path}")
        return

    print(f"Chargement de : {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path, process=False) # process=False garde la géométrie brute
    
    # --- CHANGEMENT CLÉ : Choix de la source ---
    # Au lieu du point le plus loin (argmax), on prend le plus proche du centre (argmin)
    # C'est beaucoup plus sûr pour éviter les morceaux détachés ou les bugs.
    centroid = mesh.vertices.mean(axis=0)
    dists = np.linalg.norm(mesh.vertices - centroid, axis=1)
    source_idx = np.argmin(dists) 
    
    print(f"Source Index (Le plus central) : {source_idx}")
    
    # Calcul
    r, theta, X, G = compute_polar_coordinates(mesh, source_idx)
    
    if r is None:
        print("Échec du calcul.")
        return

    # Si r contient encore des NaNs malgré tout, on les nettoie
    r = np.nan_to_num(r)
    theta = np.nan_to_num(theta)
    X = np.nan_to_num(X)
    
    u = r * np.cos(theta)
    v = r * np.sin(theta)
    logmap_uv = np.stack([u, v], axis=1)

    print(f"Sauvegarde dans {args.out}...")
    np.savez(args.out, 
             vertices=mesh.vertices,
             faces=mesh.faces,
             r=r,
             theta=theta,
             logmap_uv=logmap_uv,
             vectors_X=X)
    print("Terminé ! Télécharge le fichier .npz et réessaie sur ton PC.")

if __name__ == "__main__":
    main()