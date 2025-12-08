

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from heat_method import compute_laplacian_matrices, _typical_edge_length

def build_connection_laplacian(V, F, t1, t2):
    """Builds the complex Connection Laplacian."""
    n = V.shape[0]
    edge_w = {}
    
    # Helper: Cotangent
    def cotangent(a, b, c):
        u, v = b - a, c - a
        return np.dot(u, v) / np.linalg.norm(np.cross(u, v))

    # Accumulate weights
    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]
        w_k = 0.5 * cotangent(vi, vj, vk)
        w_i = 0.5 * cotangent(vj, vk, vi)
        w_j = 0.5 * cotangent(vk, vi, vj)
        
        for u, v, w in [(i, j, w_k), (j, k, w_i), (k, i, w_j)]:
            key = tuple(sorted((u, v)))
            edge_w[key] = edge_w.get(key, 0.0) + w

    # Helper: Angle in tangent plane
    def angle(i, j):
        vec = V[j] - V[i]
        x, y = np.dot(vec, t1[i]), np.dot(vec, t2[i])
        return np.arctan2(y, x)

    rows, cols, data = [], [], []
    for (i, j), w in edge_w.items():
        theta = angle(j, i) - angle(i, j) + np.pi
        r_ij = np.exp(1j * theta)
        
        rows.extend([i, i, j, j])
        cols.extend([i, j, i, j])
        data.extend([w, -w*r_ij, -w*np.conj(r_ij), w])

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

def compute_tangent_frames(V, N):
    """Generates arbitrary tangent frames (t1, t2) for every vertex."""
    t1 = np.cross(N, np.array([0, 0, 1]))
    mask = np.linalg.norm(t1, axis=1) < 1e-6
    t1[mask] = np.cross(N[mask], np.array([0, 1, 0]))
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(N, t1)
    return t1, t2

def vector_heat_transport(mesh, source_idx, vector_at_source, t_mult=1.0):
    """
    Transports a vector from source_idx to all other vertices.
    Returns: (N, 3) array of vectors.
    """
    V, F = mesh.V, mesh.F
    
    # 1. Geometry setup
    # Note: Using trimesh normals if available, else simplified
    import trimesh
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    N = tm.vertex_normals
    t1, t2 = compute_tangent_frames(V, N)
    
    # 2. Build Matrices
    Lc = build_connection_laplacian(V, F, t1, t2)
    # Re-use your mass matrix logic
    from operators.mass_matrix import lumped_mass_barycentric
    M = lumped_mass_barycentric(V, F)
    
    # 3. Time step
    h = _typical_edge_length(V, F)
    t = t_mult * h ** 2

    # 4. Setup Source (in Complex Plane)
    src_complex = np.dot(vector_at_source, t1[source_idx]) + 1j * np.dot(vector_at_source, t2[source_idx])
    u0 = np.zeros(V.shape[0], dtype=np.complex128)
    u0[source_idx] = src_complex

    # 5. Solve (M - t Lc) u = M u0
    A = M.astype(np.complex128) + t * Lc
    b = M @ u0
    u = spla.spsolve(A, b)
    
    # 6. Convert back to 3D vectors
    u_normalized = u / (np.abs(u) + 1e-12) # Normalize to separate direction from magnitude
    vecs = u_normalized.real[:, None] * t1 + u_normalized.imag[:, None] * t2
    
    return vecs
    