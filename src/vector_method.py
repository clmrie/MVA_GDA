# vector_method.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

EPS = 1e-12


def build_lumped_mass(V, F):
    n = V.shape[0]
    M_diag = np.zeros(n, dtype=float)
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    for k, tri in enumerate(F):
        area = face_areas[k]
        M_diag[tri] += area / 3.0
    return sp.diags(M_diag), face_areas

def vertex_frames_from_normals(V, N):
    """
    Given normals N (n,3), return tangent frames t1,t2 per vertex (n,3).
    """
    n = V.shape[0]
    t1 = np.zeros((n,3), dtype=float)
    t2 = np.zeros((n,3), dtype=float)
    ref = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        nrm = N[i]
        r = ref
        if abs(np.dot(nrm, r)) > 0.9:
            r = np.array([0.0, 1.0, 0.0])
        a = np.cross(nrm, r)
        an = np.linalg.norm(a)
        if an < EPS:
            a = np.array([1.0, 0.0, 0.0])
            an = 1.0
        a = a / an
        b = np.cross(nrm, a)
        b = b / (np.linalg.norm(b) + EPS)
        t1[i] = a
        t2[i] = b
    return t1, t2

def cotangent_of_angle(a, b, c):
    u = b - a
    v = c - a
    cross = np.linalg.norm(np.cross(u, v))
    dot = np.dot(u, v)
    if cross < EPS:
        return 0.0
    return dot / cross


def build_connection_laplacian(V, F, t1, t2):
    """
    Build complex-valued connection Laplacian Lc (n x n, sparse complex).
    """
    n = V.shape[0]
    edge_w = {}

    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]
        cot_i = cotangent_of_angle(vi, vj, vk)
        cot_j = cotangent_of_angle(vj, vk, vi)
        cot_k = cotangent_of_angle(vk, vi, vj)
        
        def add(a,b,w):
            key = (a,b) if a < b else (b,a)
            edge_w[key] = edge_w.get(key, 0.0) + w
            
        add(i,j, 0.5 * cot_k)
        add(j,k, 0.5 * cot_i)
        add(k,i, 0.5 * cot_j)

    def edge_angle(i, j):
        d = V[j] - V[i]
        x = np.dot(d, t1[i])
        y = np.dot(d, t2[i])
        return np.arctan2(y, x)

    rows = []
    cols = []
    data = []

    for (i, j), w in edge_w.items():
        if abs(w) < EPS: 
            continue
        alpha_ij = edge_angle(i, j)
        alpha_ji = edge_angle(j, i)
        theta_ij = alpha_ji - alpha_ij
        r_ij = np.exp(1j * theta_ij)

        # i,i
        rows.append(i); cols.append(i); data.append(w)
        # i,j
        rows.append(i); cols.append(j); data.append(-w * r_ij)
        # j,i
        rows.append(j); cols.append(i); data.append(-w * np.conj(r_ij))
        # j,j
        rows.append(j); cols.append(j); data.append(w)

    Lc = sp.coo_matrix((np.array(data, dtype=np.complex128),
                        (rows, cols)), shape=(n, n)).tocsr()
    return Lc


def ambient_to_complex_per_vertex(w_ambient, t1, t2):
    a = np.einsum('ij,ij->i', w_ambient, t1)
    b = np.einsum('ij,ij->i', w_ambient, t2)
    return a + 1j * b

def complex_to_ambient(u_complex, t1, t2):
    a = u_complex.real
    b = u_complex.imag
    return a[:,None] * t1 + b[:,None] * t2


def vector_heat_transport(mesh, source_idx, vector_at_source, t_mult=1.0):
    """
    Computes parallel transport of a vector from source_idx to all vertices.
    Returns: (N, 3) array of transported vectors.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    n = V.shape[0]
    
    # 1. Geometry
    M, _ = build_lumped_mass(V, F)
    N = np.asarray(mesh.vertex_normals, dtype=float)
    t1, t2 = vertex_frames_from_normals(V, N)
    Lc = build_connection_laplacian(V, F, t1, t2)
    
    # 2. Time step (h^2)
    # Simple average edge length estimation
    edges = V[F[:, 1]] - V[F[:, 0]]
    avg_len = np.mean(np.linalg.norm(edges, axis=1))
    t = t_mult * (avg_len ** 2)

    # 3. Setup Source
    w0 = np.zeros((n, 3), dtype=float)
    
    # Normalize source vector
    v_src = np.array(vector_at_source, dtype=float)
    v_src /= (np.linalg.norm(v_src) + EPS)
    w0[source_idx] = v_src
    
    u0 = ambient_to_complex_per_vertex(w0, t1, t2)

    # 4. Solve (M + t Lc) u = M u0
    A = M.astype(np.complex128) + t * Lc
    b = M.dot(u0.astype(np.complex128))
    u = spla.spsolve(A, b)
    
    # 5. Convert back to R3
    # Normalize magnitude to purely represent direction (parallel transport preserves norm)
    # But numerical dissipation might reduce it, so we re-normalize.
    mag = np.abs(u)
    u_dir = u / (mag + EPS)
    
    vectors_out = complex_to_ambient(u_dir, t1, t2)
    return vectors_out