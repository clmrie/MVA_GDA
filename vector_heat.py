#!/usr/bin/env python3
"""
Simple Vector Heat on a .obj mesh.

Saves:
 - output PLY with vertex colors encoding the resulting tangent vectors
 - output NPZ with raw vectors (vx, vy, vz) per vertex

Usage:
    python vector_heat_obj.py mesh.obj --t 0.1 --seed_vertex -1 --out result.ply
"""



# --- compatibility shim for Python >= 3.14 ---
import importlib.util
import pkgutil

if not hasattr(pkgutil, "find_loader"):
    def _find_loader(name):
        spec = importlib.util.find_spec(name)
        # return the loader if found (pkgutil.find_loader returned a loader or None)
        return spec.loader if spec is not None else None
    pkgutil.find_loader = _find_loader
# ------------------------------------------------



import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import os

EPS = 1e-12



# ---------- geometry helpers ----------

def build_lumped_mass(V, F):
    n = V.shape[0]
    M_diag = np.zeros(n, dtype=float)
    # triangle areas
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    # accumulate 1/3 * area at each incident vertex
    for k, tri in enumerate(F):
        area = face_areas[k]
        M_diag[tri] += area / 3.0
    return sp.diags(M_diag), face_areas

def vertex_frames_from_normals(V, N):
    """
    Given normals N (n,3), return tangent frames t1,t2 per vertex (n,3).
    Choose a stable arbitrary t1 per vertex.
    """
    n = V.shape[0]
    t1 = np.zeros((n,3), dtype=float)
    t2 = np.zeros((n,3), dtype=float)
    # global reference
    ref = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        nrm = N[i]
        r = ref
        if abs(np.dot(nrm, r)) > 0.9:
            r = np.array([0.0, 1.0, 0.0])
        a = np.cross(nrm, r)
        an = np.linalg.norm(a)
        if an < EPS:
            # fallback
            a = np.array([1.0, 0.0, 0.0])
            an = 1.0
        a = a / an
        b = np.cross(nrm, a)
        b = b / (np.linalg.norm(b) + EPS)
        t1[i] = a
        t2[i] = b
    return t1, t2

def cotangent_of_angle(a, b, c):
    # cot(angle at a) for triangle (a,b,c)
    u = b - a
    v = c - a
    cross = np.linalg.norm(np.cross(u, v))
    dot = np.dot(u, v)
    if cross < EPS:
        return 0.0
    return dot / cross

# ---------- connection Laplacian builder ----------

def build_connection_laplacian(V, F, normals, t1, t2):
    """
    Build complex-valued connection Laplacian Lc (n x n, sparse complex)
    using cotangent weights and per-vertex frames (t1,t2).
    """
    n = V.shape[0]

    # accumulate cotangent weights per undirected edge
    # We'll use a dict keyed by tuple(min,max) -> weight
    edge_w = {}

    # compute cotangents per triangle
    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]
        cot_i = cotangent_of_angle(vi, vj, vk)  # opposite (j,k)
        cot_j = cotangent_of_angle(vj, vk, vi)  # opposite (k,i)
        cot_k = cotangent_of_angle(vk, vi, vj)  # opposite (i,j)
        def add(a,b,w):
            key = (a,b) if a < b else (b,a)
            edge_w[key] = edge_w.get(key, 0.0) + w
        # using 0.5 * cot as conventional for discrete Laplacian
        add(i,j, 0.5 * cot_k)
        add(j,k, 0.5 * cot_i)
        add(k,i, 0.5 * cot_j)

    # helper: angle of vector (V[j]-V[i]) in frame at i
    def edge_angle(i, j):
        d = V[j] - V[i]
        # project onto tangent plane at i (should already be near tangent)
        x = np.dot(d, t1[i])
        y = np.dot(d, t2[i])
        return np.arctan2(y, x)

    rows = []
    cols = []
    data = []

    # For each undirected edge (i,j) with weight w:
    # Add contributions so that (L u)_i += w * (u_i - r_ij u_j)
    # This results in matrix entries:
    #  (i,i) += w
    #  (i,j) += - w * r_ij
    #  (j,i) += - w * conj(r_ij)
    #  (j,j) += w
    for (i, j), w in edge_w.items():
        if abs(w) < EPS: 
            continue
        alpha_ij = edge_angle(i, j)
        alpha_ji = edge_angle(j, i)
        theta_ij = alpha_ji - alpha_ij
        r_ij = np.exp(1j * theta_ij)  # complex unit rotation
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

# ---------- conversions between ambient and complex tangent ----------

def ambient_to_complex_per_vertex(w_ambient, t1, t2):
    # w_ambient: (n,3)
    a = np.einsum('ij,ij->i', w_ambient, t1)
    b = np.einsum('ij,ij->i', w_ambient, t2)
    return a + 1j * b

def complex_to_ambient(u_complex, t1, t2):
    a = u_complex.real
    b = u_complex.imag
    return a[:,None] * t1 + b[:,None] * t2

# ---------- solve vector heat ----------

def vector_heat_step(Lc, M, u0_complex, t_step):
    # A = M + t * Lc  (complex system)
    A = M.astype(np.complex128) + t_step * Lc
    b = M.dot(u0_complex.astype(np.complex128))
    # Solve sparse linear system
    u = spla.spsolve(A, b)
    return u

# ---------- utility: save output as PLY with vertex colors ----------

def vectors_to_colors(vecs):
    """
    Map vectors (n,3) to RGB colors (n,4 uint8) for quick visualization.
    We'll map x->R, y->G, z->B using signed values: shift to [0,1].
    Also encode alpha as 255.
    """
    mags = np.linalg.norm(vecs, axis=1)
    # avoid divide by zero
    maxmag = mags.max() if mags.max() > EPS else 1.0
    v = vecs / (maxmag + EPS)
    # shift [-1,1] -> [0,1]
    c = (v * 0.5) + 0.5
    c = np.clip(c, 0.0, 1.0)
    colors = (c * 255).astype(np.uint8)
    alpha = np.full((colors.shape[0],1), 255, dtype=np.uint8)
    rgba = np.concatenate([colors, alpha], axis=1)
    return rgba

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', help='.obj or other triangle mesh file')
    parser.add_argument('--t', type=float, default=0.1, help='time step for vector heat')
    parser.add_argument('--seed_vertex', type=int, default=-1, help='index of seed vertex (use -1 to choose centroid-closest)')
    parser.add_argument('--out', default='vector_heat_out.ply', help='output PLY filename (also saves .npz)')
    parser.add_argument('--seed_vector', nargs=3, type=float, default=[1.0,0.0,0.0], help='ambient seed vector direction at seed vertex')
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded mesh is not a Trimesh (file may contain multiple objects).")

    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    n = V.shape[0]
    print(f"Loaded mesh: {len(V)} vertices, {len(F)} faces.")

    # mass matrix
    M, face_areas = build_lumped_mass(V, F)

    # normals (trimesh provides per-vertex normals)
    N = np.asarray(mesh.vertex_normals, dtype=float)
    # frames
    t1, t2 = vertex_frames_from_normals(V, N)

    # Build connection Laplacian
    print("Building connection Laplacian...")
    Lc = build_connection_laplacian(V, F, N, t1, t2)

    # initial vector field: zero everywhere except one seed vertex
    w0 = np.zeros((n,3), dtype=float)
    if args.seed_vertex >= 0:
        seed = args.seed_vertex
    else:
        # pick vertex closest to centroid
        centroid = V.mean(axis=0)
        seed = np.argmin(np.linalg.norm(V - centroid[None,:], axis=1))
    print(f"Using seed vertex index = {seed}")

    seed_dir = np.asarray(args.seed_vector, dtype=float)
    if np.linalg.norm(seed_dir) < EPS:
        seed_dir = np.array([1.0,0.0,0.0])
    seed_dir = seed_dir / (np.linalg.norm(seed_dir) + EPS)
    w0[seed] = seed_dir

    # encode to complex tangent representation
    u0 = ambient_to_complex_per_vertex(w0, t1, t2)

    # solve vector heat
    print(f"Solving (M + t L) u = M u0 with t = {args.t} ...")
    u = vector_heat_step(Lc, M, u0, args.t)

    # decode to ambient
    w = complex_to_ambient(u, t1, t2)  # (n,3)

    # save raw vectors
    out_base, out_ext = os.path.splitext(args.out)
    npz_name = out_base + ".npz"
    np.savez(npz_name, vertices=V, faces=F, vectors=w)
    print(f"Saved raw vectors to: {npz_name}")

    # colorize and save PLY
    colors = vectors_to_colors(w)  # (n,4 uint8)
    mesh_out = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors, process=False)
    ply_name = args.out if args.out.lower().endswith('.ply') else args.out + '.ply'
    mesh_out.export(ply_name)
    print(f"Saved visualization PLY to: {ply_name}")
    print("Done.")

if __name__ == "__main__":
    main()
