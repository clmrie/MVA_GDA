#!/usr/bin/env python3
import importlib.util
import pkgutil

if not hasattr(pkgutil, "find_loader"):
    def _find_loader(name):
        spec = importlib.util.find_spec(name)
        return spec.loader if spec is not None else None
    pkgutil.find_loader = _find_loader


import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import os

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

def build_connection_laplacian(V, F, normals, t1, t2):
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
        rows.append(i); cols.append(i); data.append(w)
        rows.append(i); cols.append(j); data.append(-w * r_ij)
        rows.append(j); cols.append(i); data.append(-w * np.conj(r_ij))
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

def vector_heat_step(Lc, M, u0_complex, t_step):
    A = M.astype(np.complex128) + t_step * Lc
    b = M.dot(u0_complex.astype(np.complex128))
    u = spla.spsolve(A, b)
    return u

def vectors_to_colors(vecs):
    mags = np.linalg.norm(vecs, axis=1)
    maxmag = mags.max() if mags.max() > EPS else 1.0
    v = vecs / (maxmag + EPS)
    c = (v * 0.5) + 0.5
    c = np.clip(c, 0.0, 1.0)
    colors = (c * 255).astype(np.uint8)
    alpha = np.full((colors.shape[0],1), 255, dtype=np.uint8)
    rgba = np.concatenate([colors, alpha], axis=1)
    return rgba

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

    M, face_areas = build_lumped_mass(V, F)

    N = np.asarray(mesh.vertex_normals, dtype=float)
    t1, t2 = vertex_frames_from_normals(V, N)

    print("Building connection Laplacian...")
    Lc = build_connection_laplacian(V, F, N, t1, t2)

    w0 = np.zeros((n,3), dtype=float)
    if args.seed_vertex >= 0:
        seed = args.seed_vertex
    else:
        centroid = V.mean(axis=0)
        seed = np.argmin(np.linalg.norm(V - centroid[None,:], axis=1))
    print(f"Using seed vertex index = {seed}")

    seed_dir = np.asarray(args.seed_vector, dtype=float)
    if np.linalg.norm(seed_dir) < EPS:
        seed_dir = np.array([1.0,0.0,0.0])
    seed_dir = seed_dir / (np.linalg.norm(seed_dir) + EPS)
    w0[seed] = seed_dir

    u0 = ambient_to_complex_per_vertex(w0, t1, t2)

    print(f"Solving (M + t L) u = M u0 with t = {args.t} ...")
    u = vector_heat_step(Lc, M, u0, args.t)

    w = complex_to_ambient(u, t1, t2)

    out_base, out_ext = os.path.splitext(args.out)
    npz_name = out_base + ".npz"
    np.savez(npz_name, vertices=V, faces=F, vectors=w)
    print(f"Saved raw vectors to: {npz_name}")

    colors = vectors_to_colors(w)
    mesh_out = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors, process=False)
    ply_name = args.out if args.out.lower().endswith('.ply') else args.out + '.ply'
    mesh_out.export(ply_name)
    print(f"Saved visualization PLY to: {ply_name}")
    print("Done.")

if __name__ == "__main__":
    main()