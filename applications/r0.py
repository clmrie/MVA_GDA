import numpy as np
from typing import Dict, List, Tuple

def _angle(u: np.ndarray, v: np.ndarray) -> float:
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    if den == 0.0:
        return 0.0
    c = num / den
    c = 1.0 if c > 1.0 else c
    c = -1.0 if c < -1.0 else c
    return float(np.arccos(c))

def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))

def _edge_faces(F: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    adj: Dict[Tuple[int, int], List[int]] = {}
    for f_idx, (a, b, c) in enumerate(F):
        for e0, e1 in ((a, b), (b, c), (c, a)):
            key = (e0, e1) if e0 < e1 else (e1, e0)
            adj.setdefault(key, []).append(f_idx)
    return adj

def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V)
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(vj - vi, vk - vi)
    np.add.at(n, F[:, 0], fn)
    np.add.at(n, F[:, 1], fn)
    np.add.at(n, F[:, 2], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-16)
    return n / norm

def _vertex_frames(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normals
    t1 = np.zeros_like(n)
    t2 = np.zeros_like(n)
    ref = np.array([0.0, 0.0, 1.0])
    for i in range(n.shape[0]):
        r = ref
        if abs(float(np.dot(n[i], r))) > 0.9:
            r = np.array([0.0, 1.0, 0.0])
        a = np.cross(n[i], r)
        an = float(np.linalg.norm(a))
        if an < 1e-12:
            a = np.array([1.0, 0.0, 0.0])
            an = 1.0
        a = a / an
        b = np.cross(n[i], a)
        b = b / (float(np.linalg.norm(b)) + 1e-12)
        t1[i] = a
        t2[i] = b
    return t1, t2

def _angle_in_frame(vec: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> float:
    x = float(np.dot(vec, e1))
    y = float(np.dot(vec, e2))
    return float(np.arctan2(y, x))

def _tri_with_vertex_first(face: np.ndarray, v_idx: int) -> Tuple[int, int]:
    if face[0] == v_idx:
        return int(face[1]), int(face[2])
    if face[1] == v_idx:
        return int(face[2]), int(face[0])
    return int(face[0]), int(face[1])

def initialize_R0(V: np.ndarray, F: np.ndarray, source: int, t1: np.ndarray | None = None, t2: np.ndarray | None = None) -> np.ndarray:
    n_verts = V.shape[0]
    if t1 is None or t2 is None:
        normals = _vertex_normals(V, F)
        t1, t2 = _vertex_frames(normals)

    edge_faces = _edge_faces(F)
    R0 = np.zeros(n_verts, dtype=np.complex128)
    
    source_neighbors = np.unique(F[np.any(F == source, axis=1)].flatten())
    
    for j in source_neighbors:
        j = int(j)
        if j == source:
            continue
            
        key = (source, j) if source < j else (j, source)
        faces = edge_faces.get(key, [])
        opp = []
        for f_idx in faces:
            face = F[f_idx]
            others = [v for v in face.tolist() if v not in (source, j)]
            if len(others) == 1:
                opp.append(int(others[0]))
        
        k = opp[0] if len(opp) > 0 else None
        l = opp[1] if len(opp) > 1 else None

        term = 0.0 + 0.0j
        vi = V[source]
        vj = V[j]

        if k is not None:
            vk = V[k]
            alpha = _angle(vj - vi, vk - vi)
            area = _tri_area(vi, vj, vk)
            if area > 1e-16:
                lik = float(np.linalg.norm(vi - vk))
                c = complex(alpha * np.sin(alpha), np.sin(alpha) - alpha * np.cos(alpha))
                term += (lik / (4.0 * area)) * c

        if l is not None:
            vl = V[l]
            beta = _angle(vj - vi, vl - vi)
            area = _tri_area(vj, vi, vl)
            if area > 1e-16:
                lil = float(np.linalg.norm(vi - vl))
                c = complex(beta * np.sin(beta), beta * np.cos(beta) - np.sin(beta))
                term += (lil / (4.0 * area)) * c

        if term != 0.0j:
            phi = _angle_in_frame(vi - vj, t1[j], t2[j])
            R0[j] = -np.exp(1j * phi) * term

    xi = 0.0 + 0.0j
    incident_faces = np.where(np.any(F == source, axis=1))[0]
    
    for f_idx in incident_faces:
        face = F[f_idx]
        j, k = _tri_with_vertex_first(face, source)
        vi, vj, vk = V[source], V[j], V[k]
        
        alpha = _angle(vj - vi, vk - vi)
        area = _tri_area(vi, vj, vk)
        
        if area < 1e-16:
            continue
            
        lij = float(np.linalg.norm(vi - vj))
        lik = float(np.linalg.norm(vi - vk))
        
        cx = -np.sin(alpha) * (lik * alpha + lij * np.sin(alpha))
        cy = lij * (np.cos(alpha) * np.sin(alpha) - alpha) + lik * (alpha * np.cos(alpha) - np.sin(alpha))
        
        tilde = complex(cx, cy) / (4.0 * area)
        phi = _angle_in_frame(vj - vi, t1[source], t2[source])
        xi += np.exp(1j * phi) * tilde
        
    R0[source] = xi
    return R0