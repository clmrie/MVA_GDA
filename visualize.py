import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from src.mesh import Mesh
from src.heat_method import heat_geodesic_from_sources

try:
    from src.vector_method import vector_heat_transport
    HAS_VECTOR_METHOD = True
except ImportError:
    HAS_VECTOR_METHOD = False
    print("⚠️ Notice: vector_method.py not found. Using gradient approximation.")

# Data Names
MESH_NAME = "Bunny Mesh"
Q_DIST = "Geodesic Distance"
Q_VEC = "Log Map Vectors"
P_SRC = "Source Point"
P_TGT = "Target Points (Red)"
P_MEAN = "Karcher Mean (Green)"

def main(mesh_path):
    # 1. Init
    ps.init()
    ps.set_program_name("Geometric Data Analysis - Interactive")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_transparency_mode("none")
    
    # 2. Load
    if not os.path.exists(mesh_path):
        print(f"Error: {mesh_path} not found.")
        return

    M = Mesh.load(mesh_path)
    ps_mesh = ps.register_surface_mesh(MESH_NAME, M.V, M.F, smooth_shade=True)

    print("Computing Physics...")
    
    # --- PRE-COMPUTE DATA ---
    src_idx = int(np.argmax(np.linalg.norm(M.V - M.V.mean(0), axis=1)))
    phi, info = heat_geodesic_from_sources(M, src_idx)
    r = phi - phi.min()
    
    if HAS_VECTOR_METHOD:
        vecs = vector_heat_transport(M, src_idx, np.array([0., 1., 0.]))
    else:
        grad_u = info['X']
        vecs = np.zeros_like(M.V)
        np.add.at(vecs, M.F, grad_u[:, None, :])
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6)

    log_map_vecs = vecs * (r[:, None] / r.max() * 0.1)

    # Karcher Data
    np.random.seed(42)
    targets = M.V[np.random.choice(M.V.shape[0], size=3, replace=False)]
    mean_approx = np.mean(targets, axis=0)
    src_pos = M.V[[src_idx]]

    # 3. Initial State: Show Log Map
    ps_mesh.add_scalar_quantity(Q_DIST, r, cmap='turbo', enabled=True)
    ps_mesh.add_vector_quantity(Q_VEC, log_map_vecs, color=(0.2,0.2,0.2), length=0.03, radius=0.003, enabled=True)
    ps.register_point_cloud(P_SRC, src_pos, radius=0.015, color=(1,1,0), enabled=True)
    ps.register_point_cloud(P_TGT, targets, radius=0.02, color=(1,0,0), enabled=False)
    ps.register_point_cloud(P_MEAN, mean_approx[None,:], radius=0.025, color=(0,1,0), enabled=False)

    # 4. Define the UI Callback
    def my_callback():
        psim.PushItemWidth(150)
        psim.TextUnformatted("Select View Mode:")
        psim.Separator()

        # BUTTON 1: LOG MAP
        if psim.Button("View 1: Log Map"):
            
            ps_mesh.set_color((1.0, 1.0, 1.0))
            
            ps_mesh.add_scalar_quantity(Q_DIST, r, cmap='turbo', enabled=True)
            ps_mesh.add_vector_quantity(Q_VEC, log_map_vecs, color=(0.2,0.2,0.2), length=0.03, radius=0.003, enabled=True)
            ps.register_point_cloud(P_SRC, src_pos, radius=0.015, color=(1,1,0), enabled=True)
            
            ps.register_point_cloud(P_TGT, targets, radius=0.02, color=(1,0,0), enabled=False)
            ps.register_point_cloud(P_MEAN, mean_approx[None,:], radius=0.025, color=(0,1,0), enabled=False)

        # BUTTON 2: KARCHER MEAN
        if psim.Button("View 2: Karcher Mean"):
            ps_mesh.set_color((0.8, 0.8, 0.8))
            
            ps_mesh.add_scalar_quantity(Q_DIST, r, cmap='turbo', enabled=False)
            ps_mesh.add_vector_quantity(Q_VEC, log_map_vecs, color=(0.2,0.2,0.2), length=0.03, radius=0.003, enabled=False)
            ps.register_point_cloud(P_SRC, src_pos, radius=0.015, color=(1,1,0), enabled=False)
            
            ps.register_point_cloud(P_TGT, targets, radius=0.02, color=(1,0,0), enabled=True)
            ps.register_point_cloud(P_MEAN, mean_approx[None,:], radius=0.025, color=(0,1,0), enabled=True)
            
        psim.Separator()
        psim.TextUnformatted("Rotate: Left Click")
        psim.TextUnformatted("Screenshot: Camera Icon")

    # 5. Hook up the callback
    ps.set_user_callback(my_callback)

    print("\n✅ Interactive Window Open.")
    ps.show()

if __name__ == "__main__":
    path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    main(path)