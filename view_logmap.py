import numpy as np
import polyscope as ps
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', default='logmap_data.npz', nargs='?')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found.")
        return

    # --- FORCE RESET: DELETE CONFIG FILE ---
    if os.path.exists("imgui.ini"):
        try:
            os.remove("imgui.ini")
        except:
            pass
    # ---------------------------------------

    print(f"📂 Loading data from {args.file}...")
    data = np.load(args.file)
    V = data['vertices']
    F = data['faces']
    logmap_uv = data['logmap_uv']

    # CRITICAL CHANGE: We DO NOT load 'r' (distance) or 'vec_X' (vectors).
    # If we don't load them, Polyscope CANNOT show them.

    # Check for corruption
    if np.any(np.isnan(V)) or np.any(np.isnan(logmap_uv)):
        print("❌ CRITICAL ERROR: Data contains NaNs.")
        return

    print("🎨 Initializing Polyscope...")
    ps.init() 
    ps.set_up_dir("z_up") 
    ps.set_ground_plane_mode("shadow_only")

    # Register the mesh
    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # Add ONLY the Log Map Grid
    # We use 'checkerboard' style here as it is often more visible than 'grid' on white meshes
    ps_mesh.add_parameterization_quantity("Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='checkerboard',
                                          cmap='blues')

    print("✅ Opening viewer window. You should ONLY see the grid/checkerboard.")
    ps.show()

if __name__ == "__main__":
    main()