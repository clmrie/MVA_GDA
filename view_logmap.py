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

    # We deliberately DO NOT load 'r' or 'vectors_X' to force the clean view.

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
    # CHANGED: 'checkerboard' -> 'checker' to match your Polyscope version
    ps_mesh.add_parameterization_quantity("Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='checker', 
                                          cmap='blues')

    print("✅ Opening viewer window. You should ONLY see the checkerboard.")
    ps.show()

if __name__ == "__main__":
    main()