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

    # --- RESET CONFIG ---
    if os.path.exists("imgui.ini"):
        try:
            os.remove("imgui.ini")
        except:
            pass
    # --------------------

    print(f"📂 Loading data from {args.file}...")
    data = np.load(args.file)
    V = data['vertices']
    F = data['faces']
    logmap_uv = data['logmap_uv']

    if np.any(np.isnan(V)) or np.any(np.isnan(logmap_uv)):
        print("❌ CRITICAL ERROR: Data contains NaNs.")
        return

    # --- THE FIX: NORMALIZE THE SCALE ---
    # We divide the coordinates by the maximum distance.
    # This acts like a "zoom" for the texture, making the squares bigger.
    max_val = np.max(np.abs(logmap_uv))
    if max_val > 0:
        logmap_uv = logmap_uv / max_val
        # Multiply by 10 to get a nice grid with about 10 squares across
        logmap_uv *= 10.0 
    # ------------------------------------

    print("🎨 Initializing Polyscope...")
    ps.init() 
    ps.set_up_dir("z_up") 
    ps.set_ground_plane_mode("shadow_only")

    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # We use 'viridis' (Blue/Green/Yellow) so it is high-contrast
    ps_mesh.add_parameterization_quantity("Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='checker', 
                                          cmap='viridis')

    print("✅ Opening viewer window. You should see a clear Yellow/Blue checkerboard.")
    ps.show()

if __name__ == "__main__":
    main()