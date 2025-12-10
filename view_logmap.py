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
    # This shrinks the numbers from ~100.0 down to ~1.0.
    # Result: The checkerboard squares become 100x bigger and visible!
    max_val = np.max(np.abs(logmap_uv))
    if max_val > 0:
        logmap_uv = logmap_uv / max_val
        # Multiply by 10 to get a nice 10x10 grid look
        logmap_uv *= 10.0 
    # ------------------------------------

    print("🎨 Initializing Polyscope...")
    ps.init() 
    ps.set_up_dir("z_up") 
    ps.set_ground_plane_mode("shadow_only")

    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # We use 'viridis' color map now, which is high-contrast (Yellow/Blue)
    # This ensures you won't confuse it with the pink distance field.
    ps_mesh.add_parameterization_quantity("Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='checker', 
                                          cmap='viridis')

    print("✅ Opening viewer window. You should see a clear Yellow/Blue checkerboard.")
    ps.show()

if __name__ == "__main__":
    main()