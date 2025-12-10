import numpy as np
import polyscope as ps
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize Karcher Mean Results")
    parser.add_argument('file', default='karcher_data.npz', nargs='?')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found.")
        return

    # Reset config for clean start
    if os.path.exists("imgui.ini"):
        try: os.remove("imgui.ini")
        except: pass

    print(f"📂 Loading {args.file}...")
    data = np.load(args.file)
    V = data['vertices']
    F = data['faces']
    targets = data['target_indices']
    mean_idx = data['mean_index']

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # 1. Mesh (Set to a neutral color for visibility)
    ps_mesh = ps.register_surface_mesh("Mesh", V, F, color=(0.8, 0.8, 0.8)) # Light grey mesh

    # 2. Target Points (Red and slightly larger)
    target_pos = V[targets]
    # Use a highly visible radius
    ps.register_point_cloud("Target Points (Input)", target_pos, 
                            radius=0.007, 
                            color=(1, 0, 0), # Bright Red
                            enabled=True)

    # 3. Karcher Mean (Green and the largest)
    mean_pos = V[mean_idx].reshape(1, 3)
    # Use a distinctive color and larger radius
    ps.register_point_cloud("Karcher Mean (Result)", mean_pos, 
                            radius=0.010, 
                            color=(0, 1, 0), # Bright Green
                            enabled=True)

    print("✅ Targets are RED. Mean is GREEN. Press SPACEBAR to center view.")
    ps.show()

if __name__ == "__main__":
    main()