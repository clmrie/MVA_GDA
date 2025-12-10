

import numpy as np
import polyscope as ps
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', default='karcher_data.npz', nargs='?')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found.")
        return

    # Reset config
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

    # 1. Mesh
    ps_mesh = ps.register_surface_mesh("Mesh", V, F)

    # 2. Target Points (Red)
    # Get coordinates of targets
    target_pos = V[targets]
    ps.register_point_cloud("Target Points", target_pos, radius=0.005, color=(1, 0, 0))

    # 3. Karcher Mean (Green)
    # Get coordinate of mean
    mean_pos = V[mean_idx].reshape(1, 3)
    ps.register_point_cloud("Karcher Mean", mean_pos, radius=0.007, color=(0, 1, 0))

    print("✅ Targets are RED. Mean is GREEN.")
    ps.show()

if __name__ == "__main__":
    main()
    