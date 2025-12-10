


import numpy as np
import polyscope as ps
import argparse
import os

def main():
    # 1. Setup argument parser to easily load different files if needed
    parser = argparse.ArgumentParser(description="Visualize Log Map from NPZ data")
    parser.add_argument('file', default='logmap_data.npz', nargs='?', help="Path to the .npz file downloaded from the cluster")
    args = parser.parse_args()

    # 2. Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        print("Make sure you downloaded 'logmap_data.npz' from your cluster to this folder.")
        return

    print(f"Loading data from {args.file}...")
    try:
        data = np.load(args.file)
        # Extract data arrays
        V = data['vertices']
        F = data['faces']
        r = data['r']                 # Geodesic Distance
        theta = data['theta']         # Polar Angle
        logmap_uv = data['logmap_uv'] # 2D Log Map coordinates (u, v)
        vec_X = data['vectors_X']     # Parallel Transport Vectors
    except KeyError as e:
        print(f"Error loading data: Missing key {e}. Is the .npz file corrupted?")
        return

    # 3. Initialize Polyscope
    print("Initializing Polyscope...")
    ps.init()
    ps.set_up_dir("z_up") # Adjust if your mesh orientation is different

    # 4. Register the mesh
    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # 5. Add Quantities for Visualization
    
    # A. Scalar Fields (Distance and Angle)
    ps_mesh.add_scalar_quantity("1. Geodesic Distance (r)", r, enabled=True, cmap='turbo')
    ps_mesh.add_scalar_quantity("2. Polar Angle (theta)", theta, enabled=False, cmap='phase')

    # B. Vector Field (Parallel Transport)
    # This shows the vectors that were transported from the source without rotation
    ps_mesh.add_vector_quantity("3. Transported Vectors (X)", vec_X, length=0.01, enabled=False)

    # C. Parameterization (The Actual Log Map)
    # This applies a texture (checkerboard/grid) based on the computed (u,v) coordinates.
    # If the Log Map is correct, you will see a clean grid centered on the source.
    ps_mesh.add_parameterization_quantity("4. Log Map (UV)", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='grid') # Options: 'checkerboard', 'grid', 'local_check'

    # 6. Show the GUI
    print("Opening viewer window...")
    ps.show()

if __name__ == "__main__":
    main()
    