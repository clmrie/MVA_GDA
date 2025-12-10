import numpy as np
import polyscope as ps
import argparse
import os
import sys

def main():
    # 1. Setup argument parser
    parser = argparse.ArgumentParser(description="Visualize Log Map from NPZ data")
    parser.add_argument('file', default='logmap_data.npz', nargs='?', 
                        help="Path to the .npz file downloaded from the cluster")
    args = parser.parse_args()

    # 2. Check if file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found.")
        print("   Make sure you are running this command in the same folder as the .npz file.")
        return

    print(f"📂 Loading data from {args.file}...")
    try:
        data = np.load(args.file)
        # Extract data arrays
        V = data['vertices']
        F = data['faces']
        r = data['r']                 # Geodesic Distance
        theta = data['theta']         # Polar Angle
        logmap_uv = data['logmap_uv'] # 2D Log Map coordinates (u, v)
        vec_X = data['vectors_X']     # Parallel Transport Vectors
        
        print(f"   - Vertices: {V.shape}")
        print(f"   - Faces:    {F.shape}")
    except KeyError as e:
        print(f"❌ Error loading data: Missing key {e}.")
        print("   Is the .npz file corrupted or from an old version of the script?")
        return

    # 3. Check for NaNs before crashing the viewer
    if np.any(np.isnan(V)) or np.any(np.isnan(logmap_uv)):
        print("❌ CRITICAL ERROR: The data contains NaNs (Not a Number).")
        print("   This means the computation on the cluster failed.")
        print("   Please re-run generate_logmap.py on the cluster with the fixed script.")
        return

    # 4. Initialize Polyscope (RESETTING SETTINGS)
    print("🎨 Initializing Polyscope (Resetting view)...")
    
    # We use a dummy config file name to force Polyscope to ignore old settings (imgui.ini)
    # This ensures the window starts fresh every time.
    ps.init(config_file="temp_reset_polyscope.ini")
    
    ps.set_up_dir("z_up") 
    ps.set_ground_plane_mode("shadow_only")

    # 5. Register the mesh
    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # 6. Add Quantities for Visualization
    
    # A. The Main Event: Log Map Parameterization (The Grid)
    # This applies a grid texture based on the (u,v) coordinates we computed.
    # We explicitly enable this one.
    ps_mesh.add_parameterization_quantity("4. Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='grid',
                                          cmap='blues')

    # B. Scalar Fields (Distance and Angle) - DISABLED by default
    # You can enable them manually in the UI if needed for the report.
    ps_mesh.add_scalar_quantity("1. Geodesic Distance (r)", r, enabled=False, cmap='turbo')
    ps_mesh.add_scalar_quantity("2. Polar Angle (theta)", theta, enabled=False, cmap='phase')

    # C. Vector Field (Parallel Transport) - DISABLED by default
    ps_mesh.add_vector_quantity("3. Transported Vectors (X)", vec_X, 
                                length=0.015, 
                                radius=0.001,
                                color=(0.2, 0.2, 0.2),
                                enabled=False)

    # 7. Show the GUI
    print("✅ Opening viewer window...")
    ps.show()
    
    # Cleanup dummy file
    if os.path.exists("temp_reset_polyscope.ini"):
        try:
            os.remove("temp_reset_polyscope.ini")
        except:
            pass

if __name__ == "__main__":
    main()