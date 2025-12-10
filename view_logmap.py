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

    # --- Manual Reset of Settings ---
    # Delete the persistent config file to force a clean view reset
    config_file = "imgui.ini"
    if os.path.exists(config_file):
        try:
            print(f"🧹 Deleting old config file ({config_file}) to force view reset.")
            os.remove(config_file)
        except Exception as e:
            print(f"⚠️  Warning: Could not delete {config_file}: {e}")
            
    # --- End Manual Reset ---

    print(f"📂 Loading data from {args.file}...")
    try:
        data = np.load(args.file)
        # Extract data arrays
        V = data['vertices']
        F = data['faces']
        r = data['r']
        theta = data['theta']
        logmap_uv = data['logmap_uv']
        vec_X = data['vectors_X']
        
        print(f"   - Vertices: {V.shape}")
        print(f"   - Faces:    {F.shape}")
    except KeyError as e:
        print(f"❌ Error loading data: Missing key {e}.")
        return

    # 3. Check for NaNs (as a final safety check)
    if np.any(np.isnan(V)) or np.any(np.isnan(logmap_uv)):
        print("❌ CRITICAL ERROR: Data contains NaNs. Computation failed.")
        return

    # 4. Initialize Polyscope (without the broken argument)
    print("🎨 Initializing Polyscope (Clean start)...")
    
    # We now call init() without any arguments, relying on the file deletion above.
    ps.init()
    
    ps.set_up_dir("z_up") 
    ps.set_ground_plane_mode("shadow_only")

    # 5. Register the mesh
    ps_mesh = ps.register_surface_mesh("LogMap Mesh", V, F)

    # 6. Add Quantities for Visualization
    
    # A. The Main Event: Log Map Parameterization (The Grid)
    ps_mesh.add_parameterization_quantity("4. Log Map Grid", logmap_uv, 
                                          coords_type='world', 
                                          enabled=True, 
                                          viz_style='grid',
                                          cmap='blues')

    # B. Scalar Fields (Distance and Angle) - DISABLED by default
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

if __name__ == "__main__":
    main()