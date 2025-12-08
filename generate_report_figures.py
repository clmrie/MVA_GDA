import os
import numpy as np
import trimesh
import pyrender # Used for offscreen rendering
import matplotlib.pyplot as plt

from mesh import Mesh
from heat_method import heat_geodesic_from_sources
# Ensure you have implemented the vector_method.py file as discussed
from vector_method import vector_heat_transport 

# --- Camera Poses ---
# Define a few standard camera transformations (4x4 matrix) for fixed views.
# These values are standard trimesh/pyrender poses, adjust for your model.
FIXED_VIEWS = {
    'front': np.array([
        [ 0.0, -1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]]),
    'side': np.array([
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]]),
    'top': np.array([
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]])
}

# --- Utility Functions ---

def colorize_scalar_field(scalar_data, cmap_name='viridis'):
    """Maps a scalar field to RGB colors (0-255) using Matplotlib."""
    scalar_data = np.nan_to_num(scalar_data, nan=0.0)
    
    # Normalize data to [0, 1] range
    if scalar_data.max() == scalar_data.min():
        norm_data = np.zeros_like(scalar_data)
    else:
        norm_data = (scalar_data - scalar_data.min()) / (scalar_data.max() - scalar_data.min())

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm_data)
    
    # Scale to 0-255 and drop alpha
    return (rgba[:,:3] * 255).astype(np.uint8)


def render_to_file(scene, camera_pose, filename, resolution=(800, 600)):
    """Sets camera, renders, and saves the image."""
    
    # Set the camera transform
    scene.camera_transform = camera_pose
    
    # Use OffscreenRenderer for cluster (headless) environment
    r = pyrender.OffscreenRenderer(*resolution)
    
    # Render with white background
    color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    
    # Save image
    import imageio.v3 as iio
    iio.imwrite(filename, color)
    
    r.delete()
    print(f"Saved figure: {filename}")


# --- Main Figure Generation ---

def generate_log_map_figure(mesh_path):
    # Setup Paths
    out_dir = "results/report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Log Map Magnitude (Figure 2)...")
    M = Mesh.load(mesh_path)
    # Use a source far from centroid for a clear distance field
    src_idx = int(np.argmax(np.linalg.norm(M.V - M.V.mean(0), axis=1)))
    
    # 1. Compute Scalar Distance
    phi, _ = heat_geodesic_from_sources(M, src_idx)
    r = phi - phi.min() 
    
    # Convert mesh to Trimesh object
    tm = trimesh.Trimesh(M.V, M.F, process=False)
    
    # Colorize based on distance magnitude (Log Map magnitude)
    tm.visual.vertex_colors = colorize_scalar_field(r, cmap_name='plasma')
    
    # Create the scene
    scene = trimesh.scene.Scene(tm)
    
    # Render and save multiple angles
    for name, transform in FIXED_VIEWS.items():
        render_to_file(
            scene, 
            transform, 
            os.path.join(out_dir, f"figure2_logmap_magnitude_{name}.png")
        )

# NOTE: Vector rendering (arrows) is much harder in headless Trimesh/Pyrender.
# The simplest professional figure for the Log Map is the magnitude heatmap.


def generate_karcher_mean_figure(mesh_path):
    out_dir = "results/report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Karcher Mean Setup (Figure 3)...")
    M = Mesh.load(mesh_path)
    
    # 1. Define 3 target points (Red) and 1 center point (Green)
    np.random.seed(42)
    indices = np.random.choice(M.V.shape[0], size=3, replace=False)
    targets = M.V[indices]
    
    # Calculate a simplified (Euclidean) mean for visualization placement
    mean_approx = np.mean(targets, axis=0)

    # 2. Build the Scene Components (Mesh + Points)
    tm = trimesh.Trimesh(M.V, M.F, process=False)
    # Set mesh color to neutral gray/white
    tm.visual.face_colors = [200, 200, 200, 255]

    # Create spheres for the points (trimesh primitives)
    targets_spheres = [trimesh.primitives.Sphere(center=p, radius=0.015, subdivision=1) 
                       for p in targets]
    mean_sphere = trimesh.primitives.Sphere(center=mean_approx, radius=0.02, subdivision=1)

    # Assign colors
    for s in targets_spheres:
        s.visual.face_colors = [255, 0, 0, 255] # Red
    mean_sphere.visual.face_colors = [0, 255, 0, 255] # Green

    # Create the scene container
    scene = trimesh.scene.Scene([tm] + targets_spheres + [mean_sphere])

    # 3. Render and save the figure
    render_to_file(
        scene, 
        FIXED_VIEWS['front'], 
        os.path.join(out_dir, "figure3_karcher_setup.png")
    )


if __name__ == "__main__":
    # You MUST install pyrender and imageio on your cluster
    # pip install pyrender imageio[py3]
    
    # Use bunny or any mesh you have
    path = os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply")
    
    generate_log_map_figure(path)
    generate_karcher_mean_figure(path)