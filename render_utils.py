import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mesh import Mesh # Re-use your wrapper

def colorize_data(scalar_data, cmap_name='viridis', scale_factor=None):
    """Maps a scalar field to RGB colors using a Matplotlib colormap."""
    scalar_data = np.nan_to_num(scalar_data, nan=0.0)
    
    # Normalize data to [0, 1] range
    if scale_factor is None:
        if scalar_data.max() == scalar_data.min():
            norm_data = np.zeros_like(scalar_data)
        else:
            norm_data = (scalar_data - scalar_data.min()) / (scalar_data.max() - scalar_data.min())
    else:
        norm_data = np.clip(scalar_data / scale_factor, 0, 1)

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm_data)
    
    # Scale to 0-255 for Trimesh/Rendering
    return (rgba * 255).astype(np.uint8)

def render_and_save_mesh(mesh: Mesh, scalar_field: np.ndarray, base_filename: str, view_angles: dict):
    """
    Renders the mesh with a scalar field colored from multiple angles.
    
    :param mesh: The Mesh object.
    :param scalar_field: (N,) array of scalar values to color the mesh.
    :param base_filename: e.g., 'logmap'
    :param view_angles: Dict {view_name: camera_transform_matrix}
    """
    V, F = mesh.V, mesh.F
    
    # Convert your custom Mesh object to a Trimesh object
    tm = trimesh.Trimesh(V, F, process=False)
    
    # Colorize the mesh based on the scalar field (e.g., geodesic distance)
    tm.visual.vertex_colors = colorize_data(scalar_field)

    # Create the scene container
    scene = tm.scene()
    
    for name, transform in view_angles.items():
        # Apply the camera transformation for the scene
        scene.camera_transform = transform
        
        # Render the image and save to file
        image = scene.save_image(resolution=(800, 600), background=[255, 255, 255, 255])
        
        output_path = f"{base_filename}_{name}.png"
        
        # Save image (needs to be written to disk)
        with open(output_path, 'wb') as f:
            f.write(image)
            
        print(f"Saved figure: {output_path}")

# --- Example Camera Poses (Customize these!) ---
# These are Trimesh-compatible 4x4 homogenous matrices
# You can find these by running trimesh interactively once locally and reading tm.camera_transform
DEFAULT_VIEWS = {
    'front': np.array([
        [ 0.0,  0.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]]),
    'side': np.array([
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]]),
    'top': np.array([
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0]])
}
