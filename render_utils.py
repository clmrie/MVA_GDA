import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os

def colorize_data(scalar_data, cmap_name='viridis', scale_factor=None):
    """Maps a scalar field to RGB colors (0-255)."""
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
    
    # Return uint8 colors
    return (rgba * 255).astype(np.uint8)


def render_to_file(trimesh_objects, camera_pose, filename, resolution=(800, 600)):
    """
    Robustly renders a list of trimesh objects to a file using pyrender.
    
    Args:
        trimesh_objects (list): List of trimesh.Trimesh objects (mesh, spheres, etc.)
        camera_pose (np.ndarray): 4x4 Homogenous camera matrix.
        filename (str): Output path.
    """
    # 1. Create a Pyrender Scene (Manual Construction)
    # This avoids the 'AttributeError' by not using trimesh.scene.Scene
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 1.0]))

    # 2. Add Geometry
    for tm_obj in trimesh_objects:
        # Convert Trimesh -> Pyrender Mesh
        mesh = pyrender.Mesh.from_trimesh(tm_obj)
        scene.add(mesh)

    # 3. Add Lighting (Crucial: without this, the mesh is black)
    # Directional light roughly from the camera position
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    
    # 4. Add Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=resolution[0]/resolution[1])
    scene.add(camera, pose=camera_pose)

    # 5. Render Offscreen
    try:
        r = pyrender.OffscreenRenderer(*resolution)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
        
        # Save using imageio
        import imageio.v3 as iio
        iio.imwrite(filename, color)
        
        r.delete()
        print(f"✅ Saved figure: {filename}")
        
    except Exception as e:
        print(f"🛑 CRITICAL RENDERING FAILURE for {filename}!")
        print(f"   Error: {e}")
        # On some clusters, you might need to raise this to see the stack trace in logs
        # raise e


# --- Standard Camera Views ---
# These assume the object is centered at (0,0,0) and roughly unit scale.
# You may need to adjust the translation (last column) if your mesh is huge/tiny.
DEFAULT_VIEWS = {
    'front': np.array([
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  2.5], # Moved back by 2.5 units
        [ 0.0,  0.0,  0.0,  1.0]
    ]),
    'side': np.array([
        [ 0.0,  0.0,  1.0,  2.5],
        [ 0.0,  1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ]),
    'iso': np.array([            # Isometric-ish view
        [ 0.707, -0.408,  0.577,  1.8],
        [ 0.0,    0.816,  0.577,  1.8],
        [-0.707, -0.408,  0.577,  1.8],
        [ 0.0,    0.0,    0.0,    1.0]
    ])
}