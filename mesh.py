# src/mesh.py
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class Mesh:
    """Tiny mesh wrapper for loading + visualizing with meshplot."""
    V: np.ndarray  
    F: np.ndarray  

    @classmethod
    def load(
        cls,
        path: str,
        process: bool = True,
        recenter: bool = True,
        rescale_unit: bool = True,
    ) -> "Mesh":
        """
        Load a surface mesh via trimesh, ensure triangles, recenter at origin,
        and rescale to unit size (so all models have comparable scale).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        obj = trimesh.load(path, process=process)

        
        if isinstance(obj, trimesh.Scene):
            if len(obj.geometry) == 0:
                raise ValueError("Scene contains no geometry.")
            tm = trimesh.util.concatenate(tuple(obj.dump()))
        elif isinstance(obj, trimesh.Trimesh):
            tm = obj
        else:
            raise TypeError(f"Unsupported type from trimesh.load: {type(obj)}")

        if tm.faces is None or len(tm.faces) == 0:
            raise ValueError("Loaded geometry has no faces (is it a point cloud?)")

        if tm.faces.shape[1] != 3:
            tm = tm.triangulate()

        translation = -tm.centroid if recenter else np.zeros(3)
        if rescale_unit:
            if tm.scale == 0:
                raise ValueError("Degenerate geometry with zero scale.")
            scale = 1.0 / float(tm.scale)
        else:
            scale = 1.0

        V = (tm.vertices + translation) * scale
        F = tm.faces.astype(np.int32, copy=False)

        return cls(V=V.astype(np.float64, copy=False), F=F)
