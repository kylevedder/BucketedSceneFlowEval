from pathlib import Path
from typing import Optional, Union

import meshio
import numpy as np
import open3d as o3d
import pyviz3d.visualizer as pyviz

from .o3d_visualizer import O3DVisualizer
from .pointcloud import PointCloud
from .se3 import SE3


class PyVisVisualizer(O3DVisualizer):
    def __init__(self, point_size: float = 0.1, save_cache_dir: Path = Path("/tmp/pyvis3d_cache")):
        super().__init__(point_size=point_size)
        self.save_cache_dir = save_cache_dir

    def _process_geomerty(
        self,
        idx: int,
        geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet],
        vis: pyviz.Visualizer,
    ):
        if isinstance(geometry, o3d.geometry.PointCloud):
            points = np.asarray(geometry.points)
            colors = np.asarray(geometry.colors)
            if colors.shape[0] == 0:
                colors = None
            else:
                # Convert from 0-1 to 0-255
                colors = (colors * 255).astype(np.uint8)
            vis.add_points(name=f"points_{idx:03d}", positions=points, colors=colors)
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            raw_path = self.save_cache_dir / f"mesh_{idx:010d}.obj"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(raw_path), geometry)
            vis.add_mesh(
                f"mesh_{idx:03d}", path=str(raw_path), rotation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        elif isinstance(geometry, o3d.geometry.LineSet):
            return
        else:
            raise ValueError(f"Unsupported geometry type {type(geometry)}")

    def _save_geometry_entry(self, idx: int, geometry) -> Path:
        raw_path = self.save_cache_dir / f"mesh_{idx:010d}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(geometry, o3d.geometry.PointCloud):
            extension = ".ply"
            path = raw_path.with_suffix(extension)
            o3d.io.write_point_cloud(str(path), geometry)
            # Covert to obj
            meshio.read(str(path)).write(str(path.with_suffix(".obj")))
            return path.with_suffix(".obj")
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            extension = ".obj"
            path = raw_path.with_suffix(extension)
            o3d.io.write_triangle_mesh(str(path), geometry)
            return path
        elif isinstance(geometry, o3d.geometry.LineSet):
            extension = ".ply"
            path = raw_path.with_suffix(extension)
            o3d.io.write_line_set(str(path), geometry)
            # Covert to obj
            meshio.read(str(path)).write(str(path.with_suffix(".obj")))
            return path.with_suffix(".obj")
        else:
            raise ValueError(f"Unsupported geometry type {type(geometry)}")

    def _load_mesh(self, path: Path, vis: pyviz.Visualizer):
        name = path.stem
        vis.add_mesh(name, path=str(path), rotation=np.array([0.0, 0.0, 0.0, 1.0]))

    def run(self):
        """
        Convert every geometry entry to an obj file, then load those obj files as meshes into the visualizer.
        """

        vis = pyviz.Visualizer()

        for i, geometry in enumerate(self.geometry_list):
            self._process_geomerty(i, geometry, vis)

        vis.save("pyvis", port=6008)

        #
        # o3d.visualization.draw_geometries(self.geometry_list)
        # ctr = self.vis.get_view_control()
        # # Set forward direction to be -X
        # ctr.set_front([-1, 0, 0])
        # # Set up direction to be +Z
        # ctr.set_up([0, 0, 1])
        # # Set lookat to be origin
        # ctr.set_lookat([0, 0, 0])
