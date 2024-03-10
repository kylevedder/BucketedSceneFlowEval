from typing import Optional, Union

import numpy as np
import open3d as o3d

from .pointcloud import PointCloud
from .se3 import SE3


class O3DVisualizer:
    def __init__(self, point_size: float = 0.1):
        self.point_size = point_size
        self.geometry_list = []

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        self.add_geometry(world_frame)

    def add_geometry(self, geometry):
        if isinstance(geometry, list):
            for g in geometry:
                if "to_o3d" in dir(g):
                    g = g.to_o3d()
                self.add_geometry(g)
        else:
            self.geometry_list.append(geometry)

    def add_pc_frame(
        self,
        pc_frame: "PointCloudFrame",
        color: Union[tuple[float, float, float], None] = None,
    ):
        self.add_pointcloud(pc_frame.global_pc, color=color)

    def add_pointcloud(
        self,
        pc: PointCloud,
        pose: SE3 = SE3.identity(),
        color: Optional[
            Union[np.ndarray, tuple[float, float, float], list[tuple[float, float, float]]]
        ] = None,
    ):
        pc = pc.transform(pose)
        pc = pc.to_o3d()
        if color is not None:
            color = np.array(color)
            if color.ndim == 1:
                pc = pc.paint_uniform_color(color)
            elif color.ndim == 2:
                assert len(color) == len(
                    pc.points
                ), f"Expected color to have length {len(pc.points)}, got {len(color)} instead"
                pc.colors = o3d.utility.Vector3dVector(color)
        self.add_geometry(pc)

    def add_sphere(self, location: np.ndarray, radius: float, color: tuple[float, float, float]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=2)
        sphere = sphere.translate(location)
        sphere.paint_uniform_color(color)
        self.add_geometry(sphere)

    def add_spheres(
        self,
        locations: list[np.ndarray],
        radius: float,
        colors: list[tuple[float, float, float]],
    ):
        assert len(locations) == len(
            colors
        ), f"Expected locations and colors to have the same length, got {len(locations)} and {len(colors)} instead"
        triangle_mesh = o3d.geometry.TriangleMesh()
        for i, location in enumerate(locations):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=2)
            sphere = sphere.translate(location)
            sphere.paint_uniform_color(colors[i])
            triangle_mesh += sphere
        self.add_geometry(triangle_mesh)

    def add_pose(self, pose: SE3):
        self.add_geometry(pose.to_o3d(simple=True))

    def add_trajectories(self, points_array: np.ndarray):
        # points_array: (n_trajectories, n_points, 3)
        assert (
            points_array.ndim == 3
        ), f"Expected points_array to have shape (n_trajectories, n_points, 3), got {points_array.shape} instead"
        assert (
            points_array.shape[2] == 3
        ), f"Expected points_array to have shape (n_trajectories, n_points, 3), got {points_array.shape} instead"

        n_trajectories = points_array.shape[0]
        n_points_per_trajectory = points_array.shape[1]

        # trajectories are now in sequence
        flat_point_array = points_array.reshape(-1, 3)

        n_to_np1_array = np.array(
            [
                np.arange(n_trajectories * n_points_per_trajectory),
                np.arange(n_trajectories * n_points_per_trajectory) + 1,
            ]
        ).T
        keep_mask = np.ones(len(n_to_np1_array), dtype=bool)
        keep_mask[(n_points_per_trajectory - 1) :: n_points_per_trajectory] = False

        # print(n_to_np1_array)
        # print(keep_mask)
        flat_index_array = n_to_np1_array[keep_mask]

        # print(flat_point_array)
        # print(flat_index_array)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(flat_point_array)
        line_set.lines = o3d.utility.Vector2iVector(flat_index_array)
        # line_set.colors = o3d.utility.Vector3dVector(
        #     np.tile(np.array(color), (len(trajectory) - 1, 1)))
        self.add_geometry(line_set)

    def add_trajectory(
        self,
        trajectory: list[np.ndarray],
        color: tuple[float, float, float],
        radius: float = 0.05,
    ):
        for i in range(len(trajectory) - 1):
            self.add_sphere(trajectory[i], radius, color)

        # Add line set between trajectory points
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(len(trajectory) - 1)])
        )
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color), (len(trajectory) - 1, 1))
        )
        self.add_geometry(line_set)

    def run(self):
        print("Running visualizer on geometry list of length", len(self.geometry_list))
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Benchmark Visualizer")
        vis.get_render_option().point_size = self.point_size

        for geometry in self.geometry_list:
            vis.add_geometry(geometry)

        vis.run()

        #
        # o3d.visualization.draw_geometries(self.geometry_list)
        # ctr = self.vis.get_view_control()
        # # Set forward direction to be -X
        # ctr.set_front([-1, 0, 0])
        # # Set up direction to be +Z
        # ctr.set_up([0, 0, 1])
        # # Set lookat to be origin
        # ctr.set_lookat([0, 0, 0])
