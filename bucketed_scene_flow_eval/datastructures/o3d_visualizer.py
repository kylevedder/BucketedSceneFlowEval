import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import open3d as o3d

from .dataclasses import EgoLidarFlow, PointCloudFrame, RGBFrame, VectorArray
from .line_mesh import LineMesh
from .pointcloud import PointCloud
from .se3 import SE3

ColorType = Union[np.ndarray, tuple[float, float, float], list[tuple[float, float, float]]]


class O3DVisualizer:
    def __init__(
        self, point_size: float = 0.1, line_width: float = 1.0, add_world_frame: bool = True
    ):
        self.point_size = point_size
        self.line_width = line_width
        self.geometry_list = []

        if add_world_frame:
            self.add_world_frame()

    def add_world_frame(self):
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

    def add_global_pc_frame(
        self,
        pc_frame: PointCloudFrame,
        color: Optional[ColorType] = None,
    ):
        self.add_pointcloud(pc_frame.global_pc, color=color)

    def _paint_o3d_color(self, o3d_geom, color: ColorType):
        assert color is not None, "Expected color to be not None"
        color = np.array(color)
        if color.ndim == 1:
            o3d_geom.paint_uniform_color(color)
        elif color.ndim == 2:
            assert len(color) == len(
                o3d_geom.points
            ), f"Expected color to have length {len(o3d_geom.points)}, got {len(color)} instead"
            o3d_geom.colors = o3d.utility.Vector3dVector(color)

    def add_lineset(
        self,
        p1s: Union[VectorArray, PointCloud],
        p2s: Union[VectorArray, PointCloud],
        color: Optional[ColorType] = None,
    ):
        # Convert to PointClouds
        if isinstance(p1s, np.ndarray):
            # Ensure it's Nx3
            assert (
                p1s.ndim == 2 and p1s.shape[1] == 3
            ), f"Expected p1s to be a Nx3 array, got {p1s.shape} instead"
            p1s = PointCloud(p1s)
        if isinstance(p2s, np.ndarray):
            # Ensure it's Nx3
            assert (
                p2s.ndim == 2 and p2s.shape[1] == 3
            ), f"Expected p2s to be a Nx3 array, got {p2s.shape} instead"
            p2s = PointCloud(p2s)

        assert len(p1s) == len(
            p2s
        ), f"Expected p1s and p2s to have the same length, got {len(p1s)} and {len(p2s)} instead"

        # Convert to o3d
        p1s_o3d = p1s.to_o3d()
        p2s_o3d = p2s.to_o3d()

        corrispondences = [(i, i) for i in range(len(p1s))]
        lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            p1s_o3d, p2s_o3d, corrispondences
        )
        if color is not None:
            self._paint_o3d_color(lineset, color)

        self.add_geometry(lineset)

    def add_global_flow(
        self, pc_frame: PointCloudFrame, ego_flow: EgoLidarFlow, color: Optional[ColorType] = None
    ):
        # Add lineset for flow vectors
        ego_pc1 = pc_frame.full_pc.mask_points(ego_flow.mask)
        ego_p2 = ego_pc1.flow(ego_flow.valid_flow)
        global_pc1 = ego_pc1.transform(pc_frame.global_pose)
        global_pc2 = ego_p2.transform(pc_frame.global_pose)
        # self.add_pointcloud(global_pc1, color=(0, 1, 0))
        # self.add_pointcloud(global_pc2, color=(0, 0, 1))
        self.add_lineset(global_pc1, global_pc2, color=color)

    def add_global_rgb_frame(self, rgb_frame: RGBFrame):
        image_plane_pc, colors = rgb_frame.camera_projection.image_to_image_plane_pc(
            rgb_frame.rgb, depth=20
        )
        image_plane_pc = image_plane_pc.transform(rgb_frame.pose.sensor_to_global)
        self.add_pointcloud(image_plane_pc, color=colors)

    def add_pointcloud(
        self,
        pc: PointCloud,
        pose: SE3 = SE3.identity(),
        color: Optional[ColorType] = None,
    ):
        pc = pc.transform(pose)
        pc_o3d = pc.to_o3d()
        if color is not None:
            self._paint_o3d_color(pc_o3d, color)
        self.add_geometry(pc_o3d)

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

        points = o3d.utility.Vector3dVector(trajectory)
        lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(len(trajectory) - 1)])
        )
        colors = o3d.utility.Vector3dVector(np.tile(np.array(color), (len(trajectory) - 1, 1)))

        line_mesh = LineMesh(points=points, lines=lines, colors=colors, radius=self.line_width / 20)
        self.add_geometry(line_mesh.cylinder_segments)

    def render(self, vis, reset_view: bool = True):
        for geometry in self.geometry_list:
            vis.add_geometry(geometry, reset_bounding_box=reset_view)

    def run(self, vis=o3d.visualization.Visualizer()):
        print("Running visualizer on geometry list of length", len(self.geometry_list))
        vis.create_window(window_name="Benchmark Visualizer")

        ro = vis.get_render_option()
        ro.point_size = self.point_size

        self.render(vis)

        vis.run()


class O3DCallbackVisualizer(O3DVisualizer):
    def __init__(self, screenshot_path: Path = Path() / "screenshots", *args, **kwargs):
        self.screenshot_path = screenshot_path
        super().__init__(*args, **kwargs)

    def _get_screenshot_path(self) -> Path:
        return self.screenshot_path / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    def save_screenshot(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        save_name = self._get_screenshot_path()
        save_name.parent.mkdir(exist_ok=True, parents=True)
        vis.capture_screen_image(str(save_name))

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        vis.register_key_callback(ord("S"), self.save_screenshot)

    def run(self, vis=o3d.visualization.VisualizerWithKeyCallback()):
        return super().run(vis)
