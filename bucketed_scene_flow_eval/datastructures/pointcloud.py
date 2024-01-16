import numpy as np
import open3d as o3d

from .se3 import SE3


def to_fixed_array(array: np.ndarray, max_len: int, pad_val=np.nan) -> np.ndarray:
    if len(array) > max_len:
        np.random.RandomState(len(array)).shuffle(array)
        sliced_pts = array[:max_len]
        return sliced_pts
    else:
        pad_tuples = [(0, max_len - len(array))]
        for _ in range(array.ndim - 1):
            pad_tuples.append((0, 0))
        return np.pad(array, pad_tuples, constant_values=pad_val)


def from_fixed_array(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        if len(array.shape) == 2:
            check_array = array[:, 0]
        elif len(array.shape) == 1:
            check_array = array
        else:
            raise ValueError(f"unknown array shape {array.shape}")
        are_valid_points = np.logical_not(np.isnan(check_array))
        are_valid_points = are_valid_points.astype(bool)
    else:
        import torch

        if len(array.shape) == 2:
            check_array = array[:, 0]
        elif len(array.shape) == 1:
            check_array = array
        else:
            raise ValueError(f"unknown array shape {array.shape}")
        are_valid_points = torch.logical_not(torch.isnan(check_array))
        are_valid_points = are_valid_points.bool()
    return array[are_valid_points]


def make_image_pixel_coordinate_grid(image_shape: tuple) -> np.ndarray:
    assert len(image_shape) == 2, f"image_shape must be a 2-tuple, got {image_shape}"
    # X positions repeated for each row
    x_positions = np.tile(np.arange(image_shape[1]), (image_shape[0], 1))
    # Y positions repeated for each column
    y_positions = np.tile(np.arange(image_shape[0]), (image_shape[1], 1)).T

    image_coordinates = (
        np.stack([x_positions, y_positions], axis=2).astype(np.float32).reshape(-1, 2)
    )
    return image_coordinates


def camera_to_world_coordiantes(points: np.ndarray) -> np.ndarray:
    world_T_camera = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    )
    return (world_T_camera @ points.T).T


class PointCloud:
    def __init__(self, points: np.ndarray) -> None:
        assert points.ndim == 2, f"points must be a 2D array, got {points.ndim}"
        assert points.shape[1] == 3, f"points must be a Nx3 array, got {points.shape}"
        self.points = points

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PointCloud):
            return False
        return np.allclose(self.points, o.points)

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self) -> str:
        return f"PointCloud with {len(self)} points"

    def __getitem__(self, idx):
        return self.points[idx]

    @staticmethod
    def from_depth_image(depth: np.ndarray, camera_projection: "CameraProjection") -> "PointCloud":
        assert depth.ndim == 2, f"depth must be a 2D array, got {depth.ndim}"
        image_coordinates = make_image_pixel_coordinate_grid(depth.shape)
        image_coordinate_depths = depth.reshape(-1, 1)

        points = camera_projection.to_camera(image_coordinates, image_coordinate_depths)
        finite_points = points[np.isfinite(points).all(axis=1)]
        return PointCloud(finite_points)

    @staticmethod
    def from_points_and_depth(
        image_coordinates: np.ndarray,
        image_coordinate_depths: np.ndarray,
        camera_projection: "CameraProjection",
    ) -> "PointCloud":
        return PointCloud(camera_projection.to_camera(image_coordinates, image_coordinate_depths))

    def transform(self, se3: SE3) -> "PointCloud":
        assert isinstance(se3, SE3), f"se3 must be an SE3, got {type(se3)}, expected {SE3}"
        return PointCloud(se3.transform_points(self.points))

    def transform_masked(self, se3: SE3, mask: np.ndarray) -> "PointCloud":
        assert isinstance(se3, SE3)
        assert mask.ndim == 1
        assert mask.shape[0] == len(self)
        updated_points = self.points.copy()
        updated_points[mask] = se3.transform_points(self.points[mask])
        return PointCloud(updated_points)

    def translate(self, translation: np.ndarray) -> "PointCloud":
        assert translation.shape == (3,)
        return PointCloud(self.points + translation)

    def flow(self, flow: np.ndarray) -> "PointCloud":
        assert (
            flow.shape == self.points.shape
        ), f"flow shape {flow.shape} must match point cloud shape {self.points.shape}"
        return PointCloud(self.points + flow)

    def flow_masked(self, flow: np.ndarray, mask: np.ndarray) -> "PointCloud":
        assert mask.ndim == 1, f"mask must be 1D, got {mask.ndim}"
        assert mask.dtype == bool, f"mask must be boolean, got {mask.dtype}"
        assert (
            self.points.shape[0] == mask.shape[0]
        ), f"mask must have same length as point cloud, got {mask.shape[0]} and {self.points.shape[0]}"
        # check that flow has the same number of entries as the boolean mask.
        assert (
            flow.shape[0] == mask.sum()
        ), f"flow must have same number of entries as the number of True values in the mask, got {flow.shape[0]} and {mask.sum()}"
        flow = flow.astype(np.float32)
        updated_points = self.points.copy()
        updated_points[mask] = self.points[mask] + flow
        return PointCloud(updated_points)

    def to_fixed_array(self, max_points: int) -> np.ndarray:
        return to_fixed_array(self.points, max_points)

    def matched_point_diffs(self, other: "PointCloud") -> np.ndarray:
        assert len(self) == len(other)
        return self.points - other.points

    def matched_point_distance(self, other: "PointCloud") -> np.ndarray:
        assert len(self) == len(other)
        return np.linalg.norm(self.matched_point_diffs(other), axis=1)

    @staticmethod
    def from_fixed_array(points) -> "PointCloud":
        return PointCloud(from_fixed_array(points))

    def to_array(self) -> np.ndarray:
        return self.points

    def copy(self) -> "PointCloud":
        return PointCloud(self.points.copy())

    def mask_points(self, mask: np.ndarray) -> "PointCloud":
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 1
        if mask.dtype == bool:
            assert mask.shape[0] == len(self)
        else:
            in_bounds = mask < len(self)
            assert np.all(
                in_bounds
            ), f"mask values must be in bounds, got {(~in_bounds).sum()} indices not in bounds out of {len(self)} points"

        return PointCloud(self.points[mask])

    def within_region_mask(self, x_min, x_max, y_min, y_max, z_min, z_max) -> np.ndarray:
        mask = np.logical_and(self.points[:, 0] < x_max, self.points[:, 0] > x_min)
        mask = np.logical_and(mask, self.points[:, 1] < y_max)
        mask = np.logical_and(mask, self.points[:, 1] > y_min)
        mask = np.logical_and(mask, self.points[:, 2] < z_max)
        mask = np.logical_and(mask, self.points[:, 2] > z_min)
        return mask

    def within_region(self, x_min, x_max, y_min, y_max, z_min, z_max) -> "PointCloud":
        mask = self.within_region_mask(x_min, x_max, y_min, y_max, z_min, z_max)
        return self.mask_points(mask)

    @property
    def shape(self) -> tuple:
        return self.points.shape

    def to_o3d(self) -> o3d.geometry.PointCloud:
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
