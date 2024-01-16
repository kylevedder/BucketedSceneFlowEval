from enum import Enum

import numpy as np

from .pointcloud import PointCloud
from .rgb_image import RGBImage


class CameraModel(Enum):
    PINHOLE = 1
    FIELD_OF_VIEW = 2


class CameraProjection:
    """
    There are three coordinate frames:

    - Pixel Space (positive X down, positive Y right)
    - View Space (positive X down, positive Y right, positive Z forward)
    - Camera Space (Right Hand Rule, positive X forward, positive Y left, positive Z up)
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float, camera_model: CameraModel):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_model = camera_model

    def __repr__(self) -> str:
        return f"CameraProjection(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, camera_model={self.camera_model})"

    def image_to_image_plane_pc(
        self, image: RGBImage, depth: float = 1.0
    ) -> tuple[PointCloud, np.ndarray]:
        # Make pixel coordinate grid
        image_shape = image.image.shape[:2]
        image_coordinates = (
            np.stack(
                np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0])),
                axis=2,
            )
            .astype(np.float32)
            .reshape(-1, 2)
        )
        image_coordinate_depths = np.ones((len(image_coordinates), 1)) * depth

        resulting_points = self.to_camera(image_coordinates, image_coordinate_depths)
        colors = image.image.reshape(-1, 3)
        return PointCloud(resulting_points), colors

    def _camera_to_view_coordinates(self, camera_points: np.ndarray):
        assert (
            len(camera_points.shape) == 2
        ), f"camera_points must have shape (N, 3), got {camera_points.shape}"
        assert (
            camera_points.shape[1] == 3
        ), f"camera_points must have shape (N, 3), got {camera_points.shape}"
        camera_T_view = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )

        return camera_points @ camera_T_view

    def _view_to_camera_coordinates(self, view_points: np.ndarray):
        assert (
            len(view_points.shape) == 2
        ), f"view_points must have shape (N, 2), got {view_points.shape}"
        assert (
            view_points.shape[1] == 3
        ), f"view_points must have shape (N, 3), got {view_points.shape}"
        view_T_camera = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ]
        )

        return view_points @ view_T_camera

    def view_frame_to_pixels(self, view_points: np.ndarray):
        """
        Input: camera_frame_ego_points of shape (N, 3)

        Expects the view frame ego points to be in sensor coordinates, with
        the sensor looking down the positive Z axis, positive X being right,
        and positive Y being down.

        Output: image_points of shape (N, 2)

        The image frame is defined as follows:
        0,0 is the top left corner
        """

        assert (
            len(view_points.shape) == 2
        ), f"view_points must have shape (N, 3), got {view_points.shape}"
        assert (
            view_points.shape[1] == 3
        ), f"view_points must have shape (N, 3), got {view_points.shape}"

        K = np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ]
        )

        pixel_points_3d = view_points @ K.T
        pixel_points_2d = pixel_points_3d[:, :2] / pixel_points_3d[:, 2:]

        return pixel_points_2d

    def camera_frame_to_pixels(self, camera_points: np.ndarray):
        """
        Input: camera_frame_ego_points of shape (N, 3)

        Expects the camera frame ego points to be in right hand coordinates, with
        the camera looking down the positive X axis.

        Output: image_points of shape (N, 2)

        The image frame is defined as follows:
        0,0 is the top left corner
        """
        assert (
            len(camera_points.shape) == 2
        ), f"camera_points must have shape (N, 3), got {camera_points.shape}"
        assert (
            camera_points.shape[1] == 3
        ), f"camera_points must have shape (N, 3), got {camera_points.shape}"

        view_points = self._camera_to_view_coordinates(camera_points)
        return self.view_frame_to_pixels(view_points)

    def to_camera(self, pixel_coordinates, pixel_coordinate_depths):
        """
        Input: pixel_coordinates of shape (N, 2)

        Expects the pixel coordinates to have the origin at the top left corner.

        Output: camera_points of shape (N, 3)

        Camera frame is right hand coordinates (standard robotics coordinates):
        X is forward, Y is left, Z is up
        """
        if self.camera_model == CameraModel.PINHOLE:
            return self._points_and_depth_to_3d_pinhole(pixel_coordinates, pixel_coordinate_depths)
        elif self.camera_model == CameraModel.FIELD_OF_VIEW:
            return self._points_and_depth_to_3d_fov(pixel_coordinates, pixel_coordinate_depths)
        else:
            raise NotImplementedError(f"Camera model {self.camera_model} not implemented")

    def _points_and_depth_to_3d_pinhole(
        self, pixel_coordinates: np.ndarray, pixel_coordinate_depths: np.ndarray
    ) -> np.ndarray:
        assert (
            pixel_coordinates.ndim == 2
        ), f"pixel_coordinates must be a 2D array, got {pixel_coordinates.ndim}"
        assert (
            pixel_coordinates.shape[1] == 2
        ), f"pixel_coordinates must be a Nx2 array, got {pixel_coordinates.shape}"

        assert (
            pixel_coordinate_depths.ndim == 2
        ), f"depth must be a 2D array, got {pixel_coordinate_depths.ndim}"
        assert (
            pixel_coordinate_depths.shape[1] == 1
        ), f"depth must be a Nx1 array, got {pixel_coordinate_depths.shape}"

        assert (
            pixel_coordinates.shape[0] == pixel_coordinate_depths.shape[0]
        ), f"number of points in pixel_coordinates {pixel_coordinates.shape[0]} must match number of points in image_coordinate_depths {pixel_coordinate_depths.shape[0]}"

        # Standard camera intrinsics matrix
        K = np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ]
        )

        # These points are at the pixel locations of the image.
        pixel_coordinate_points = np.concatenate(
            [pixel_coordinates, np.ones((len(pixel_coordinates), 1))], axis=1
        )

        # Camera plane is the plane of ray points with a depth of 1 in the view coordinate frame.
        camera_plane_points = pixel_coordinate_points @ np.linalg.inv(K.T)

        # Multiplying by the depth scales the ray of each point in the view coordinate frame to
        # the distance measured by the depth image.
        view_points = camera_plane_points * pixel_coordinate_depths

        return self._view_to_camera_coordinates(view_points)

    def _points_and_depth_to_3d_ndc_fov(
        self,
        ndc_coordinates: np.ndarray,
        ndc_coordinate_depths: np.ndarray,
        ndc_fx: float,
        ndc_fy: float,
    ) -> np.ndarray:
        assert (
            ndc_coordinates.ndim == 2
        ), f"ndc_coordinates must be a 2D array, got {ndc_coordinates.ndim}"
        assert (
            ndc_coordinates <= 1.0
        ).all(), f"ndc_coordinates must be in NDC space (<= 1), got {ndc_coordinates}"
        assert (
            ndc_coordinates >= 0.0
        ).all(), f"ndc_coordinates must be in NDC space (>= 0), got {ndc_coordinates}"

        ndc_coordinate_points = np.concatenate(
            [ndc_coordinates, np.ones((len(ndc_coordinates), 1))], axis=1
        )

        # Camera intrinsics matrix converted to Normalized Device Coordinates (NDC)
        K = np.array(
            [
                [ndc_fx, 0, 0.5],
                [0, ndc_fy, 0.5],
                [0, 0, 1],
            ]
        )

        # Camera plane is the plane of ray points with a depth of 1 in the view coordinate frame.
        camera_plane_points = ndc_coordinate_points @ np.linalg.inv(K.T)

        # Normalize the ray vectors to be unit length to form the camera plane.
        # This is the essential difference between a pinhole camera and a field of view camera.
        camera_ball_points = camera_plane_points / np.linalg.norm(
            camera_plane_points, axis=1, keepdims=True
        )

        # Multiplying by the depth scales the ray of each point in the view frame to
        # the distance measured by the depth image.
        view_points = camera_ball_points * ndc_coordinate_depths

        return self._view_to_camera_coordinates(view_points)

    def _points_and_depth_to_3d_fov(
        self, image_coordinates: np.ndarray, depths: np.ndarray
    ) -> np.ndarray:
        assert (
            image_coordinates.ndim == 2
        ), f"image_space_points must be a 2D array, got {image_coordinates.ndim}"
        assert (
            image_coordinates.shape[1] == 2
        ), f"image_space_points must be a Nx2 array, got {image_coordinates.shape}"

        assert depths.ndim == 2, f"depth must be a 2D array, got {depths.ndim}"
        assert depths.shape[1] == 1, f"depth must be a Nx1 array, got {depths.shape}"

        assert (
            image_coordinates.shape[0] == depths.shape[0]
        ), f"number of points in image_coordinates {image_coordinates.shape[0]} must match number of points in image_coordinate_depths {depths.shape[0]}"

        image_shape = (self.cx * 2, self.cy * 2)
        ndc_fx = self.fx / image_shape[1]
        ndc_fy = self.fy / image_shape[0]

        # Convert from pixels to raster space with the + 0.5, then to NDC space
        ndc_coordinates = (image_coordinates + 0.5) / np.array(image_shape)[None, :]

        return self._points_and_depth_to_3d_ndc_fov(ndc_coordinates, depths, ndc_fx, ndc_fy)
