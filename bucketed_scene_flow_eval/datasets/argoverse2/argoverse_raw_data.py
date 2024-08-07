import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from bucketed_scene_flow_eval.datastructures import (
    SE2,
    SE3,
    CameraModel,
    CameraProjection,
    MaskArray,
    PointCloud,
    PointCloudFrame,
    PoseInfo,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
    RGBImageCrop,
    TimeSyncedAVLidarData,
    TimeSyncedRawFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractSequence, CachedSequenceLoader
from bucketed_scene_flow_eval.utils import load_json

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters


PointCloudRange = tuple[float, float, float, float, float, float]

DEFAULT_POINT_CLOUD_RANGE: PointCloudRange = (
    -48,
    -48,
    -2.5,
    48,
    48,
    2.5,
)


class ImageOp(ABC):
    @abstractmethod
    def apply(self, rgb_frame: RGBFrame) -> RGBFrame:
        raise NotImplementedError


class Resize(ImageOp):
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def apply(self, rgb_frame: RGBFrame) -> RGBFrame:
        current_long_size = max(rgb_frame.rgb.shape[0], rgb_frame.rgb.shape[1])
        target_long_size = max(self.target_shape[0], self.target_shape[1])
        reduction_factor = current_long_size / target_long_size
        rgb_frame.rgb = rgb_frame.rgb.rescale(reduction_factor)
        rgb_frame.camera_projection = rgb_frame.camera_projection.rescale(reduction_factor)
        return rgb_frame


class EmbedTranspose(ImageOp):
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def apply(self, rgb_frame: RGBFrame) -> RGBFrame:
        # Ensure image is a transpose of the target shape
        assert (rgb_frame.rgb.shape[1] == self.target_shape[0]) and (
            rgb_frame.rgb.shape[0] == self.target_shape[1]
        ), f"Image shape {rgb_frame.rgb.shape} is not a transpose of the target shape {self.target_shape}"
        # Image shape is transpose of the target shape.
        # Extract the overlapping center of the two images and embed it in the target array.
        src_array = rgb_frame.rgb.full_image
        tgt_array = np.zeros(self.target_shape, dtype=src_array.dtype)

        # Calculate the center of both source and target
        src_center_y, src_center_x = src_array.shape[0] // 2, src_array.shape[1] // 2
        tgt_center_y, tgt_center_x = tgt_array.shape[0] // 2, tgt_array.shape[1] // 2

        # Determine the dimensions of the region to extract from the source
        # This is based on the minimum of the source's and target's dimensions
        extract_height = min(src_array.shape[0], tgt_array.shape[0])
        extract_width = min(src_array.shape[1], tgt_array.shape[1])

        # Calculate start and end indices for extraction from the source
        src_start_y = max(0, src_center_y - extract_height // 2)
        src_end_y = src_start_y + extract_height
        src_start_x = max(0, src_center_x - extract_width // 2)
        src_end_x = src_start_x + extract_width

        # Calculate start and end indices for insertion into the target
        tgt_start_y = max(0, tgt_center_y - extract_height // 2)
        tgt_end_y = tgt_start_y + extract_height
        tgt_start_x = max(0, tgt_center_x - extract_width // 2)
        tgt_end_x = tgt_start_x + extract_width

        # Extract the region from the source image
        extracted_region = src_array[src_start_y:src_end_y, src_start_x:src_end_x, :]

        # Place the extracted region into the target array, centered
        tgt_array[tgt_start_y:tgt_end_y, tgt_start_x:tgt_end_x, :] = extracted_region
        valid_crop = RGBImageCrop(
            min_x=tgt_start_x, max_x=tgt_end_x, min_y=tgt_start_y, max_y=tgt_end_y
        )

        rgb_frame.rgb = RGBImage(tgt_array, valid_crop)
        rgb_frame.camera_projection = rgb_frame.camera_projection.transpose()
        return rgb_frame


@dataclass(kw_only=True)
class CameraInfo:
    rgb_frame_paths: list[Path]
    rgb_timestamp_to_rgb_file_map: dict[int, Path]
    timestamp_to_rgb_timestamp_map: dict[int, int]
    rgb_camera_projection: CameraProjection
    rgb_camera_ego_pose: SE3
    expected_shape: tuple[int, int, int]

    def timestamp_to_rgb_path(self, timestamp: int) -> Path:
        assert timestamp in self.timestamp_to_rgb_timestamp_map, f"timestamp {timestamp} not found"
        rgb_timestamp = self.timestamp_to_rgb_timestamp_map[timestamp]
        assert (
            rgb_timestamp in self.rgb_timestamp_to_rgb_file_map
        ), f"rgb_timestamp {rgb_timestamp} not found"
        return self.rgb_timestamp_to_rgb_file_map[rgb_timestamp]

    def load_rgb(self, timestamp: int) -> RGBImage:
        rgb_path = self.timestamp_to_rgb_path(timestamp)
        # Read the image, keep the same color space
        raw_img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        assert (
            raw_img.shape[2] == self.expected_shape[2]
        ), f"expected {self.expected_shape[2]} channels, got {raw_img.shape[2]} for {rgb_path}"
        # Convert from CV2 standard BGR to RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        return RGBImage(raw_img)

    def _reshaped_frame(self, rgb_frame: RGBFrame) -> RGBFrame:
        """
        Reshape the image to the expected shape,
        and update the camera projection to account for the scaling.
        """
        current_area = rgb_frame.rgb.shape[0] * rgb_frame.rgb.shape[1]
        expected_area = self.expected_shape[0] * self.expected_shape[1]

        if expected_area < current_area:
            rgb_frame = Resize(self.expected_shape).apply(rgb_frame)

        current_aspect_ratio = rgb_frame.rgb.shape[1] / rgb_frame.rgb.shape[0]
        target_aspect_ratio = self.expected_shape[1] / self.expected_shape[0]

        if abs(current_aspect_ratio - target_aspect_ratio) > 0.1:
            rgb_frame = EmbedTranspose(self.expected_shape).apply(rgb_frame)

        return rgb_frame

    def load_rgb_frame(self, timestamp: int, global_pose: SE3) -> RGBFrame:
        rgb = self.load_rgb(timestamp)
        rgb_frame = RGBFrame(
            rgb=rgb,
            pose=PoseInfo(sensor_to_ego=self.rgb_camera_ego_pose, ego_to_global=global_pose),
            camera_projection=self.rgb_camera_projection,
        )
        return self._reshaped_frame(rgb_frame)


class RangeCropType(enum.Enum):
    GLOBAL = "global"
    EGO = "ego"


class ArgoverseRawSequence(AbstractSequence):
    """
    Argoverse Raw Sequence.

    Every sequence is a collection of frames. Unfortunately, RGB cameras are not perfectly
    synced with the lidar frames, so we have to pull the most recent RGB image as a first order
    approximation of the RGB at the given timestamp.
    """

    def __init__(
        self,
        log_id: str,
        dataset_dir: Path,
        verbose: bool = False,
        with_rgb: bool = False,
        with_auxillary_pc: bool = False,
        point_cloud_range: Optional[PointCloudRange] = DEFAULT_POINT_CLOUD_RANGE,
        range_crop_type: RangeCropType | str = RangeCropType.GLOBAL,
        sample_every: Optional[int] = None,
        camera_names: list[str] = [
            "ring_side_left",
            "ring_front_left",
            "ring_front_center",
            "ring_front_right",
            "ring_side_right",
        ],
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
    ):
        self.log_id = log_id
        self.point_cloud_range = point_cloud_range

        if isinstance(range_crop_type, str):
            range_crop_type = RangeCropType(range_crop_type.lower())
        self.range_crop_type = range_crop_type

        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.is_dir(), f"dataset_dir {dataset_dir} does not exist"

        # Load the vehicle pose information.
        self.frame_infos = pd.read_feather(self.dataset_dir / "city_SE3_egovehicle.feather")
        self.timestamp_to_info_idx_map = {
            int(timestamp): idx
            for idx, timestamp in enumerate(self.frame_infos["timestamp_ns"].values)
        }
        info_timestamps = set(self.timestamp_to_info_idx_map.keys())

        (
            self.lidar_frame_paths,
            self.timestamp_to_lidar_file_map,
            self.lidar_file_timestamps,
        ) = self._load_lidar_info()

        self.with_rgb = with_rgb
        self.with_auxillary_pc = with_auxillary_pc

        self.camera_names = camera_names
        if not with_rgb:
            camera_names = []
            self.camera_names = []

        self.camera_info_lookup: dict[str, CameraInfo] = {
            camera_name: self._prep_camera_info(camera_name, expected_camera_shape)
            for camera_name in camera_names
        }

        self.timestamp_list = sorted(self.lidar_file_timestamps.intersection(info_timestamps))
        assert len(self.timestamp_list) > 0, f"no timestamps found in {self.dataset_dir}"

        self.auxillary_pc_paths = self._load_auxillary_pc_info()

        if sample_every is not None:
            self.timestamp_list = self.timestamp_list[::sample_every]

        (
            self.raster_heightmap,
            self.global_to_raster_se2,
            self.global_to_raster_scale,
        ) = self._load_ground_height_raster()

        if verbose:
            print(
                f"Loaded {len(self.timestamp_list)} frames from {self.dataset_dir} at timestamp {time.time():.3f}"
            )

    def _prep_camera_info(self, camera_name: str, camera_shape: tuple[int, int, int]) -> CameraInfo:
        (
            rgb_frame_paths,
            rgb_timestamp_to_rgb_file_map,
        ) = self._load_rgb_info(camera_name)

        rgb_camera_projection = self._load_camera_projection(camera_name)
        rgb_camera_ego_pose = self._load_camera_ego_pose(camera_name)

        timestamp_to_rgb_timestamp_map = {
            lidar_timestamp: min(
                rgb_timestamp_to_rgb_file_map.keys(),
                key=lambda rgb_timestamp: abs(rgb_timestamp - lidar_timestamp),
            )
            for lidar_timestamp in self.lidar_file_timestamps
        }

        return CameraInfo(
            rgb_frame_paths=rgb_frame_paths,
            rgb_timestamp_to_rgb_file_map=rgb_timestamp_to_rgb_file_map,
            timestamp_to_rgb_timestamp_map=timestamp_to_rgb_timestamp_map,
            rgb_camera_projection=rgb_camera_projection,
            rgb_camera_ego_pose=rgb_camera_ego_pose,
            expected_shape=camera_shape,
        )

    def _load_lidar_info(self) -> tuple[list[Path], dict[int, Path], set[int]]:
        # Load the lidar frame information.
        lidar_frame_directory = self.dataset_dir / "sensors" / "lidar"
        lidar_frame_paths = sorted(lidar_frame_directory.glob("*.feather"))
        assert len(lidar_frame_paths) > 0, f"no frames found in {lidar_frame_directory}"
        timestamp_to_lidar_file_map = {int(e.stem): e for e in lidar_frame_paths}
        lidar_file_timestamps = set(timestamp_to_lidar_file_map.keys())
        return lidar_frame_paths, timestamp_to_lidar_file_map, lidar_file_timestamps

    def _load_rgb_info(self, camera_name: str) -> tuple[list[Path], dict[int, Path]]:
        image_frame_directory = self.dataset_dir / "sensors" / "cameras" / camera_name
        image_frame_paths = sorted(image_frame_directory.glob("*.jpg"))
        assert len(image_frame_paths) > 0, f"no frames found in {image_frame_directory}"
        rgb_timestamp_to_rgb_file_map = {int(e.stem): e for e in image_frame_paths}
        return image_frame_paths, rgb_timestamp_to_rgb_file_map

    def _load_auxillary_pc_info(self) -> list[Path] | None:
        if not self.with_auxillary_pc:
            return None
        camera_pc_directory = self.dataset_dir / "sensors" / "camera_pc"
        if not camera_pc_directory.is_dir():
            return None
        camera_pc_paths = sorted(camera_pc_directory.glob("*.feather"))
        if len(camera_pc_paths) == 0:
            return None
        return camera_pc_paths

    def _quat_to_mat(self, qw, qx, qy, qz):
        """Convert a quaternion to a 3D rotation matrix.

        NOTE: SciPy uses the scalar last quaternion notation.

        Returns:
            (3,3) 3D rotation matrix.
        """

        # Convert quaternion from scalar first to scalar last.
        quat_xyzw = np.stack([qx, qy, qz, qw], axis=-1)
        mat = Rotation.from_quat(quat_xyzw).as_matrix()
        return mat

    def _load_camera_projection(self, sensor_name: str) -> CameraProjection:
        intrinsics_feather = self.dataset_dir / "calibration" / "intrinsics.feather"
        assert (
            intrinsics_feather.is_file()
        ), f"Expected intrinsics feather file at {intrinsics_feather}"
        raw_df = pd.read_feather(intrinsics_feather).set_index("sensor_name")
        params = raw_df.loc[sensor_name]
        fx = params["fx_px"]
        fy = params["fy_px"]
        cx = params["cx_px"]
        cy = params["cy_px"]
        return CameraProjection(fx, fy, cx, cy, CameraModel.PINHOLE)

    def _load_camera_ego_pose(self, sensor_name: str) -> SE3:
        sensor_poses_feather = self.dataset_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        assert (
            sensor_poses_feather.is_file()
        ), f"Expected sensor poses feather file at {sensor_poses_feather}"
        raw_df = pd.read_feather(sensor_poses_feather).set_index("sensor_name")
        params = raw_df.loc[sensor_name]
        # Stored as a quaternion plus translation vector in the feather file.
        qw = params["qw"]
        qx = params["qx"]
        qy = params["qy"]
        qz = params["qz"]
        tx = params["tx_m"]
        ty = params["ty_m"]
        tz = params["tz_m"]
        rotation = self._quat_to_mat(qw, qx, qy, qz)
        translation = np.array([tx, ty, tz])

        # fmt: off
        coordinate_transform_matrix = np.array(
            [[0, -1, 0],
             [0, 0, -1],
             [1, 0, 0]],
        )
        # fmt: on

        rotation = rotation @ coordinate_transform_matrix

        return SE3(rotation_matrix=rotation, translation=translation)

    def _load_ground_height_raster(self):
        raster_height_paths = list(
            (self.dataset_dir / "map").glob("*_ground_height_surface____*.npy")
        )
        assert (
            len(raster_height_paths) == 1
        ), f'Expected 1 raster, got {len(raster_height_paths)} in path {self.dataset_dir / "map"}'
        raster_height_path = raster_height_paths[0]

        transform_paths = list((self.dataset_dir / "map").glob("*img_Sim2_city.json"))
        assert len(transform_paths) == 1, f"Expected 1 transform, got {len(transform_paths)}"
        transform_path = transform_paths[0]

        raster_heightmap = np.load(raster_height_path)
        transform = load_json(transform_path, verbose=False)

        transform_rotation = np.array(transform["R"]).reshape(2, 2)
        transform_translation = np.array(transform["t"])
        transform_scale = np.array(transform["s"])

        transform_se2 = SE2(rotation=transform_rotation, translation=transform_translation)

        return raster_heightmap, transform_se2, transform_scale

    def get_ground_heights(self, global_point_cloud: PointCloud) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3) in global coordinates.
        Returns:
            ground_height_values: Numpy array of shape (k,)
        """

        global_points_xy = global_point_cloud.points[:, :2]

        raster_points_xy = (
            self.global_to_raster_se2.transform_point_cloud(global_points_xy)
            * self.global_to_raster_scale
        )

        raster_points_xy = np.round(raster_points_xy).astype(np.int64)

        ground_height_values = np.full((raster_points_xy.shape[0]), np.nan)
        # outside max X
        outside_max_x = (raster_points_xy[:, 0] >= self.raster_heightmap.shape[1]).astype(bool)
        # outside max Y
        outside_max_y = (raster_points_xy[:, 1] >= self.raster_heightmap.shape[0]).astype(bool)
        # outside min X
        outside_min_x = (raster_points_xy[:, 0] < 0).astype(bool)
        # outside min Y
        outside_min_y = (raster_points_xy[:, 1] < 0).astype(bool)
        ind_valid_pts = ~np.logical_or(
            np.logical_or(outside_max_x, outside_max_y),
            np.logical_or(outside_min_x, outside_min_y),
        )

        ground_height_values[ind_valid_pts] = self.raster_heightmap[
            raster_points_xy[ind_valid_pts, 1], raster_points_xy[ind_valid_pts, 0]
        ]

        return ground_height_values

    def is_ground_points(self, global_point_cloud: PointCloud) -> np.ndarray:
        """Remove ground points from a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,3) in global coordinates.
        Returns:
            ground_removed_point_cloud: Numpy array of shape (k,3) in global coordinates.
        """
        ground_height_values = self.get_ground_heights(global_point_cloud)
        is_ground_boolean_arr = (
            np.absolute(global_point_cloud[:, 2] - ground_height_values) <= GROUND_HEIGHT_THRESHOLD
        ) | (np.array(global_point_cloud[:, 2] - ground_height_values) < 0)
        return is_ground_boolean_arr

    def is_in_range(self, point_cloud: PointCloud) -> MaskArray:
        if self.point_cloud_range is None:
            return np.ones(len(point_cloud), dtype=bool)
        xmin = self.point_cloud_range[0]
        ymin = self.point_cloud_range[1]
        zmin = self.point_cloud_range[2]
        xmax = self.point_cloud_range[3]
        ymax = self.point_cloud_range[4]
        zmax = self.point_cloud_range[5]
        return point_cloud.within_region_mask(xmin, xmax, ymin, ymax, zmin, zmax)

    def __repr__(self) -> str:
        return f"ArgoverseSequence with {len(self)} frames"

    def __len__(self):
        return len(self.timestamp_list)

    def _timestamp_to_idx(self, timestamp: int) -> int:
        return self.timestamp_list.index(timestamp)

    def _load_pc(self, idx) -> PointCloud:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        timestamp = self.timestamp_list[idx]
        frame_path = self.timestamp_to_lidar_file_map[timestamp]
        frame_content = pd.read_feather(frame_path)
        xs = frame_content["x"].values
        ys = frame_content["y"].values
        zs = frame_content["z"].values
        points = np.stack([xs, ys, zs], axis=1)
        return PointCloud(points)

    def _load_auxillary_pc(self, idx) -> PointCloud | None:
        if self.auxillary_pc_paths is None:
            return None
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        if idx >= len(self.auxillary_pc_paths):
            # Sometimes the last few frames do not have camera point clouds.
            # For example, with a tracker window step size of 10 and a sequence length of 156,
            # the last 6 frames will have no camera point clouds.
            return PointCloud(np.zeros((0, 3), dtype=np.float32))
        camera_pc_path = self.auxillary_pc_paths[idx]
        frame_content = pd.read_feather(camera_pc_path)
        xs = frame_content["x"].values
        ys = frame_content["y"].values
        zs = frame_content["z"].values
        points = np.stack([xs, ys, zs], axis=1)
        return PointCloud(points)

    def _load_pose(self, idx) -> SE3:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        timestamp = self.timestamp_list[idx]
        infos_idx = self.timestamp_to_info_idx_map[timestamp]
        frame_info = self.frame_infos.iloc[infos_idx]
        se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
            frame_info["qw"],
            frame_info["qx"],
            frame_info["qy"],
            frame_info["qz"],
            frame_info["tx_m"],
            frame_info["ty_m"],
            frame_info["tz_m"],
        )
        return se3

    def load(
        self, idx: int, relative_to_idx: int
    ) -> tuple[TimeSyncedRawFrame, TimeSyncedAVLidarData]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        timestamp = self.timestamp_list[idx]
        ego_pc = self._load_pc(idx)
        auxillary_ego_pc = self._load_auxillary_pc(idx)

        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)
        absolute_global_frame_pc = ego_pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        relative_global_frame_pc_with_ground = ego_pc.transform(relative_pose)

        match self.range_crop_type:
            case RangeCropType.GLOBAL:
                in_range_mask_with_ground = self.is_in_range(relative_global_frame_pc_with_ground)
            case RangeCropType.EGO:
                in_range_mask_with_ground = self.is_in_range(ego_pc)
            case _:
                raise ValueError(f"Invalid range crop type {self.range_crop_type}")

        pc_frame = PointCloudFrame(
            full_pc=ego_pc,
            pose=PoseInfo(sensor_to_ego=SE3.identity(), ego_to_global=relative_pose),
            mask=np.ones(len(ego_pc), dtype=bool),
        )

        auxillary_pc_frame = None
        if auxillary_ego_pc is not None:
            auxillary_pc_frame = PointCloudFrame(
                full_pc=auxillary_ego_pc,
                pose=PoseInfo(sensor_to_ego=SE3.identity(), ego_to_global=relative_pose),
                mask=np.ones(len(auxillary_ego_pc), dtype=bool),
            )

        rgb_frames = RGBFrameLookup(
            {
                name: self.camera_info_lookup[name].load_rgb_frame(timestamp, relative_pose)
                for name in self.camera_names
            },
            self.camera_names,
        )

        return (
            TimeSyncedRawFrame(
                pc=pc_frame,
                auxillary_pc=auxillary_pc_frame,
                rgbs=rgb_frames,
                log_id=self.log_id,
                log_idx=idx,
                log_timestamp=timestamp,
            ),
            TimeSyncedAVLidarData(
                is_ground_points=is_ground_points, in_range_mask=in_range_mask_with_ground
            ),
        )

    def load_frame_list(
        self, relative_to_idx: Optional[int] = 0
    ) -> list[tuple[TimeSyncedRawFrame, TimeSyncedAVLidarData]]:
        return [
            self.load(idx, relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]


class ArgoverseRawSequenceLoader(CachedSequenceLoader):
    def __init__(
        self,
        sequence_dir: Path,
        with_rgb: bool = False,
        with_auxillary_pc: bool = False,
        log_subset: Optional[list[str]] = None,
        verbose: bool = False,
        num_sequences: Optional[int] = None,
        per_sequence_sample_every: Optional[int] = None,
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
        point_cloud_range: Optional[PointCloudRange] = DEFAULT_POINT_CLOUD_RANGE,
    ):
        super().__init__()
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        self.with_rgb = with_rgb
        self.with_auxillary_pc = with_auxillary_pc
        self.per_sequence_sample_every = per_sequence_sample_every
        self.expected_camera_shape = expected_camera_shape
        self.point_cloud_range = point_cloud_range
        assert self.dataset_dir.is_dir(), f"dataset_dir {sequence_dir} does not exist"
        self.log_lookup = {e.name: e for e in self.dataset_dir.glob("*/")}
        if log_subset is not None:
            log_subset = set(log_subset)
            log_keys = set(self.log_lookup.keys())
            assert log_subset.issubset(
                log_keys
            ), f"log_subset {log_subset} is not a subset of {log_keys}"
            self.log_lookup = {k: v for k, v in self.log_lookup.items() if k in log_subset}

        if num_sequences is not None:
            self.log_lookup = {
                k: v for idx, (k, v) in enumerate(self.log_lookup.items()) if idx < num_sequences
            }

        if self.verbose:
            print(f"Loaded {len(self.log_lookup)} logs")

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def _load_sequence_uncached(self, log_id: str) -> ArgoverseRawSequence:
        assert log_id in self.log_lookup, f"log_id {log_id} is not in the {len(self.log_lookup)}"
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f"log_id {log_id} does not exist"
        return ArgoverseRawSequence(
            log_id,
            log_dir,
            verbose=self.verbose,
            sample_every=self.per_sequence_sample_every,
            with_rgb=self.with_rgb,
            with_auxillary_pc=self.with_auxillary_pc,
            expected_camera_shape=self.expected_camera_shape,
            point_cloud_range=self.point_cloud_range,
        )

    def cache_folder_name(self) -> str:
        return f"av2_raw_data_with_rgb_{self.with_rgb}_sample_every_{self.per_sequence_sample_every}_dataset_dir_{self.dataset_dir.name}"
