import numpy as np
import pandas as pd
from pathlib import Path
from bucketed_scene_flow_eval.datastructures import PointCloud, SE3, SE2, RGBImage, CameraModel, CameraProjection
from bucketed_scene_flow_eval.utils import load_json
from typing import List, Tuple, Dict, Optional, Any
import time
import cv2
from scipy.spatial.transform import Rotation

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters


class ArgoverseRawSequence():
    """
    Argoverse Raw Sequence.

    Every sequence is a collection of frames. Unfortunately, RGB cameras are not perfectly 
    synced with the lidar frames, so we have to pull the most recent RGB image as a first order 
    approximation of the RGB at the given timestamp.
    """
    def __init__(self,
                 log_id: str,
                 dataset_dir: Path,
                 verbose: bool = False,
                 with_rgb: bool = True,
                 POINT_CLOUD_RANGE=(-48, -48, -2.5, 48, 48, 2.5),
                 sample_every: Optional[int] = None):
        self.log_id = log_id
        self.POINT_CLOUD_RANGE = POINT_CLOUD_RANGE

        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {dataset_dir} does not exist'

        # Load the vehicle pose information.
        self.frame_infos = pd.read_feather(self.dataset_dir /
                                           'city_SE3_egovehicle.feather')
        self.timestamp_to_info_idx_map = {
            int(timestamp): idx
            for idx, timestamp in enumerate(
                self.frame_infos['timestamp_ns'].values)
        }
        info_timestamps = set(self.timestamp_to_info_idx_map.keys())

        self.lidar_frame_paths, self.timestamp_to_lidar_file_map, lidar_file_timestamps = self._load_lidar_info()

        # Load the RGB frame information.
        camera_name = 'ring_front_center'

        if with_rgb:
            self.rgb_frame_paths, self.rgb_timestamp_to_rgb_file_map = self._load_rgb_info(camera_name)
    

            # Load the RGB intrinsics.
            self.rgb_camera_projection = self._load_camera_projection(camera_name)
            self.rgb_camera_ego_pose = self._load_camera_ego_pose(camera_name)

            # Find the nearest RGB percept to each lidar frame.
            # This is N^2. TODO: Figure out if this is a bottleneck.
            self.timestamp_to_rgb_timestamp_map = {
                lidar_timestamp:
                min(self.rgb_timestamp_to_rgb_file_map.keys(),
                    key=lambda rgb_timestamp: abs(rgb_timestamp - lidar_timestamp))
                for lidar_timestamp in lidar_file_timestamps
            }
        else:
            self.rgb_frame_paths = None
            self.rgb_timestamp_to_rgb_file_map = {}
            self.timestamp_to_rgb_timestamp_map = {}
            self.rgb_camera_projection = None
            self.rgb_camera_ego_pose = None

        self.timestamp_list = sorted(
            lidar_file_timestamps.intersection(info_timestamps))
        assert len(self.timestamp_list
                   ) > 0, f'no timestamps found in {self.dataset_dir}'

        if sample_every is not None:
            self.timestamp_list = self.timestamp_list[::sample_every]

        self.raster_heightmap, self.global_to_raster_se2, self.global_to_raster_scale = self._load_ground_height_raster(
        )

        if verbose:
            print(
                f'Loaded {len(self.timestamp_list)} frames from {self.dataset_dir} at timestamp {time.time():.3f}'
            )

        self.with_rgb = with_rgb

    def _load_lidar_info(self):
        # Load the lidar frame information.
        lidar_frame_directory = self.dataset_dir / 'sensors' / 'lidar'
        lidar_frame_paths = sorted(
            lidar_frame_directory.glob('*.feather'))
        assert len(lidar_frame_paths
                   ) > 0, f'no frames found in {lidar_frame_directory}'
        timestamp_to_lidar_file_map = {
            int(e.stem): e
            for e in lidar_frame_paths
        }
        lidar_file_timestamps = set(timestamp_to_lidar_file_map.keys())
        return lidar_frame_paths, timestamp_to_lidar_file_map, lidar_file_timestamps

    def _load_rgb_info(self, camera_name : str):
        image_frame_directory = self.dataset_dir / 'sensors' / 'cameras' / camera_name
        image_frame_paths = sorted(image_frame_directory.glob('*.jpg'))
        assert len(image_frame_paths
                   ) > 0, f'no frames found in {image_frame_directory}'
        rgb_timestamp_to_rgb_file_map = {
            int(e.stem): e
            for e in image_frame_paths
        }
        return image_frame_paths, rgb_timestamp_to_rgb_file_map


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
        assert intrinsics_feather.is_file(
        ), f'Expected intrinsics feather file at {intrinsics_feather}'
        raw_df = pd.read_feather(intrinsics_feather).set_index("sensor_name")
        params = raw_df.loc[sensor_name]
        fx = params["fx_px"]
        fy = params["fy_px"]
        cx = params["cx_px"]
        cy = params["cy_px"]
        return CameraProjection(fx, fy, cx, cy, CameraModel.PINHOLE)

    def _load_camera_ego_pose(self, sensor_name: str) -> SE3:
        sensor_poses_feather = self.dataset_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        assert sensor_poses_feather.is_file(
        ), f'Expected sensor poses feather file at {sensor_poses_feather}'
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

        coordinate_transform_matrix = np.array([[ 0, -1,  0],  # noqa
                                                [ 0,  0, -1],  # noqa
                                                [ 1,  0,  0]]) # noqa

        rotation = rotation @ coordinate_transform_matrix

        return SE3(rotation_matrix=rotation, translation=translation)

    def _load_ground_height_raster(self):
        raster_height_paths = list(
            (self.dataset_dir /
             'map').glob("*_ground_height_surface____*.npy"))
        assert len(
            raster_height_paths
        ) == 1, f'Expected 1 raster, got {len(raster_height_paths)} in path {self.dataset_dir / "map"}'
        raster_height_path = raster_height_paths[0]

        transform_paths = list(
            (self.dataset_dir / 'map').glob("*img_Sim2_city.json"))
        assert len(transform_paths
                   ) == 1, f'Expected 1 transform, got {len(transform_paths)}'
        transform_path = transform_paths[0]

        raster_heightmap = np.load(raster_height_path)
        transform = load_json(transform_path, verbose=False)

        transform_rotation = np.array(transform['R']).reshape(2, 2)
        transform_translation = np.array(transform['t'])
        transform_scale = np.array(transform['s'])

        transform_se2 = SE2(rotation=transform_rotation,
                            translation=transform_translation)

        return raster_heightmap, transform_se2, transform_scale

    def get_ground_heights(self, global_point_cloud: PointCloud) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3) in global coordinates.
        Returns:
            ground_height_values: Numpy array of shape (k,)
        """

        global_points_xy = global_point_cloud.points[:, :2]

        raster_points_xy = self.global_to_raster_se2.transform_point_cloud(
            global_points_xy) * self.global_to_raster_scale

        raster_points_xy = np.round(raster_points_xy).astype(np.int64)

        ground_height_values = np.full((raster_points_xy.shape[0]), np.nan)
        # outside max X
        outside_max_x = (raster_points_xy[:, 0] >=
                         self.raster_heightmap.shape[1]).astype(bool)
        # outside max Y
        outside_max_y = (raster_points_xy[:, 1] >=
                         self.raster_heightmap.shape[0]).astype(bool)
        # outside min X
        outside_min_x = (raster_points_xy[:, 0] < 0).astype(bool)
        # outside min Y
        outside_min_y = (raster_points_xy[:, 1] < 0).astype(bool)
        ind_valid_pts = ~np.logical_or(
            np.logical_or(outside_max_x, outside_max_y),
            np.logical_or(outside_min_x, outside_min_y))

        ground_height_values[ind_valid_pts] = self.raster_heightmap[
            raster_points_xy[ind_valid_pts, 1], raster_points_xy[ind_valid_pts,
                                                                 0]]

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
            np.absolute(global_point_cloud[:, 2] - ground_height_values) <=
            GROUND_HEIGHT_THRESHOLD) | (
                np.array(global_point_cloud[:, 2] - ground_height_values) < 0)
        return is_ground_boolean_arr

    def is_in_range(self, global_point_cloud: PointCloud) -> np.ndarray:
        xmin = self.POINT_CLOUD_RANGE[0]
        ymin = self.POINT_CLOUD_RANGE[1]
        zmin = self.POINT_CLOUD_RANGE[2]
        xmax = self.POINT_CLOUD_RANGE[3]
        ymax = self.POINT_CLOUD_RANGE[4]
        zmax = self.POINT_CLOUD_RANGE[5]
        return global_point_cloud.within_region_mask(xmin, xmax, ymin, ymax,
                                                     zmin, zmax)

    def __repr__(self) -> str:
        return f'ArgoverseSequence with {len(self)} frames'

    def __len__(self):
        return len(self.timestamp_list)

    def _timestamp_to_idx(self, timestamp: int) -> int:
        return self.timestamp_list.index(timestamp)

    def _load_pc(self, idx) -> PointCloud:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        frame_path = self.timestamp_to_lidar_file_map[timestamp]
        frame_content = pd.read_feather(frame_path)
        xs = frame_content['x'].values
        ys = frame_content['y'].values
        zs = frame_content['z'].values
        points = np.stack([xs, ys, zs], axis=1)
        return PointCloud(points)

    def _load_rgb(self, idx) -> RGBImage:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        rgb_timestamp = self.timestamp_to_rgb_timestamp_map[timestamp]
        rgb_path = self.rgb_timestamp_to_rgb_file_map[rgb_timestamp]
        # Read the image, keep the same color space
        raw_img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.0
        # Convert from CV2 standard BGR to RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        return RGBImage(raw_img)

    def _load_pose(self, idx) -> SE3:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        infos_idx = self.timestamp_to_info_idx_map[timestamp]
        frame_info = self.frame_infos.iloc[infos_idx]
        se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
            frame_info['qw'], frame_info['qx'], frame_info['qy'],
            frame_info['qz'], frame_info['tx_m'], frame_info['ty_m'],
            frame_info['tz_m'])
        return se3

    def load(self, idx: int, relative_to_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        ego_pc = self._load_pc(idx)
        if self.with_rgb:
            img = self._load_rgb(idx)
        else:
            img = None
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)
        absolute_global_frame_pc = ego_pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        relative_global_frame_pc_with_ground = ego_pc.transform(relative_pose)
        relative_global_frame_pc_no_ground = relative_global_frame_pc_with_ground.mask_points(
            ~is_ground_points)
        ego_pc_no_ground = ego_pc.mask_points(~is_ground_points)

        in_range_mask_with_ground = self.is_in_range(
            relative_global_frame_pc_with_ground)
        in_range_mask_no_ground = self.is_in_range(
            relative_global_frame_pc_no_ground)

        return {
            "ego_pc": ego_pc_no_ground,
            "ego_pc_with_ground": ego_pc,
            "relative_pc": relative_global_frame_pc_no_ground,
            "relative_pc_with_ground": relative_global_frame_pc_with_ground,
            "relative_pose": relative_pose,
            "is_ground_points": is_ground_points,
            "in_range_mask": in_range_mask_with_ground,
            "in_range_mask_no_ground": in_range_mask_no_ground,
            "rgb": img,
            "rgb_camera_projection": self.rgb_camera_projection,
            "rgb_camera_ego_pose": self.rgb_camera_ego_pose,
            "log_id": self.log_id,
            "log_idx": idx,
            "log_timestamp": timestamp,
        }

    def load_frame_list(
            self, relative_to_idx: Optional[int]) -> List[Dict[str, Any]]:

        return [
            self.load(idx,
                      relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]


class ArgoverseRawSequenceLoader():
    def __init__(self,
                 sequence_dir: Path,
                 log_subset: Optional[List[str]] = None,
                 verbose: bool = False,
                 num_sequences: Optional[int] = None,
                 per_sequence_sample_every: Optional[int] = None):
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        self.per_sequence_sample_every = per_sequence_sample_every
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {sequence_dir} does not exist'
        self.log_lookup = {e.name: e for e in self.dataset_dir.glob('*/')}
        if log_subset is not None:
            log_subset = set(log_subset)
            log_keys = set(self.log_lookup.keys())
            assert log_subset.issubset(
                log_keys
            ), f'log_subset {log_subset} is not a subset of {log_keys}'
            self.log_lookup = {
                k: v
                for k, v in self.log_lookup.items() if k in log_subset
            }

        if num_sequences is not None:
            self.log_lookup = {
                k: v
                for idx, (k, v) in enumerate(self.log_lookup.items())
                if idx < num_sequences
            }

        if self.verbose:
            print(f'Loaded {len(self.log_lookup)} logs')

        self.last_loaded_sequence = None
        self.last_loaded_sequence_id = None

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def _load_sequence_uncached(self, log_id: str) -> ArgoverseRawSequence:
        assert log_id in self.log_lookup, f'log_id {log_id} is not in the {len(self.log_lookup)}'
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f'log_id {log_id} does not exist'
        return ArgoverseRawSequence(
            log_id,
            log_dir,
            verbose=self.verbose,
            sample_every=self.per_sequence_sample_every)

    def load_sequence(self, log_id: str) -> ArgoverseRawSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != log_id:
            self.last_loaded_sequence = self._load_sequence_uncached(log_id)
            self.last_loaded_sequence_id = log_id

        return self.last_loaded_sequence
