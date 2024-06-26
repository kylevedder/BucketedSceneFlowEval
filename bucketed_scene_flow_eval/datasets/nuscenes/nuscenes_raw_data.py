from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from nuscenes.utils.data_classes import LidarPointCloud as NuscLidarPointCloud
from PIL import Image
from pyquaternion import Quaternion

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_raw_data import (
    DEFAULT_POINT_CLOUD_RANGE,
    PointCloudRange,
)
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    CameraModel,
    CameraProjection,
    PointCloud,
    PointCloudFrame,
    PoseInfo,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
    TimeSyncedAVLidarData,
    TimeSyncedRawFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractSequence, CachedSequenceLoader

from .nuscenes_utils import NuScenesWithInstanceBoxes, create_splits_tokens

NuscDict = dict[str, Union[str, int, list]]
NuscSample = dict[str, NuscDict]
NuscSampleData = NuscDict
NuscCalibratedSensor = NuscDict
NuscEgoPose = NuscDict

import matplotlib.pyplot as plt

MaskArray = NDArray[np.bool_]


@dataclass
class NuScenesSyncedSampleData:
    cam_front: NuscSampleData
    lidar_top: NuscSampleData

    def _load_pointcloud(
        self, nusc: NuScenesWithInstanceBoxes, pointsensor: NuscSampleData
    ) -> PointCloud:
        pc_path = Path(nusc.dataroot) / pointsensor["filename"]  # type: ignore
        assert pc_path.is_file(), f"pointcloud file {pc_path} does not exist"
        nusc_lidar_pc = NuscLidarPointCloud.from_file(str(pc_path.absolute()))
        xyz_points = nusc_lidar_pc.points[:3].T
        return PointCloud(xyz_points)

    def _load_poseinfo(
        self, nusc: NuScenesWithInstanceBoxes, sample_data: NuscSampleData
    ) -> PoseInfo:
        # Load Sensor to Ego
        sensor: NuscCalibratedSensor = nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        sensor_to_ego_translation = np.array(sensor["translation"])
        sensor_to_ego_rotation = Quaternion(sensor["rotation"])
        sensor_to_ego_se3 = SE3(
            translation=sensor_to_ego_translation,
            rotation_matrix=sensor_to_ego_rotation.rotation_matrix,
        )

        # Load Ego to World (map)
        poserecord: NuscEgoPose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        ego_to_world_translation = np.array(poserecord["translation"])
        ego_to_world_rotation = Quaternion(poserecord["rotation"])
        ego_to_world_se3 = SE3(
            translation=ego_to_world_translation,
            rotation_matrix=ego_to_world_rotation.rotation_matrix,
        )

        return PoseInfo(
            sensor_to_ego=sensor_to_ego_se3,
            ego_to_global=ego_to_world_se3,
        )

    def _load_rgb(self, nusc: NuScenesWithInstanceBoxes, sample_data: NuscSampleData) -> RGBImage:
        data_path = nusc.get_sample_data_path(sample_data["token"])
        data = Image.open(data_path)
        np_data_uint8 = np.array(data)
        np_data_float32 = np_data_uint8.astype(np.float32) / 255.0
        return RGBImage(np_data_float32)

    def _load_camera_projection(
        self, nusc: NuScenesWithInstanceBoxes, sample_data: NuscSampleData
    ) -> CameraProjection:
        sensor: NuscCalibratedSensor = nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        camera_matrix_3x3 = np.array(sensor["camera_intrinsic"])
        fx = camera_matrix_3x3[0, 0]
        fy = camera_matrix_3x3[1, 1]
        cx = camera_matrix_3x3[0, 2]
        cy = camera_matrix_3x3[1, 2]
        return CameraProjection(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_model=CameraModel.PINHOLE,
        )

    def lidar_to_pc_frame(self, nusc: NuScenesWithInstanceBoxes) -> PointCloudFrame:
        pc = self._load_pointcloud(nusc, self.lidar_top)
        pose_info = self._load_poseinfo(nusc, self.lidar_top)
        mask = np.ones(pc.points.shape[0], dtype=bool)
        return PointCloudFrame(pc, pose_info, mask)

    def camera_to_rgb_frame_lookup(self, nusc: NuScenesWithInstanceBoxes) -> RGBFrameLookup:
        rgb = self._load_rgb(nusc, self.cam_front)
        pose_info = self._load_poseinfo(nusc, self.cam_front)
        camera_projection = self._load_camera_projection(nusc, self.cam_front)
        cam_front_rgb_frame = RGBFrame(rgb, pose_info, camera_projection)
        return RGBFrameLookup({"cam_front": cam_front_rgb_frame}, ["cam_front"])

    def cam_front_with_points(self, nusc: NuScenesWithInstanceBoxes):
        pointsensor_token = self.lidar_top["token"]
        camera_token = self.cam_front["token"]
        points, depth, im = nusc.explorer.map_pointcloud_to_image(
            pointsensor_token, camera_token, render_intensity=False
        )

        np_im = np.array(im)
        plt.imshow(np_im)
        plt.scatter(points[0, :], points[1, :], c=depth, s=1)
        plt.show()


class NuScenesRawSequence(AbstractSequence):
    """
    NuScenes Raw Sequence.

    Every sequence is a collection of frames. Unfortunately, RGB cameras are not perfectly
    synced with the lidar frames, so we have to pull the most recent RGB image as a first order
    approximation of the RGB at the given timestamp.
    """

    def __init__(
        self,
        nusc: NuScenesWithInstanceBoxes,
        log_id: str,
        scene_info: NuscDict,
        verbose: bool = False,
        with_rgb: bool = False,
        point_cloud_range: PointCloudRange | None = DEFAULT_POINT_CLOUD_RANGE,
    ):
        self.nusc = nusc
        self.scene_info = scene_info
        self.log_id = log_id
        self.point_cloud_range = point_cloud_range
        self.with_rgb = with_rgb
        self.sensor_to_sample_data_table = self._sensor_to_sample_data_table(
            scene_info, ["CAM_FRONT", "LIDAR_TOP"]
        )

        self.synced_sensors = self._sync_sensors(
            self.sensor_to_sample_data_table["CAM_FRONT"],
            self.sensor_to_sample_data_table["LIDAR_TOP"],
        )

    def _sync_sensors(
        self, cam_front: list[NuscSampleData], lidar_top: list[NuscSampleData]
    ) -> list[NuScenesSyncedSampleData]:
        """
        Sync the sensors to the same timestamp.
        """
        synced_list = []
        for sync_sensor_sample_data in cam_front:
            lidar_top_sample_data = self._get_nearest_timestamp(
                sync_sensor_sample_data["timestamp"],  # type: ignore
                lidar_top,
            )
            synced_list.append(
                NuScenesSyncedSampleData(
                    cam_front=sync_sensor_sample_data,
                    lidar_top=lidar_top_sample_data,
                )
            )
        return synced_list

    def _sensor_to_sample_data_table(
        self, scene_info: NuscDict, sensors_of_interest: list[str]
    ) -> dict[str, list[NuscSampleData]]:
        first_sample_token = scene_info["first_sample_token"]
        first_sample = self.nusc.get("sample", first_sample_token)

        sensor_to_sample_data_table = {
            sensor: self._extract_sample_data_list(sensor, first_sample)
            for sensor in sensors_of_interest
        }

        return sensor_to_sample_data_table

    def _get_nearest_timestamp(
        self,
        timestamp: int,
        sample_datas: list[NuscSampleData],
    ) -> NuscSampleData:
        """
        Given a timestamp, find the nearest timestamp in the list of sample datas.
        """
        assert len(sample_datas) > 0, f"sample_datas must have at least one element"
        min_idx = np.argmin(np.abs(np.array([sd["timestamp"] for sd in sample_datas]) - timestamp))
        return sample_datas[min_idx]

    def _extract_sample_data_list(
        self, sensor_name: str, sample: NuscSample
    ) -> list[NuscSampleData]:
        sample_data_token = sample["data"][sensor_name]
        sample_data = self.nusc.get("sample_data", sample_data_token)

        sample_data_list = []
        while sample_data["next"] != "":
            sample_data_list.append(sample_data)
            sample_data = self.nusc.get("sample_data", sample_data["next"])
        return sample_data_list

    def _load_cam_data(self, cam_data: NuscSampleData) -> RGBImage:
        # Render the camera
        self.nusc.render_sample_data(cam_data["token"])

        # Load the calibrated sensor
        sensor: NuscCalibratedSensor = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )

        # Load the RGB image
        data_path = self.nusc.get_sample_data_path(cam_data["token"])
        data = Image.open(data_path)
        np_data_uint8 = np.array(data)
        np_data_float32 = np_data_uint8.astype(np.float32) / 255.0
        return RGBImage(np_data_float32)

    def _compute_ground_mask(self, pc_frame: PointCloudFrame) -> MaskArray:
        """Given a point cloud frame, compute the ground plane.

        Args:
            pc_frame: a PointCloudFrame object
        Returns:
            is_ground_mask: true if the index is part of the ground plane, false otherwise
        """
        # Consider only points with Z values close to 0 otherwise RANSAC may produce planes that are not actually on the ground
        points_close_to_ground_mask = pc_frame.full_ego_pc.within_region_mask(
            -1000, 1000, -1000, 1000, -2, 0.5
        )
        points_close_to_ground = pc_frame.full_ego_pc.mask_points(points_close_to_ground_mask)
        # Open3D's RANSAC returns a list of indices (in the input point cloud) of inliers in the plane
        o3d.utility.random.seed(
            42
        )  # We set the seed so that the ground plane extraction is deterministic
        _, ground_plane_inliers = points_close_to_ground.to_o3d().segment_plane(
            distance_threshold=0.2, ransac_n=3, num_iterations=100, probability=1.0
        )
        # Since we passed in only the points close to the ground to the RANSAC segmentation, the indices are with respect to points_close_to_ground and not the original point cloud
        # In order to get the indices of the points in the ground plane with respect to the original point cloud we must first conver the indices to a mask
        actual_ground_in_points_close_to_ground_mask = np.zeros(
            (points_close_to_ground.shape[0]), dtype=bool
        )
        actual_ground_in_points_close_to_ground_mask[
            ground_plane_inliers
        ] = 1  # Convert indices to a binary mask of points_close_to_ground
        # Create another mask to represent the final ground mask with respect to the full point cloud
        is_ground_mask = np.zeros_like(
            points_close_to_ground_mask, dtype=bool
        )  # This has the same shape as the original point cloud
        # This line does all the heavy lifting. is_ground_mask[points_close_to_ground_mask] takes a mask with the same size of the original point cloud
        # and crops it to only the indices that were in our Z-height cropped pc. Then we set those indices to be equal to their value in the inlier mask from RANSAC.
        is_ground_mask[points_close_to_ground_mask] = actual_ground_in_points_close_to_ground_mask
        return is_ground_mask

    def _in_range_mask(self, global_point_cloud: PointCloud) -> MaskArray:
        if self.point_cloud_range is None:
            return np.ones(len(global_point_cloud), dtype=bool)
        xmin = self.point_cloud_range[0]
        ymin = self.point_cloud_range[1]
        zmin = self.point_cloud_range[2]
        xmax = self.point_cloud_range[3]
        ymax = self.point_cloud_range[4]
        zmax = self.point_cloud_range[5]
        return global_point_cloud.within_region_mask(xmin, xmax, ymin, ymax, zmin, zmax)

    def _load_pose(self, idx: int) -> PoseInfo:
        """Returns the PoseInfo object for the LiDAR data at a given index. Importantly there is NO RELATIVE INDEX for this function.

        Thus the "global" transformation contains the EGO -> MAP FRAME transformation.
        """
        sample = self.synced_sensors[idx]
        return sample._load_poseinfo(self.nusc, sample.lidar_top)

    def load(
        self,
        idx: int,
        relative_to_idx: int,
    ) -> tuple[TimeSyncedRawFrame, TimeSyncedAVLidarData]:
        """The load function loads a sample at some index, relative to another.

        Concretely, the nuScenes API convention is that there are 3 frames. Sensor frame (origin = the sensor), ego frame (origin = vehicle) and global frame (origin = some point on the map)
        For the purposes of scene flow, we want our "global frame" origin to be the ego pose of a different sample in the sequence.
        We abuse the terminology a bit and temporarily stored the ego -> map transform in the "global" field of the PoseInfo object.

        To summarize. When the sample is loaded from the nuScenes API using `lidar_to_pc_frame` global = map.
        But after this load function is called, we replace the value such that global = the ego pose at the relative idx.

        Args:
            idx: the sample to load
            relative_to_idx: the origin of the sequence, all point clouds can be represented in a frame relative to this one
        """

        assert 0 <= idx < len(self), f"idx must be in range [0, {len(self)}), got {idx}"
        synced_sample = self.synced_sensors[idx]

        pc_frame = synced_sample.lidar_to_pc_frame(self.nusc)
        if self.with_rgb:
            rgb_frames = synced_sample.camera_to_rgb_frame_lookup(self.nusc)
        else:
            rgb_frames = RGBFrameLookup({}, [])

        start_ego_pose = self._load_pose(relative_to_idx).ego_to_global
        relative_pose = start_ego_pose.inverse().compose(pc_frame.pose.ego_to_global)
        pc_frame.pose.ego_to_global = relative_pose

        return (
            TimeSyncedRawFrame(
                pc=pc_frame,
                auxillary_pc=None,
                rgbs=rgb_frames,
                log_id=self.log_id,
                log_idx=idx,
                log_timestamp=idx,
            ),
            TimeSyncedAVLidarData(
                is_ground_points=self._compute_ground_mask(pc_frame),
                in_range_mask=self._in_range_mask(pc_frame.global_pc),
            ),
        )

    def __len__(self):
        return len(self.synced_sensors)


class NuScenesRawSequenceLoader(CachedSequenceLoader):
    def __init__(
        self,
        sequence_dir: Path,
        split: str,
        version: str = "v1.0-mini",
        verbose: bool = False,
        point_cloud_range: PointCloudRange | None = DEFAULT_POINT_CLOUD_RANGE,
    ):
        super().__init__()
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        assert self.dataset_dir.is_dir(), f"dataset_dir {sequence_dir} does not exist"
        self.nusc = NuScenesWithInstanceBoxes(
            version=version, dataroot=sequence_dir, verbose=verbose
        )
        self.log_lookup: dict[str, NuscDict] = {e["token"]: e for e in self.nusc.scene}
        self.log_lookup: dict[str, NuscDict] = {
            k: self.log_lookup[k] for k in create_splits_tokens(split, self.nusc)
        }

        self.point_cloud_range = point_cloud_range

        if self.verbose:
            print(f"Loaded {len(self)} logs")

    def __len__(self):
        return len(self.log_lookup)

    def get_sequence_ids(self) -> list:
        return sorted(self.log_lookup.keys())

    def __getitem__(self, idx: int) -> NuScenesRawSequence:
        seq_id = self.get_sequence_ids()[idx]
        return self.load_sequence(seq_id)

    def _load_sequence_uncached(self, log_id: str) -> NuScenesRawSequence:
        assert log_id in self.log_lookup, f"log_id {log_id} is not in the {len(self.log_lookup)}"
        log_info_dict = self.log_lookup[log_id]
        return NuScenesRawSequence(
            self.nusc,
            log_id,
            log_info_dict,
            verbose=self.verbose,
            point_cloud_range=self.point_cloud_range,
        )

    def cache_folder_name(self) -> str:
        return f"nuscenes_dataset_{self.nusc.version}_dataset_dir_{self.dataset_dir.name}"
