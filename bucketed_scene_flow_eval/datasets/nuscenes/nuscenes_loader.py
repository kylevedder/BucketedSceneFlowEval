import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud as NuscLidarPointCloud
from PIL import Image
from pyquaternion import Quaternion

from bucketed_scene_flow_eval.datasets.shared_datastructures import (
    AbstractSequence,
    AbstractSequenceLoader,
    CachedSequenceLoader,
    RawItem,
)
from bucketed_scene_flow_eval.datastructures import (
    SE2,
    SE3,
    CameraModel,
    CameraProjection,
    PointCloud,
    PointCloudFrame,
    PoseInfo,
    RGBFrame,
    RGBImage,
)

NuscDict = dict[str, Union[str, int, list]]
NuscSample = dict[str, NuscDict]
NuscSampleData = NuscDict
NuscCalibratedSensor = NuscDict
NuscEgoPose = NuscDict

import matplotlib.pyplot as plt


@dataclass
class NuScenesSyncedSampleData:
    cam_front: NuscSampleData
    lidar_top: NuscSampleData

    def _load_pointcloud(self, nusc: NuScenes, pointsensor: NuscSampleData) -> PointCloud:
        pc_path = Path(nusc.dataroot) / pointsensor["filename"]  # type: ignore
        assert pc_path.is_file(), f"pointcloud file {pc_path} does not exist"
        nusc_lidar_pc = NuscLidarPointCloud.from_file(str(pc_path.absolute()))
        xyz_points = nusc_lidar_pc.points[:3].T
        return PointCloud(xyz_points)

    def _load_poseinfo(self, nusc: NuScenes, sample_data: NuscSampleData) -> PoseInfo:
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

        # Load Ego to World
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

    def _load_rgb(self, nusc: NuScenes, sample_data: NuscSampleData) -> RGBImage:
        data_path = nusc.get_sample_data_path(sample_data["token"])
        data = Image.open(data_path)
        np_data_uint8 = np.array(data)
        np_data_float32 = np_data_uint8.astype(np.float32) / 255.0
        return RGBImage(np_data_float32)

    def _load_camera_projection(
        self, nusc: NuScenes, sample_data: NuscSampleData
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

    def lidar_to_pc_frame(self, nusc: NuScenes) -> PointCloudFrame:
        pc = self._load_pointcloud(nusc, self.lidar_top)
        pose_info = self._load_poseinfo(nusc, self.lidar_top)
        mask = np.ones(pc.points.shape[0], dtype=bool)
        return PointCloudFrame(pc, pose_info, mask)

    def camera_to_rgb_frame(self, nusc: NuScenes) -> RGBFrame:
        rgb = self._load_rgb(nusc, self.cam_front)
        pose_info = self._load_poseinfo(nusc, self.cam_front)
        camera_projection = self._load_camera_projection(nusc, self.cam_front)
        return RGBFrame(rgb, pose_info, camera_projection)

    def cam_front_with_points(self, nusc: NuScenes):
        pointsensor_token = self.lidar_top["token"]
        camera_token = self.cam_front["token"]
        points, depth, im = nusc.explorer.map_pointcloud_to_image(
            pointsensor_token, camera_token, render_intensity=False
        )

        np_im = np.array(im)
        plt.imshow(np_im)
        plt.scatter(points[0, :], points[1, :], c=depth, s=1)
        plt.show()

        breakpoint()


class NuScenesSequence(AbstractSequence):
    """
    Argoverse Raw Sequence.

    Every sequence is a collection of frames. Unfortunately, RGB cameras are not perfectly
    synced with the lidar frames, so we have to pull the most recent RGB image as a first order
    approximation of the RGB at the given timestamp.
    """

    def __init__(
        self,
        nusc: NuScenes,
        log_id: str,
        scene_info: NuscDict,
        verbose: bool = False,
    ):
        self.nusc = nusc
        self.scene_info = scene_info
        self.log_id = log_id

        sensor_to_sample_data_table = self._sensor_to_sample_data_table(
            scene_info, ["CAM_FRONT", "LIDAR_TOP"]
        )

        self.synced_sensors = self._sync_sensors(
            sensor_to_sample_data_table["CAM_FRONT"],
            sensor_to_sample_data_table["LIDAR_TOP"],
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

        # Load the calibarted sensor
        sensor: NuscCalibratedSensor = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )

        # Load the RGB image
        data_path = self.nusc.get_sample_data_path(cam_data["token"])
        data = Image.open(data_path)
        np_data_uint8 = np.array(data)
        np_data_float32 = np_data_uint8.astype(np.float32) / 255.0
        return RGBImage(np_data_float32)

    def load(self, idx: int, relative_to_idx: int) -> RawItem:
        assert 0 <= idx < len(self), f"idx must be in range [0, {len(self)}), got {idx}"
        synced_sample = self.synced_sensors[idx]

        pc_frame = synced_sample.lidar_to_pc_frame(self.nusc)
        rgb_frame = synced_sample.camera_to_rgb_frame(self.nusc)

        return RawItem(
            pc=pc_frame,
            is_ground_points=np.zeros(len(pc_frame.pc), dtype=bool),
            in_range_mask=np.ones(len(pc_frame.pc), dtype=bool),
            rgbs=[rgb_frame],
            log_id=self.log_id,
            log_idx=idx,
            log_timestamp=idx,
        )

    def __len__(self):
        return len(self.synced_sensors)


class NuScenesLoader(CachedSequenceLoader):
    def __init__(
        self,
        sequence_dir: Path,
        version: str = "v1.0",
        verbose: bool = False,
    ):
        super().__init__()
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        assert self.dataset_dir.is_dir(), f"dataset_dir {sequence_dir} does not exist"
        self.nusc = NuScenes(version=version, dataroot=sequence_dir, verbose=verbose)
        self.log_lookup: dict[str, NuscDict] = {e["token"]: e for e in self.nusc.scene}

        if self.verbose:
            print(f"Loaded {len(self)} logs")

    def __len__(self):
        return len(self.log_lookup)

    def get_sequence_ids(self) -> list:
        return sorted(self.log_lookup.keys())

    def _load_sequence_uncached(self, log_id: str) -> NuScenesSequence:
        assert log_id in self.log_lookup, f"log_id {log_id} is not in the {len(self.log_lookup)}"
        log_info_dict = self.log_lookup[log_id]
        return NuScenesSequence(
            self.nusc,
            log_id,
            log_info_dict,
            verbose=self.verbose,
        )
