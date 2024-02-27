import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from nuscenes.nuscenes import NuScenes

from bucketed_scene_flow_eval.datasets.shared_dataclasses import RawItem
from bucketed_scene_flow_eval.datastructures import (
    SE2,
    SE3,
    CameraModel,
    CameraProjection,
    PointCloud,
    RGBImage,
)
from bucketed_scene_flow_eval.utils import load_json

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters


class NuScenesRawSequence:
    """
    Argoverse Raw Sequence.

    Every sequence is a collection of frames. Unfortunately, RGB cameras are not perfectly
    synced with the lidar frames, so we have to pull the most recent RGB image as a first order
    approximation of the RGB at the given timestamp.
    """

    def __init__(
        self,
        log_id: str,
        scene_info: dict[str, Union[str, int]],
        verbose: bool = False,
        with_rgb: bool = False,
        POINT_CLOUD_RANGE=(-48, -48, -2.5, 48, 48, 2.5),
    ):
        self.scene_info = scene_info
        self.log_id = log_id
        self.POINT_CLOUD_RANGE = POINT_CLOUD_RANGE

        # self.sample_list

    def _load_sample_list(self) -> list:

    def __repr__(self) -> str:
        return f"NuScenesSequence with {len(self)} frames. Description: {self.scene_info['description']}"

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

    def _load_rgb(self, idx) -> RGBImage:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        timestamp = self.timestamp_list[idx]
        rgb_timestamp = self.timestamp_to_rgb_timestamp_map[timestamp]
        rgb_path = self.rgb_timestamp_to_rgb_file_map[rgb_timestamp]
        # Read the image, keep the same color space
        raw_img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        # Convert from CV2 standard BGR to RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        return RGBImage(raw_img)

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

    def load(self, idx: int, relative_to_idx: int) -> RawItem:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
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
            ~is_ground_points
        )
        ego_pc_no_ground = ego_pc.mask_points(~is_ground_points)

        in_range_mask_with_ground = self.is_in_range(relative_global_frame_pc_with_ground)
        in_range_mask_no_ground = self.is_in_range(relative_global_frame_pc_no_ground)
        return RawItem(
            ego_pc=ego_pc_no_ground,
            ego_pc_with_ground=ego_pc,
            relative_pc=relative_global_frame_pc_no_ground,
            relative_pc_with_ground=relative_global_frame_pc_with_ground,
            is_ground_points=is_ground_points,
            in_range_mask=in_range_mask_no_ground,
            in_range_mask_with_ground=in_range_mask_with_ground,
            rgb=img,
            rgb_camera_projection=self.rgb_camera_projection,
            rgb_camera_ego_pose=self.rgb_camera_ego_pose,
            relative_pose=relative_pose,
            log_id=self.log_id,
            log_idx=idx,
            log_timestamp=timestamp,
        )

    def load_frame_list(self, relative_to_idx: Optional[int]) -> list[RawItem]:
        return [
            self.load(idx, relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]


class NuScenesRawSequenceLoader:
    def __init__(
        self,
        sequence_dir: Path,
        verbose: bool = False,
    ):
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        assert self.dataset_dir.is_dir(), f"dataset_dir {sequence_dir} does not exist"
        self.nusc = NuScenes(version="v1.0", dataroot="sequence_dir", verbose=verbose)
        self.log_lookup: dict[str, dict[str, Union[str, int]]] = {
            e["token"]: e for e in self.nusc.scene
        }

        if self.verbose:
            print(f"Loaded {len(self.log_lookup)} logs")

        self.last_loaded_sequence = None
        self.last_loaded_sequence_id = None

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def _load_sequence_uncached(self, log_id: str) -> NuScenesRawSequence:
        assert log_id in self.log_lookup, f"log_id {log_id} is not in the {len(self.log_lookup)}"
        log_info_dict = self.log_lookup[log_id]
        return NuScenesRawSequence(
            log_id,
            log_info_dict,
            verbose=self.verbose,
        )

    def load_sequence(self, log_id: str) -> NuScenesRawSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != log_id:
            self.last_loaded_sequence = self._load_sequence_uncached(log_id)
            self.last_loaded_sequence_id = log_id

        return self.last_loaded_sequence
