from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from bucketed_scene_flow_eval.datastructures import PointCloud, SE3, SE2
import numpy as np

from . import ArgoverseRawSequence

CATEGORY_MAP = {
    -1: 'BACKGROUND',
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}

class ArgoverseSupervisedSceneFlowSequence(ArgoverseRawSequence):
    def __init__(self,
                 log_id: str,
                 dataset_dir: Path,
                 flow_data_lst: List[Tuple[int, Path]],
                 with_rgb: bool = True):
        super().__init__(log_id, dataset_dir, with_rgb=with_rgb)

        # Each flow contains info for the t and t+1 timestamps. This means the last pointcloud in the sequence
        # will not have a corresponding flow.
        self.timestamp_to_flow_map = {
            int(timestamp): flow_data_file
            for timestamp, flow_data_file in flow_data_lst
        }
        # Make sure all of the timestamps in self.timestamp_list *other than the last timestamp* have a corresponding flow.
        assert set(self.timestamp_list[:-1]) == set(
            self.timestamp_to_flow_map.keys(
            )), f'Flow data missing for some timestamps in {self.dataset_dir}'

    @staticmethod
    def get_class_str(class_id: int) -> Optional[str]:
        if class_id not in CATEGORY_MAP:
            return None
        return CATEGORY_MAP[class_id]

    def _load_flow(self, idx) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        # There is no flow information for the last pointcloud in the sequence.
        if idx == len(self) - 1 or idx == -1:
            return None, None
        timestamp = self.timestamp_list[idx]
        flow_data_file = self.timestamp_to_flow_map[timestamp]
        flow_info = dict(np.load(flow_data_file))
        flow_0_1 = flow_info['flow_0_1']
        classes_0 = flow_info['classes_0']
        return flow_0_1, classes_0

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
        flow_0_1, classes_0 = self._load_flow(idx)
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)

        relative_pose = start_pose.inverse().compose(idx_pose)

        # Global frame PC is needed to compute the ground point mask.
        absolute_global_frame_pc = ego_pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)

        relative_global_frame_pc_with_ground = ego_pc.transform(relative_pose)
        relative_global_frame_pc = relative_global_frame_pc_with_ground.mask_points(
            ~is_ground_points)
        if flow_0_1 is not None:
            ix_plus_one_pose = self._load_pose(idx + 1)
            relative_pose_plus_one = start_pose.inverse().compose(
                ix_plus_one_pose)
            ego_flowed_pc = ego_pc.flow(flow_0_1)
            ego_flowed_pc_no_ground = ego_flowed_pc.mask_points(
                ~is_ground_points)
            relative_global_frame_flowed_pc = ego_flowed_pc.transform_masked(
                relative_pose_plus_one, ~is_ground_points)
            relative_global_frame_flowed_pc_no_ground = relative_global_frame_flowed_pc.mask_points(
                ~is_ground_points)
            classes_0_no_ground = classes_0[~is_ground_points]
        else:
            ego_flowed_pc = None
            ego_flowed_pc_no_ground = None
            relative_global_frame_flowed_pc_no_ground = None
            relative_global_frame_flowed_pc = None
            classes_0 = None
            classes_0_no_ground = None

        ego_pc_no_ground = ego_pc.mask_points(~is_ground_points)

        in_range_mask = self.is_in_range(relative_global_frame_pc_with_ground)
        in_range_mask_no_ground = self.is_in_range(relative_global_frame_pc)

        return {
            "ego_pc": ego_pc_no_ground,
            "ego_pc_with_ground": ego_pc,
            "ego_flowed_pc": ego_flowed_pc_no_ground,
            "ego_flowed_pc_with_ground": ego_flowed_pc,
            "relative_pc": relative_global_frame_pc,
            "relative_pc_with_ground": relative_global_frame_pc_with_ground,
            "is_ground_points": is_ground_points,
            "in_range_mask": in_range_mask_no_ground,
            "in_range_mask_with_ground": in_range_mask,
            "rgb": img,
            "rgb_camera_projection": self.rgb_camera_projection,
            "rgb_camera_ego_pose": self.rgb_camera_ego_pose,
            "relative_pose": relative_pose,
            "relative_flowed_pc": relative_global_frame_flowed_pc_no_ground,
            "relative_flowed_pc_with_ground": relative_global_frame_flowed_pc,
            "pc_classes": classes_0_no_ground,
            "pc_classes_with_ground": classes_0,
            "log_id": self.log_id,
            "log_idx": idx,
            "log_timestamp": timestamp,
        }

    @staticmethod
    def category_ids() -> List[int]:
        return ArgoverseSupervisedSceneFlowSequenceLoader.category_ids()

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return ArgoverseSupervisedSceneFlowSequenceLoader.category_id_to_name(
            category_id)

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return ArgoverseSupervisedSceneFlowSequenceLoader.category_name_to_id(
            category_name)


class ArgoverseSupervisedSceneFlowSequenceLoader():
    def __init__(self,
                 raw_data_path: Path,
                 with_rgb: bool = True,
                 flow_data_path: Optional[Path] = None,
                 log_subset: Optional[List[str]] = None,
                 num_sequences: Optional[int] = None):

        self.raw_data_path = Path(raw_data_path)
        assert self.raw_data_path.is_dir(
        ), f'raw_data_path {raw_data_path} does not exist'

        self.with_rgb = with_rgb

        self.flow_data_path = flow_data_path
        if self.flow_data_path is None:
            self.flow_data_path = self.raw_data_path.parent / (
                self.raw_data_path.name + '_sceneflow')
        else:
            self.flow_data_path = Path(flow_data_path)

        assert self.flow_data_path.is_dir(
        ), f'flow_data_path {flow_data_path} does not exist'

        # Convert folder of flow NPZ files to a lookup table for different sequences
        flow_data_files = sorted(self.flow_data_path.glob('*.npz'))
        self.sequence_id_to_flow_lst = defaultdict(list)
        for flow_data_file in flow_data_files:
            sequence_id, timestamp = flow_data_file.stem.split('_')
            timestamp = int(timestamp)
            self.sequence_id_to_flow_lst[sequence_id].append(
                (timestamp, flow_data_file))

        self.sequence_id_to_raw_data = {}
        for sequence_id in self.sequence_id_to_flow_lst.keys():
            sequence_folder = self.raw_data_path / sequence_id
            assert sequence_folder.is_dir(
            ), f'sequence_folder {sequence_folder} does not exist'
            self.sequence_id_to_raw_data[sequence_id] = sequence_folder

        self.sequence_id_lst = sorted(self.sequence_id_to_flow_lst.keys())
        if log_subset is not None:
            self.sequence_id_lst = [
                sequence_id for sequence_id in self.sequence_id_lst
                if sequence_id in log_subset
            ]

        if num_sequences is not None:
            self.sequence_id_lst = self.sequence_id_lst[:num_sequences]

        self.last_loaded_sequence : Optional[ArgoverseSupervisedSceneFlowSequence] = None
        self.last_loaded_sequence_id : Optional[str] = None

    def __len__(self):
        return len(self.sequence_id_lst)

    def __getitem__(self, idx):
        return self.load_sequence(self.sequence_id_lst[idx])

    def get_sequence_ids(self):
        return self.sequence_id_lst

    def _sequence_id_to_idx(self, sequence_id: str):
        return self.sequence_id_lst.index(sequence_id)

    def _load_sequence_uncached(
            self, sequence_id: str) -> ArgoverseSupervisedSceneFlowSequence:
        assert sequence_id in self.sequence_id_to_flow_lst, f'sequence_id {sequence_id} does not exist'
        return ArgoverseSupervisedSceneFlowSequence(
            sequence_id,
            self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_flow_lst[sequence_id],
            with_rgb=self.with_rgb)

    def load_sequence(
            self, sequence_id: str) -> ArgoverseSupervisedSceneFlowSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_id:
            self.last_loaded_sequence = self._load_sequence_uncached(
                sequence_id)
            self.last_loaded_sequence_id = sequence_id
        return self.last_loaded_sequence

    @staticmethod
    def category_ids() -> List[int]:
        return list(CATEGORY_MAP.keys())

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return CATEGORY_MAP[category_id]

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return {v: k for k, v in CATEGORY_MAP.items()}[category_name]
