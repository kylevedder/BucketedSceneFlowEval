from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from bucketed_scene_flow_eval.datastructures import PointCloud, SE3, SE2
import numpy as np

from . import ArgoverseRawSequence, CATEGORY_MAP


class ArgoverseUnsupervisedFlowSequence(ArgoverseRawSequence):

    def __init__(self, log_id: str, dataset_dir: Path, flow_data_lst: Path, with_rgb: bool = True):
        super().__init__(log_id, dataset_dir, with_rgb=with_rgb)

        # The flow data does not have a timestamp, so we need to just rely on the order of the files.
        self.flow_data_files = sorted(flow_data_lst.glob('*.npz'))

        assert len(self.timestamp_list) > len(
            self.flow_data_files
        ), f"More flow data files in {flow_data_lst} than pointclouds in {self.dataset_dir};  {len(self.timestamp_list)} vs {len(self.flow_data_files)}"

        # The first len(self.flow_data_files) timestamps have flow data.
        # We keep those timestamps, plus the final timestamp.
        self.timestamp_list = self.timestamp_list[:len(self.flow_data_files) +
                                                  1]

        self.background_class_id = {v : k for k, v in CATEGORY_MAP.items()}['BACKGROUND']

    def _load_flow(self, idx):
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        # There is no flow information for the last pointcloud in the sequence.
        if idx == len(self) - 1 or idx == -1:
            return None, None

        flow_data_file = self.flow_data_files[idx]
        flow_info = dict(np.load(flow_data_file))
        flow_0_1, valid_idxes = flow_info['flow'], flow_info['valid_idxes']
        return flow_0_1, valid_idxes

    def load(self, idx, relative_to_idx) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        ego_pc_with_ground = self._load_pc(idx)
        if self.with_rgb:
            img = self._load_rgb(idx)
        else:
            img = None
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)

        relative_global_frame_flow_0_1, flow_valid_idxes = self._load_flow(idx)

        # Global frame PC is needed to compute the ground point mask.
        absolute_global_frame_pc = ego_pc_with_ground.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)


        relative_global_frame_pc_with_ground = ego_pc_with_ground.transform(relative_pose)
        relative_global_frame_pc_no_ground = relative_global_frame_pc_with_ground.mask_points(
            ~is_ground_points)
        if relative_global_frame_flow_0_1 is not None:
            relative_global_frame_no_ground_flowed_pc = relative_global_frame_pc_no_ground.copy()
            relative_global_frame_no_ground_flowed_pc.points[flow_valid_idxes] += relative_global_frame_flow_0_1.reshape(-1, 3)

            relative_global_frame_pc_with_ground_flowed_pc = relative_global_frame_pc_with_ground.copy()
            relative_global_frame_pc_with_ground_flowed_pc.points[~is_ground_points] += relative_global_frame_no_ground_flowed_pc.points
        else:
            relative_global_frame_no_ground_flowed_pc = relative_global_frame_pc_no_ground.copy()
            relative_global_frame_pc_with_ground_flowed_pc = relative_global_frame_pc_with_ground.copy()

        ego_flowed_pc_no_ground = relative_global_frame_no_ground_flowed_pc.transform(
            relative_pose.inverse())
        ego_flowed_pc_with_ground = relative_global_frame_pc_with_ground_flowed_pc.transform(
            relative_pose.inverse())

        ego_pc_no_ground = ego_pc_with_ground.mask_points(~is_ground_points)

        in_range_mask_with_ground = self.is_in_range(relative_global_frame_pc_with_ground)
        in_range_mask_no_ground = self.is_in_range(relative_global_frame_pc_no_ground)

        classes_0_with_ground = np.ones(len(relative_global_frame_pc_with_ground.points), dtype=np.int32) * self.background_class_id
        classes_0_no_ground = classes_0_with_ground[~is_ground_points]
        

        return {
            "ego_pc": ego_pc_no_ground,
            "ego_pc_with_ground": ego_pc_with_ground,
            "ego_flowed_pc": ego_flowed_pc_no_ground,
            "ego_flowed_pc_with_ground": ego_flowed_pc_with_ground,
            "relative_pc": relative_global_frame_pc_no_ground,
            "relative_pc_with_ground": relative_global_frame_pc_with_ground,
            "is_ground_points": is_ground_points,
            "in_range_mask": in_range_mask_no_ground,
            "in_range_mask_with_ground": in_range_mask_with_ground,
            "rgb": img,
            "rgb_camera_projection": self.rgb_camera_projection,
            "rgb_camera_ego_pose": self.rgb_camera_ego_pose,
            "relative_pose": relative_pose,
            "relative_flowed_pc": relative_global_frame_no_ground_flowed_pc,
            "relative_flowed_pc_with_ground": relative_global_frame_pc_with_ground_flowed_pc,
            "pc_classes": classes_0_no_ground,
            "pc_classes_with_ground": classes_0_with_ground,
            "log_id": self.log_id,
            "log_idx": idx,
            "log_timestamp": timestamp,
        }


class ArgoverseUnsupervisedFlowSequenceLoader():

    def __init__(self,
                 raw_data_path: Path,
                 with_rgb: bool = True,
                 flow_data_path: Optional[Path] = None,
                 log_subset: Optional[List[str]] = None):
        self.raw_data_path = Path(raw_data_path)

        self.with_rgb = with_rgb

        if flow_data_path is None:
            self.flow_data_path = self.raw_data_path.parent / (
                self.raw_data_path.name + '_nsfp_flow')
        else:
            self.flow_data_path = Path(flow_data_path)

        assert self.raw_data_path.is_dir(
        ), f'raw_data_path {raw_data_path} does not exist'
        assert self.flow_data_path.is_dir(
        ), f'flow_data_path {flow_data_path} does not exist'

        # Raw data folders
        raw_data_folders = sorted(self.raw_data_path.glob('*/'))
        self.sequence_id_to_raw_data = {
            folder.stem: folder
            for folder in raw_data_folders
        }
        # Flow data folders
        flow_data_folders = sorted(self.flow_data_path.glob('*/'))
        self.sequence_id_to_flow_data = {
            folder.stem: folder
            for folder in flow_data_folders
        }

        self.sequence_id_list = sorted(
            set(self.sequence_id_to_raw_data.keys()).intersection(
                set(self.sequence_id_to_flow_data.keys())))

        if log_subset is not None:
            self.sequence_id_list = [
                sequence_id for sequence_id in self.sequence_id_list
                if sequence_id in log_subset
            ]

        self.last_loaded_sequence : Optional[ArgoverseUnsupervisedFlowSequence] = None
        self.last_loaded_sequence_id : Optional[str] = None

    def __getitem__(self, idx):
        return self.load_sequence(self.sequence_id_list[idx])

    def get_sequence_ids(self):
        return self.sequence_id_list

    def get_sequence_ids(self):
        return self.sequence_id_list

    def _load_sequence_uncached(
            self, sequence_id: str) -> ArgoverseUnsupervisedFlowSequence:
        assert sequence_id in self.sequence_id_to_flow_data, f'sequence_id {sequence_id} does not exist'
        return ArgoverseUnsupervisedFlowSequence(
            sequence_id, self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_flow_data[sequence_id], with_rgb=self.with_rgb)

    def load_sequence(self,
                      sequence_id: str) -> ArgoverseUnsupervisedFlowSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_id:
            self.last_loaded_sequence = self._load_sequence_uncached(sequence_id)
            self.last_loaded_sequence_id = sequence_id
        return self.last_loaded_sequence
