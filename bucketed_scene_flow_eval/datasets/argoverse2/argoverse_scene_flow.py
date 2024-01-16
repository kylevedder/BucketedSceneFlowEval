from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import PointCloud
from bucketed_scene_flow_eval.utils.loaders import load_feather

from . import ArgoverseRawSequence

CATEGORY_MAP = {
    -1: "BACKGROUND",
    0: "ANIMAL",
    1: "ARTICULATED_BUS",
    2: "BICYCLE",
    3: "BICYCLIST",
    4: "BOLLARD",
    5: "BOX_TRUCK",
    6: "BUS",
    7: "CONSTRUCTION_BARREL",
    8: "CONSTRUCTION_CONE",
    9: "DOG",
    10: "LARGE_VEHICLE",
    11: "MESSAGE_BOARD_TRAILER",
    12: "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    13: "MOTORCYCLE",
    14: "MOTORCYCLIST",
    15: "OFFICIAL_SIGNALER",
    16: "PEDESTRIAN",
    17: "RAILED_VEHICLE",
    18: "REGULAR_VEHICLE",
    19: "SCHOOL_BUS",
    20: "SIGN",
    21: "STOP_SIGN",
    22: "STROLLER",
    23: "TRAFFIC_LIGHT_TRAILER",
    24: "TRUCK",
    25: "TRUCK_CAB",
    26: "VEHICULAR_TRAILER",
    27: "WHEELCHAIR",
    28: "WHEELED_DEVICE",
    29: "WHEELED_RIDER",
}

CATEGORY_MAP_INV = {v: k for k, v in CATEGORY_MAP.items()}


class ArgoverseSceneFlowSequence(ArgoverseRawSequence):
    def __init__(
        self,
        log_id: str,
        dataset_dir: Path,
        flow_data_lst: Path,
        with_rgb: bool = False,
        with_classes: bool = False,
    ):
        super().__init__(log_id, dataset_dir, with_rgb=with_rgb)
        self.with_classes = with_classes

        # The flow data does not have a timestamp, so we need to just rely on the order of the files.
        self.flow_data_files = sorted(flow_data_lst.glob("*.feather"))

        assert len(self.timestamp_list) > len(
            self.flow_data_files
        ), f"More flow data files in {flow_data_lst} than pointclouds in {self.dataset_dir};  {len(self.timestamp_list)} vs {len(self.flow_data_files)}"

        # The first len(self.flow_data_files) timestamps have flow data.
        # We keep those timestamps, plus the final timestamp.
        self.timestamp_list = self.timestamp_list[: len(self.flow_data_files) + 1]

    @staticmethod
    def get_class_str(class_id: int) -> Optional[str]:
        if class_id not in CATEGORY_MAP:
            return None
        return CATEGORY_MAP[class_id]

    def _load_flow(
        self, idx, ego_pc_with_ground: PointCloud
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        # There is no flow information for the last pointcloud in the sequence.

        classes_0 = (
            np.ones(len(ego_pc_with_ground.points), dtype=np.int32) * CATEGORY_MAP_INV["BACKGROUND"]
        )
        if idx == len(self) - 1 or idx == -1:
            return None, None, classes_0
        flow_data_file = self.flow_data_files[idx]
        flow_data = load_feather(flow_data_file, verbose=False)
        is_valid_arr = flow_data["is_valid"].values

        # The flow data is stored as 3 1D arrays, one for each dimension.
        xs = flow_data["flow_tx_m"].values
        ys = flow_data["flow_ty_m"].values
        zs = flow_data["flow_tz_m"].values

        flow_0_1 = np.stack([xs, ys, zs], axis=1)

        if self.with_classes:
            classes_0 = flow_data["classes_0"].values

        return flow_0_1, is_valid_arr, classes_0

    def load(self, idx: int, relative_to_idx: int) -> dict[str, Any]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        timestamp = self.timestamp_list[idx]
        ego_pc_with_ground = self._load_pc(idx)
        if self.with_rgb:
            img = self._load_rgb(idx)
        else:
            img = None
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)
        (
            relative_global_frame_flow_0_1_with_ground,
            is_valid_flow_with_ground_arr,
            classes_0_with_ground,
        ) = self._load_flow(idx, ego_pc_with_ground)

        # fmt: off
        # Global frame PC is needed to compute the ground point mask.
        absolute_global_frame_pc = ego_pc_with_ground.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)


        relative_global_frame_pc_with_ground = ego_pc_with_ground.transform(relative_pose)
        relative_global_frame_pc_no_ground = relative_global_frame_pc_with_ground.mask_points(
            ~is_ground_points)

        relative_global_frame_no_ground_flowed_pc = relative_global_frame_pc_no_ground.copy()
        relative_global_frame_with_ground_flowed_pc = relative_global_frame_pc_with_ground.copy()
        if relative_global_frame_flow_0_1_with_ground is not None:
            relative_global_frame_with_ground_flowed_pc.points[is_valid_flow_with_ground_arr] += relative_global_frame_flow_0_1_with_ground[is_valid_flow_with_ground_arr]
            relative_global_frame_no_ground_flowed_pc.points = relative_global_frame_with_ground_flowed_pc[~is_ground_points].copy()


        ego_flowed_pc_no_ground = relative_global_frame_no_ground_flowed_pc.transform(
            relative_pose.inverse())
        ego_flowed_pc_with_ground = relative_global_frame_with_ground_flowed_pc.transform(
            relative_pose.inverse())

        ego_pc_no_ground = ego_pc_with_ground.mask_points(~is_ground_points)

        in_range_mask_with_ground = self.is_in_range(relative_global_frame_pc_with_ground)
        in_range_mask_no_ground = self.is_in_range(relative_global_frame_pc_no_ground)
        # fmt: on

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
            "relative_flowed_pc_with_ground": relative_global_frame_with_ground_flowed_pc,
            "pc_classes": classes_0_no_ground,
            "pc_classes_with_ground": classes_0_with_ground,
            "log_id": self.log_id,
            "log_idx": idx,
            "log_timestamp": timestamp,
        }

    @staticmethod
    def category_ids() -> list[int]:
        return ArgoverseSceneFlowSequenceLoader.category_ids()

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return ArgoverseSceneFlowSequenceLoader.category_id_to_name(category_id)

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return ArgoverseSceneFlowSequenceLoader.category_name_to_id(category_name)


class ArgoverseSceneFlowSequenceLoader:
    def __init__(
        self,
        raw_data_path: Union[Path, list[Path]],
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        use_gt_flow: bool = True,
        with_rgb: bool = False,
        log_subset: Optional[list[str]] = None,
    ):
        self.use_gt_flow = use_gt_flow
        raw_data_path = self._sanitize_raw_data_path(raw_data_path)
        flow_data_path = self._sanitize_flow_data_path(use_gt_flow, flow_data_path, raw_data_path)

        # Raw data folders
        self.sequence_id_to_raw_data = self._load_sequence_data(raw_data_path)
        # Flow data folders
        self.sequence_id_to_flow_data = self._load_sequence_data(flow_data_path)

        self.sequence_id_lst = sorted(
            set(self.sequence_id_to_raw_data.keys()).intersection(
                set(self.sequence_id_to_flow_data.keys())
            )
        )

        if log_subset is not None:
            self.sequence_id_lst = [
                sequence_id for sequence_id in self.sequence_id_lst if sequence_id in log_subset
            ]
        self.with_rgb = with_rgb
        self.last_loaded_sequence: Optional[ArgoverseSceneFlowSequence] = None
        self.last_loaded_sequence_id: Optional[str] = None

    def _sanitize_raw_data_path(self, raw_data_path: Union[Path, list[Path]]) -> list[Path]:
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)
        if isinstance(raw_data_path, Path):
            raw_data_path = [raw_data_path]

        assert isinstance(
            raw_data_path, list
        ), f"raw_data_path must be a Path, list of Paths, or a string, got {raw_data_path}"
        raw_data_path = [Path(path) for path in raw_data_path]
        # Make sure the paths exist
        for path in raw_data_path:
            assert path.exists(), f"raw_data_path {path} does not exist"
        return raw_data_path

    def _sanitize_flow_data_path(
        self,
        use_gt_flow: bool,
        flow_data_path: Optional[Union[Path, list[Path]]],
        raw_data_path: list[Path],
    ) -> list[Path]:
        if not flow_data_path is None:
            return self._sanitize_raw_data_path(flow_data_path)

        # Load default flow data path
        flow_suffix = "_sceneflow_feather" if use_gt_flow else "_nsfp_flow_feather"
        flow_paths = [path.parent / (path.name + flow_suffix) for path in raw_data_path]
        return flow_paths

    def _load_sequence_data(self, path_info: Union[Path, list[Path]]) -> dict[str, Path]:
        if isinstance(path_info, Path):
            path_info = [path_info]

        sequence_folders: list[Path] = []
        for path in path_info:
            sequence_folders.extend(path.glob("*/"))

        sequence_id_to_path = {folder.stem: folder for folder in sorted(sequence_folders)}
        return sequence_id_to_path

    def __len__(self):
        return len(self.sequence_id_lst)

    def __getitem__(self, idx):
        return self.load_sequence(self.sequence_id_lst[idx])

    def get_sequence_ids(self):
        return self.sequence_id_lst

    def _sequence_id_to_idx(self, sequence_id: str):
        return self.sequence_id_lst.index(sequence_id)

    def _load_sequence_uncached(self, sequence_id: str) -> ArgoverseSceneFlowSequence:
        assert (
            sequence_id in self.sequence_id_to_flow_data
        ), f"sequence_id {sequence_id} does not exist"
        return ArgoverseSceneFlowSequence(
            sequence_id,
            self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_flow_data[sequence_id],
            with_rgb=self.with_rgb,
            with_classes=self.use_gt_flow,
        )

    def load_sequence(self, sequence_id: str) -> Optional[ArgoverseSceneFlowSequence]:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_id:
            self.last_loaded_sequence = self._load_sequence_uncached(sequence_id)
            self.last_loaded_sequence_id = sequence_id
        return self.last_loaded_sequence

    @staticmethod
    def category_ids() -> list[int]:
        return list(CATEGORY_MAP.keys())

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return CATEGORY_MAP[category_id]

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return {v: k for k, v in CATEGORY_MAP.items()}[category_name]
