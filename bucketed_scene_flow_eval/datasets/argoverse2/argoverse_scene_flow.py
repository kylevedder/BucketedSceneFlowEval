from pathlib import Path
from typing import Optional, Union

import numpy as np

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_raw_data import (
    DEFAULT_POINT_CLOUD_RANGE,
    ArgoverseRawSequence,
    PointCloudRange,
)
from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    MaskArray,
    PointCloud,
    SemanticClassId,
    SemanticClassIdArray,
    SupervisedPointCloudFrame,
    TimeSyncedAVLidarData,
    TimeSyncedRawFrame,
    TimeSyncedSceneFlowFrame,
    VectorArray,
)
from bucketed_scene_flow_eval.interfaces import (
    AbstractAVLidarSequence,
    CachedSequenceLoader,
)
from bucketed_scene_flow_eval.utils.loaders import load_feather

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


class ArgoverseSceneFlowSequence(ArgoverseRawSequence, AbstractAVLidarSequence):
    def __init__(
        self,
        log_id: str,
        dataset_dir: Path,
        flow_dir: Path,
        with_rgb: bool = False,
        with_auxillary_pc: bool = False,
        with_classes: bool = False,
        **kwargs,
    ):
        super().__init__(
            log_id,
            dataset_dir,
            with_rgb=with_rgb,
            with_auxillary_pc=with_auxillary_pc,
            **kwargs,
        )
        self.with_classes = with_classes
        self.flow_data_files: list[Path] = []
        self._prep_flow(flow_dir)

    def _prep_flow(self, flow_dir: Path):
        # The flow data does not have a timestamp, so we need to just rely on the order of the files.
        # Only select files that have number and nothing else, e.g. 0000000069.feather, not 0000000069_occ.feather
        self.flow_data_files = sorted(
            file for file in flow_dir.glob("*.feather") if file.stem.isdigit()
        )

        assert len(self.timestamp_list) > len(
            self.flow_data_files
        ), f"More flow data files in {flow_dir} than pointclouds in {self.dataset_dir};  {len(self.timestamp_list)} vs {len(self.flow_data_files)}"

        # The first len(self.flow_data_files) timestamps have flow data.
        # We keep those timestamps, plus the final timestamp.
        self.timestamp_list = self.timestamp_list[: len(self.flow_data_files) + 1]

    @staticmethod
    def get_class_str(class_id: SemanticClassId) -> Optional[str]:
        class_id_int = int(class_id)
        if class_id_int not in CATEGORY_MAP:
            return None
        return CATEGORY_MAP[class_id_int]

    def _make_default_classes(self, pc: PointCloud) -> SemanticClassIdArray:
        return np.ones(len(pc.points), dtype=SemanticClassId) * CATEGORY_MAP_INV["BACKGROUND"]

    def _load_flow_feather(
        self, idx: int, classes_0: SemanticClassIdArray
    ) -> tuple[VectorArray, MaskArray, SemanticClassIdArray]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        # There is no flow information for the last pointcloud in the sequence.

        assert (
            idx != len(self) - 1
        ), f"idx {idx} is the last frame in the sequence, which has no flow data"
        assert idx >= 0, f"idx {idx} is out of range"
        flow_data_file = self.flow_data_files[idx]
        flow_data = load_feather(flow_data_file, verbose=False)
        is_valid_arr = flow_data["is_valid"].values.astype(bool)
        # Ensure that is_valid_arr is a boolean array.
        assert (
            is_valid_arr.dtype == bool
        ), f"is_valid_arr must be a boolean array, got {is_valid_arr.dtype} from {flow_data_file.absolute()}"

        # The flow data is stored as 3 1D arrays, one for each dimension.
        xs = flow_data["flow_tx_m"].values
        ys = flow_data["flow_ty_m"].values
        zs = flow_data["flow_tz_m"].values

        flow_0_1 = np.stack([xs, ys, zs], axis=1)

        if self.with_classes:
            classes_0 = flow_data["classes_0"].values

        return flow_0_1, is_valid_arr, classes_0

    def _make_tssf_item(
        self, raw_item: TimeSyncedRawFrame, classes_0: SemanticClassIdArray, flow: EgoLidarFlow
    ) -> TimeSyncedSceneFlowFrame:
        supervised_pc = SupervisedPointCloudFrame(
            **vars(raw_item.pc),
            full_pc_classes=classes_0,
        )
        return TimeSyncedSceneFlowFrame(
            pc=supervised_pc,
            auxillary_pc=raw_item.auxillary_pc,
            rgbs=raw_item.rgbs,
            log_id=raw_item.log_id,
            log_idx=raw_item.log_idx,
            log_timestamp=raw_item.log_timestamp,
            flow=flow,
        )

    def _load_no_flow(
        self, raw_item: TimeSyncedRawFrame, metadata: TimeSyncedAVLidarData
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        classes_0 = self._make_default_classes(raw_item.pc.pc)
        flow = EgoLidarFlow.make_no_flow(len(classes_0))
        return self._make_tssf_item(raw_item, classes_0, flow), metadata

    def _load_with_flow(
        self,
        raw_item: TimeSyncedRawFrame,
        metadata: TimeSyncedAVLidarData,
        idx: int,
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        (
            ego_flow_with_ground,
            is_valid_flow_with_ground_arr,
            classes_0_with_ground,
        ) = self._load_flow_feather(idx, self._make_default_classes(raw_item.pc.pc))
        flow = EgoLidarFlow(full_flow=ego_flow_with_ground, mask=is_valid_flow_with_ground_arr)
        return (self._make_tssf_item(raw_item, classes_0_with_ground, flow), metadata)

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = True
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        raw_item, metadata = super().load(idx, relative_to_idx)

        if with_flow:
            return self._load_with_flow(raw_item, metadata, idx)
        else:
            return self._load_no_flow(raw_item, metadata)

    def load_frame_list(
        self, relative_to_idx: Optional[int] = 0
    ) -> list[tuple[TimeSyncedRawFrame, TimeSyncedAVLidarData]]:
        return [
            self.load(
                idx=idx,
                relative_to_idx=(relative_to_idx if relative_to_idx is not None else idx),
                with_flow=(idx != len(self) - 1),
            )
            for idx in range(len(self))
        ]

    @staticmethod
    def category_ids() -> list[int]:
        return ArgoverseSceneFlowSequenceLoader.category_ids()

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return ArgoverseSceneFlowSequenceLoader.category_id_to_name(category_id)

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return ArgoverseSceneFlowSequenceLoader.category_name_to_id(category_name)


class ArgoverseSceneFlowSequenceLoader(CachedSequenceLoader):
    def __init__(
        self,
        raw_data_path: Union[Path, list[Path]],
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        use_gt_flow: bool = True,
        log_subset: Optional[list[str]] = None,
        **kwargs,
    ):
        self.load_sequence_kwargs = kwargs
        self._setup_raw_data(raw_data_path, use_gt_flow)
        self._setup_flow_data(use_gt_flow, flow_data_path)
        self._subset_log(log_subset)

    def _setup_raw_data(
        self,
        raw_data_path: Union[Path, list[Path]],
        use_gt_flow: bool,
    ):
        super().__init__()
        self.use_gt_flow = use_gt_flow
        self.raw_data_path = self._sanitize_raw_data_path(raw_data_path)

        # Raw data folders
        self.sequence_id_to_raw_data = self._load_sequence_data(self.raw_data_path)
        assert len(self.sequence_id_to_raw_data) > 0, f"No raw data found in {self.raw_data_path}"

        self.sequence_id_lst: list[str] = sorted(self.sequence_id_to_raw_data.keys())

    def _setup_flow_data(
        self, use_gt_flow: bool, flow_data_path: Optional[Union[Path, list[Path]]]
    ):
        self.flow_data_path = self._sanitize_flow_data_path(
            use_gt_flow, flow_data_path, self.raw_data_path
        )
        # Flow data folders
        self.sequence_id_to_flow_data = self._load_sequence_data(self.flow_data_path)

        # Make sure both raw and flow have non-zero number of entries.

        assert (
            len(self.sequence_id_to_flow_data) > 0
        ), f"No flow data found in {self.flow_data_path}"

        self.sequence_id_lst = sorted(
            set(self.sequence_id_lst).intersection(set(self.sequence_id_to_flow_data.keys()))
        )

    def _subset_log(self, log_subset: Optional[list[str]]):
        if log_subset is not None:
            self.sequence_id_lst = [
                sequence_id for sequence_id in self.sequence_id_lst if sequence_id in log_subset
            ]
            assert len(self.sequence_id_lst) > 0, f"No sequences found in log_subset {log_subset}"

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
            assert path.exists(), f"path {path} does not exist"
            sequence_folders.extend(path.glob("*/"))

        sequence_id_to_path = {folder.stem: folder for folder in sorted(sequence_folders)}
        return sequence_id_to_path

    def __len__(self):
        return len(self.sequence_id_lst)

    def load_sequence(self, sequence_id: str) -> ArgoverseSceneFlowSequence:
        return super().load_sequence(sequence_id)

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
            with_classes=self.use_gt_flow,
            **self.load_sequence_kwargs,
        )

    @staticmethod
    def category_ids() -> list[int]:
        return list(CATEGORY_MAP.keys())

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return CATEGORY_MAP[category_id]

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return {v: k for k, v in CATEGORY_MAP.items()}[category_name]

    def cache_folder_name(self) -> str:
        return f"av2_raw_data_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_flow_data_path_{self.flow_data_path}"


class ArgoverseNoFlowSequence(ArgoverseSceneFlowSequence):
    def _prep_flow(self, flow_dir: Path):
        pass

    def _load_flow_feather(
        self, idx: int, classes_0: SemanticClassIdArray
    ) -> tuple[VectorArray, MaskArray, SemanticClassIdArray]:
        raise NotImplementedError("No flow data available for ArgoverseNoFlowSequence")

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = False
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        return super().load(idx, relative_to_idx, with_flow=False)


class ArgoverseNoFlowSequenceLoader(ArgoverseSceneFlowSequenceLoader):
    def __init__(
        self,
        raw_data_path: Union[Path, list[Path]],
        log_subset: Optional[list[str]] = None,
        **kwargs,
    ):
        self.load_sequence_kwargs = kwargs
        self._setup_raw_data(
            raw_data_path=raw_data_path,
            use_gt_flow=False,
        )
        self._subset_log(log_subset)

    def _load_sequence_uncached(self, sequence_id: str) -> ArgoverseNoFlowSequence:
        assert (
            sequence_id in self.sequence_id_to_raw_data
        ), f"sequence_id {sequence_id} does not exist"
        return ArgoverseNoFlowSequence(
            sequence_id,
            self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_raw_data[sequence_id],
            with_classes=False,
            **self.load_sequence_kwargs,
        )

    def cache_folder_name(self) -> str:
        return f"av2_raw_data_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_No_flow_data_path"
