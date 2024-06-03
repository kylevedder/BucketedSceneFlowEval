from pathlib import Path

import numpy as np

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import (
    ArgoverseSceneFlowSequenceLoader,
)
from bucketed_scene_flow_eval.datasets.nuscenes.nuscenes_raw_data import (
    DEFAULT_POINT_CLOUD_RANGE,
    NuscDict,
    NuScenesRawSequence,
    NuScenesWithInstanceBoxes,
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

from .nuscenes_utils import create_splits_tokens

CATEGORY_MAP = {
    -1: "background",
    0: "animal",
    1: "human.pedestrian.adult",
    2: "human.pedestrian.child",
    3: "human.pedestrian.construction_worker",
    4: "human.pedestrian.personal_mobility",
    5: "human.pedestrian.police_officer",
    6: "human.pedestrian.stroller",
    7: "human.pedestrian.wheelchair",
    8: "movable_object.barrier",
    9: "movable_object.debris",
    10: "movable_object.pushable_pullable",
    11: "movable_object.trafficcone",
    12: "static_object.bicycle_rack",
    13: "vehicle.bicycle",
    14: "vehicle.bus.bendy",
    15: "vehicle.bus.rigid",
    16: "vehicle.car",
    17: "vehicle.construction",
    18: "vehicle.emergency.ambulance",
    19: "vehicle.emergency.police",
    20: "vehicle.motorcycle",
    21: "vehicle.trailer",
    22: "vehicle.truck",
}

CATEGORY_MAP_INV = {v: k for k, v in CATEGORY_MAP.items()}


class NuScenesSceneFlowSequence(NuScenesRawSequence, AbstractAVLidarSequence):
    def __init__(
        self,
        nusc: NuScenesWithInstanceBoxes,
        log_id: str,
        scene_info: NuscDict,
        flow_dir: Path,
        with_rgb: bool = False,
        with_classes: bool = False,
        point_cloud_range: PointCloudRange | None = DEFAULT_POINT_CLOUD_RANGE,
    ):
        super().__init__(
            nusc=nusc,
            log_id=log_id,
            scene_info=scene_info,
            point_cloud_range=point_cloud_range,
        )
        self.with_classes = with_classes
        self.with_rgb = with_rgb
        self.flow_dir = flow_dir

    @staticmethod
    def get_class_str(class_id: SemanticClassId) -> str | None:
        class_id_int = int(class_id)
        if class_id_int not in CATEGORY_MAP:
            return None
        return CATEGORY_MAP[class_id_int]

    def _make_default_classes(self, pc: PointCloud) -> SemanticClassIdArray:
        return np.ones(len(pc.points), dtype=SemanticClassId) * CATEGORY_MAP_INV["background"]

    def _load_flow_feather(
        self, idx: int, classes_0: SemanticClassIdArray
    ) -> tuple[VectorArray, MaskArray, SemanticClassIdArray]:
        assert idx < len(self), f"idx {idx} out of range, len {len(self)} for {self.dataset_dir}"
        # There is no flow information for the last pointcloud in the sequence.

        assert (
            idx != len(self) - 1
        ), f"idx {idx} is the last frame in the sequence, which has no flow data"
        assert idx >= 0, f"idx {idx} is out of range"
        flow_data_file = self.flow_dir / f"{idx}.feather"
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

    def _make_tssf_item(
        self, raw_item: TimeSyncedRawFrame, classes_0: SemanticClassIdArray, flow: EgoLidarFlow
    ) -> TimeSyncedSceneFlowFrame:
        supervised_pc = SupervisedPointCloudFrame(
            **vars(raw_item.pc),
            full_pc_classes=classes_0,
        )
        return TimeSyncedSceneFlowFrame(
            pc=supervised_pc,
            auxillary_pc=None,
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
        self, relative_to_idx: int | None = 0
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
        return NuScenesSceneFlowSequenceLoader.category_ids()

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return NuScenesSceneFlowSequenceLoader.category_id_to_name(category_id)

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return NuScenesSceneFlowSequenceLoader.category_name_to_id(category_name)


class NuScenesSceneFlowSequenceLoader(ArgoverseSceneFlowSequenceLoader, CachedSequenceLoader):
    def __init__(
        self,
        raw_data_path: Path | list[Path],
        split: str,
        nuscenes_version: str = "v1.0-mini",
        flow_data_path: Path | list[Path] | None = None,
        use_gt_flow: bool = True,
        with_rgb: bool = False,
        log_subset: list[str] | None = None,
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
        point_cloud_range: PointCloudRange | None = DEFAULT_POINT_CLOUD_RANGE,
    ):
        CachedSequenceLoader.__init__(self)
        self.use_gt_flow = use_gt_flow
        self.raw_data_path = self._sanitize_raw_data_path(raw_data_path)
        self.with_rgb = with_rgb
        self.expected_camera_shape = expected_camera_shape
        self.point_cloud_range = point_cloud_range
        self.nuscenes = NuScenesWithInstanceBoxes(nuscenes_version, raw_data_path)
        self.sequence_id_to_raw_data: dict[str, NuscDict] = {
            e["token"]: e for e in self.nuscenes.scene
        }
        self.sequence_id_to_raw_data: dict[str, NuscDict] = {
            k: self.sequence_id_to_raw_data[k] for k in create_splits_tokens(split, self.nuscenes)
        }
        self.sequence_id_lst: list[str] = sorted(self.sequence_id_to_raw_data.keys())
        self._setup_flow_data(use_gt_flow, flow_data_path)
        self._subset_log(log_subset)

    def _load_sequence_uncached(self, sequence_id: str) -> NuScenesSceneFlowSequence:
        assert (
            sequence_id in self.sequence_id_to_flow_data
        ), f"sequence_id {sequence_id} does not exist"
        return NuScenesSceneFlowSequence(
            nusc=self.nuscenes,
            log_id=sequence_id,
            scene_info=self.sequence_id_to_raw_data[sequence_id],
            flow_dir=self.sequence_id_to_flow_data[sequence_id],
            with_rgb=self.with_rgb,
            with_classes=self.use_gt_flow,
            point_cloud_range=self.point_cloud_range,
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
        return f"nuscenes_raw_data_with_rgb_{self.with_rgb}_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_flow_data_path_{self.flow_data_path}"


class NuScenesNoFlowSequence(NuScenesSceneFlowSequence):
    def _prep_flow(self, flow_dir: Path):
        pass

    def _load_flow_feather(
        self, idx: int, classes_0: SemanticClassIdArray
    ) -> tuple[VectorArray, MaskArray, SemanticClassIdArray]:
        raise NotImplementedError("No flow data available for NuScenesNoFlowSequence")

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = True
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        return super().load(idx, relative_to_idx, with_flow=False)


class NuScenesNoFlowSequenceLoader(NuScenesSceneFlowSequenceLoader):
    def __init__(
        self,
        raw_data_path: Path | list[Path],
        split: str,
        nuscenes_version: str = "v1.0-mini",
        with_rgb: bool = False,
        log_subset: list[str] | None = None,
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
        point_cloud_range: PointCloudRange | None = DEFAULT_POINT_CLOUD_RANGE,
    ):
        CachedSequenceLoader.__init__(self)
        self.use_gt_flow = False
        self.raw_data_path = raw_data_path
        self.with_rgb = with_rgb
        self.expected_camera_shape = expected_camera_shape
        self.point_cloud_range = point_cloud_range
        self.nuscenes = NuScenesWithInstanceBoxes(nuscenes_version, raw_data_path)
        self.sequence_id_to_raw_data: dict[str, NuscDict] = {
            e["token"]: e for e in self.nuscenes.scene
        }
        self.sequence_id_to_raw_data: dict[str, NuscDict] = {
            k: self.sequence_id_to_raw_data[k] for k in create_splits_tokens(split, self.nuscenes)
        }
        self.sequence_id_lst: list[str] = sorted(self.sequence_id_to_raw_data.keys())
        self._subset_log(log_subset)

    def _load_sequence_uncached(self, sequence_id: str) -> NuScenesNoFlowSequence:
        assert (
            sequence_id in self.sequence_id_to_raw_data
        ), f"sequence_id {sequence_id} does not exist"
        return NuScenesNoFlowSequence(
            nusc=self.nuscenes,
            log_id=sequence_id,
            scene_info=self.sequence_id_to_raw_data[sequence_id],
            flow_dir=self.sequence_id_to_raw_data[sequence_id],
            with_rgb=self.with_rgb,
            with_classes=False,
            point_cloud_range=self.point_cloud_range,
        )

    def cache_folder_name(self) -> str:
        return f"nuscenes_raw_data_with_rgb_{self.with_rgb}_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_No_flow_data_path"
