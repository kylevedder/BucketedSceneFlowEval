from pathlib import Path
from typing import Any, Optional

import numpy as np

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    EgoLidarFlow,
    PointCloud,
    PointCloudFrame,
    PoseInfo,
    RGBFrame,
    RGBFrameLookup,
    SemanticClassId,
    SupervisedPointCloudFrame,
    TimeSyncedAVLidarData,
    TimeSyncedBaseAuxilaryData,
    TimeSyncedRawFrame,
    TimeSyncedSceneFlowFrame,
)
from bucketed_scene_flow_eval.interfaces import (
    AbstractAVLidarSequence,
    AbstractSequence,
    AbstractSequenceLoader,
    CachedSequenceLoader,
)
from bucketed_scene_flow_eval.utils import load_pickle

CATEGORY_MAP = {
    0: "BACKGROUND",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "SIGN",
    4: "CYCLIST",
}


class WaymoSupervisedSceneFlowSequence(AbstractAVLidarSequence):
    def __init__(self, sequence_folder: Path, verbose: bool = False):
        self.sequence_folder = Path(sequence_folder)
        self.sequence_files = sorted(self.sequence_folder.glob("*.pkl"))
        assert len(self.sequence_files) > 0, f"no frames found in {self.sequence_folder}"

    def __repr__(self) -> str:
        return f"WaymoRawSequence with {len(self)} frames"

    def __len__(self):
        return len(self.sequence_files)

    def _load_idx(self, idx: int):
        assert idx < len(
            self
        ), f"idx {idx} out of range, len {len(self)} for {self.sequence_folder}"
        pickle_path = self.sequence_files[idx]
        pkl = load_pickle(pickle_path, verbose=False)
        pc = PointCloud(pkl["car_frame_pc"])
        flow = pkl["flow"]
        labels = pkl["label"]
        pose = SE3.from_array(pkl["pose"])
        return pc, flow, labels, pose

    def cleanup_flow(self, flow):
        flow[np.isnan(flow)] = 0
        flow[np.isinf(flow)] = 0
        flow_speed = np.linalg.norm(flow, axis=1)
        flow[flow_speed > 30] = 0
        return flow

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = True
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        assert idx < len(
            self
        ), f"idx {idx} out of range, len {len(self)} for {self.sequence_folder}"

        ego_pc, ego_flow, idx_labels, idx_pose = self._load_idx(idx)
        # Unfortunatly, the flow has some artifacts that we need to clean up. These are very adhoc and will
        # need to be updated if the flow is updated.
        ego_flow = self.cleanup_flow(ego_flow)
        _, _, _, start_pose = self._load_idx(relative_to_idx)
        relative_pose = start_pose.inverse().compose(idx_pose)

        # From the Waymo Open dataset.proto:
        # // If the point is not annotated with scene flow information, class is set
        # // to -1. A point is not annotated if it is in a no-label zone or if its label
        # // bounding box does not have a corresponding match in the previous frame,
        # // making it infeasible to estimate the motion of the point.
        # // Otherwise, (vx, vy, vz) are velocity along (x, y, z)-axis for this point
        # // and class is set to one of the following values:
        # //  -1: no-flow-label, the point has no flow information.
        # //   0:  unlabeled or "background,", i.e., the point is not contained in a
        # //       bounding box.
        # //   1: vehicle, i.e., the point corresponds to a vehicle label box.
        # //   2: pedestrian, i.e., the point corresponds to a pedestrian label box.
        # //   3: sign, i.e., the point corresponds to a sign label box.
        # //   4: cyclist, i.e., the point corresponds to a cyclist label box.

        cleaned_idx_labels = idx_labels.astype(SemanticClassId)
        cleaned_idx_labels[cleaned_idx_labels == -1] = 0

        pc_frame = SupervisedPointCloudFrame(
            full_pc=ego_pc,
            pose=PoseInfo(sensor_to_ego=SE3.identity(), ego_to_global=relative_pose),
            mask=np.ones(len(ego_pc), dtype=bool),
            full_pc_classes=cleaned_idx_labels,
        )
        flow_wrapper = EgoLidarFlow(ego_flow, np.ones(len(ego_pc), dtype=bool))

        return (
            TimeSyncedSceneFlowFrame(
                pc=pc_frame,
                rgbs=RGBFrameLookup.empty(),
                flow=flow_wrapper,
                log_id=self.sequence_folder.name,
                log_idx=idx,
                log_timestamp=idx,
            ),
            TimeSyncedAVLidarData(
                is_ground_points=np.zeros(len(ego_pc), dtype=bool),
                in_range_mask=np.ones(len(ego_pc), dtype=bool),
            ),
        )

    def load_frame_list(
        self, relative_to_idx: Optional[int]
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        return [
            self.load(idx, relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]

    @property
    def log_id(self):
        return self.sequence_folder.name

    @staticmethod
    def category_ids() -> list[int]:
        return WaymoSupervisedSceneFlowSequenceLoader.category_ids()

    @staticmethod
    def category_id_to_name(category_id: int) -> str:
        return WaymoSupervisedSceneFlowSequenceLoader.category_id_to_name(category_id)

    @staticmethod
    def category_name_to_id(category_name: str) -> int:
        return WaymoSupervisedSceneFlowSequenceLoader.category_name_to_id(category_name)


class WaymoSupervisedSceneFlowSequenceLoader(CachedSequenceLoader):
    def __init__(
        self,
        sequence_dir: Path,
        log_subset: Optional[list[str]] = None,
        verbose: bool = False,
        with_rgb: bool = False,
    ):
        super().__init__()
        self.dataset_dir = Path(sequence_dir)
        self.with_rgb = with_rgb
        self.verbose = verbose
        assert self.dataset_dir.is_dir(), f"dataset_dir {sequence_dir} does not exist"

        # Load list of sequences from sequence_dir
        sequence_dir_lst = sorted(self.dataset_dir.glob("*/"))

        self.log_lookup = {e.name: e for e in sequence_dir_lst}

        # Intersect with log_subset
        if log_subset is not None:
            self.log_lookup = {
                k: v for k, v in sorted(self.log_lookup.items()) if k in set(log_subset)
            }

        self.log_lookup_keys = sorted(self.log_lookup.keys())

    def __len__(self):
        return len(self.log_lookup_keys)

    def __getitem__(self, idx):
        return self.load_sequence(self.log_lookup_keys[idx])

    def get_sequence_ids(self):
        return self.log_lookup_keys

    def _load_sequence_uncached(self, log_id: str) -> WaymoSupervisedSceneFlowSequence:
        sequence_folder = self.log_lookup[log_id]
        return WaymoSupervisedSceneFlowSequence(sequence_folder, verbose=self.verbose)

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
        return f"waymo_supervised_dataset_dir_{self.dataset_dir.name}_with_rgb_{self.with_rgb}"
