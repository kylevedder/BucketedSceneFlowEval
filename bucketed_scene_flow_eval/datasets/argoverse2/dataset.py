import enum
from pathlib import Path
from typing import Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    PerClassRawEPEEvaluator,
    PerClassThreewayEPEEvaluator,
)
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .argoverse_scene_flow import CATEGORY_MAP, ArgoverseSceneFlowSequenceLoader
from .av2_metacategories import METACATAGORIES


class EvalType(enum.Enum):
    RAW_EPE = 0
    CLASS_THREEWAY_EPE = 1
    BUCKETED_EPE = 2


class Argoverse2SceneFlow:
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        subsequence_length: int = 2,
        with_ground: bool = True,
        with_rgb: bool = False,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
    ) -> None:
        self.with_ground = with_ground
        self.use_gt_flow = use_gt_flow
        self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
            root_dir, with_rgb=with_rgb, use_gt_flow=use_gt_flow
        )
        self.subsequence_length = subsequence_length
        self.cache_path = self._cache_path(cache_root, root_dir)

        # Lookup keys
        self.ego_pc_key = "ego_pc_with_ground"
        self.ego_pc_flowed_key = "ego_flowed_pc_with_ground"
        self.relative_pc_key = "relative_pc_with_ground"
        self.relative_pc_flowed_key = "relative_flowed_pc_with_ground"
        self.pc_classes_key = "pc_classes_with_ground"
        self.in_range_mask_key = "in_range_mask_with_ground"
        self.is_ground_key = "is_ground_points"

        self.with_rgb = with_rgb

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx()
        self.sequence_subsequence_idx_to_dataset_idx = {
            value: key for key, value in enumerate(self.dataset_to_sequence_subsequence_idx)
        }

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _cache_path(self, cache_root: Path, root_dir: Union[Path, list[Path]]) -> Path:
        if isinstance(root_dir, list):
            parent_name = "_".join([Path(root_dir_part).parent.name for root_dir_part in root_dir])
            folder_name = "_".join([Path(root_dir_part).name for root_dir_part in root_dir])
        else:
            parent_name = Path(root_dir).parent.name
            folder_name = Path(root_dir).name
        return cache_root / "argo" / parent_name / folder_name

    def _load_dataset_to_sequence_subsequence_idx(self) -> list[tuple[int, int]]:
        cache_file = (
            self.cache_path
            / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}_use_gt_{self.use_gt_flow}_with_rgb_{self.with_rgb}_with_ground_{self.with_ground}.pkl"
        )
        if cache_file.exists():
            cache_pkl = load_pickle(cache_file)
            # Sanity check that the cache is the right length by ensuring that it
            # has the same length as the sequence loader.
            if len(cache_pkl) == len(self.sequence_loader):
                return cache_pkl

        print("Building dataset index...")
        # Build map from dataset index to sequence and subsequence index.
        dataset_to_sequence_subsequence_idx = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(len(sequence) - self.subsequence_length + 1):
                dataset_to_sequence_subsequence_idx.append((sequence_idx, subsequence_start_idx))

        print(
            f"Loaded {len(dataset_to_sequence_subsequence_idx)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_to_sequence_subsequence_idx)
        return dataset_to_sequence_subsequence_idx

    def __len__(self):
        return len(self.dataset_to_sequence_subsequence_idx)

    def _av2_sequence_id_and_timestamp_to_idx(self, av2_sequence_id: str, timestamp: int) -> int:
        sequence_loader_idx = self.sequence_loader._sequence_id_to_idx(av2_sequence_id)
        sequence = self.sequence_loader.load_sequence(av2_sequence_id)
        sequence_idx = sequence._timestamp_to_idx(timestamp)
        return self.sequence_subsequence_idx_to_dataset_idx[(sequence_loader_idx, sequence_idx)]

    def _make_scene_sequence(self, subsequence_frames: list[dict], log_id: str) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: dict[Timestamp, RawSceneItem] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            pc: PointCloud = entry[self.ego_pc_key]
            lidar_to_ego = SE3.identity()
            ego_to_world: SE3 = entry["relative_pose"]

            mask = ~entry[self.is_ground_key]
            if self.with_ground:
                mask = np.ones_like(mask, dtype=bool)

            point_cloud_frame = PointCloudFrame(pc, PoseInfo(lidar_to_ego, ego_to_world), mask)

            rgb_to_ego: SE3 = entry["rgb_camera_ego_pose"]
            rgb_camera_projection: CameraProjection = entry["rgb_camera_projection"]
            if self.with_rgb:
                rgb_frame = RGBFrame(
                    entry["rgb"],
                    PoseInfo(rgb_to_ego, ego_to_world),
                    rgb_camera_projection,
                )
            else:
                rgb_frame = None
            percept_lookup[dataset_idx] = RawSceneItem(
                pc_frame=point_cloud_frame, rgb_frame=rgb_frame
            )

        return RawSceneSequence(percept_lookup, log_id)

    def _make_query_scene_sequence(
        self,
        scene_sequence: RawSceneSequence,
        subsequence_frames: list[dict],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> QuerySceneSequence:
        # Build query scene sequence. This requires enumerating all points in the source frame.
        query_timestamps: list[Timestamp] = [
            subsequence_src_index,
            subsequence_tgt_index,
        ]
        source_entry = subsequence_frames[subsequence_src_index]

        pc_points_array = source_entry[self.relative_pc_key].points
        in_range_points_array = source_entry[self.in_range_mask_key]

        query_particles = QueryPointLookup(len(pc_points_array), subsequence_src_index)
        particle_ids = np.arange(len(pc_points_array))
        query_particles[particle_ids[in_range_points_array]] = pc_points_array[
            in_range_points_array
        ]
        return QuerySceneSequence(scene_sequence, query_particles, query_timestamps)

    def _make_results_scene_sequence(
        self,
        query: QuerySceneSequence,
        subsequence_frames: list[dict],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> GroundTruthPointFlow:
        # Build query scene sequence. This requires enumerating all points in
        # the source frame and the associated flowed points.

        source_entry = subsequence_frames[subsequence_src_index]
        source_pc = source_entry[self.relative_pc_key].points
        target_pc = source_entry[self.relative_pc_flowed_key].points
        in_range_points_array = source_entry[self.in_range_mask_key]
        pc_class_ids = source_entry[self.pc_classes_key]
        assert len(source_pc) == len(
            target_pc
        ), "Source and target point clouds must be the same size."
        assert len(source_pc) == len(
            pc_class_ids
        ), f"Source point cloud and class ids must be the same size. Instead got {len(source_pc)} and {len(pc_class_ids)}."

        particle_trajectories = GroundTruthPointFlow(
            len(source_pc),
            np.array([subsequence_src_index, subsequence_tgt_index]),
            query.query_particles.query_init_timestamp,
            CATEGORY_MAP,
        )

        points = np.stack([source_pc, target_pc], axis=1)

        particle_ids = np.arange(len(source_pc))

        # is_valids needs to respect the points mask described in the query scene sequence pointcloud.
        first_timestamp = query.scene_sequence.get_percept_timesteps()[0]
        is_valids = query.scene_sequence[first_timestamp].pc_frame.mask

        assert len(is_valids) == len(
            points
        ), f"Is valids and points have different lengths. Is valids: {len(is_valids)}, points: {len(points)}"

        particle_trajectories[particle_ids[in_range_points_array]] = (
            points[in_range_points_array],
            pc_class_ids[in_range_points_array],
            is_valids[in_range_points_array],
        )

        return particle_trajectories

    def __getitem__(
        self, dataset_idx, verbose: bool = False
    ) -> tuple[QuerySceneSequence, GroundTruthPointFlow]:
        if verbose:
            print(f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) start")

        sequence_idx, subsequence_start_idx = self.dataset_to_sequence_subsequence_idx[dataset_idx]

        # Load sequence
        sequence = self.sequence_loader[sequence_idx]

        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        # Load subsequence

        subsequence_frames = [
            sequence.load(
                subsequence_start_idx + i,
                subsequence_start_idx + in_subsequence_tgt_index,
            )
            for i in range(self.subsequence_length)
        ]

        scene_sequence = self._make_scene_sequence(subsequence_frames, sequence.log_id)

        query_scene_sequence = self._make_query_scene_sequence(
            scene_sequence,
            subsequence_frames,
            in_subsequence_src_index,
            in_subsequence_tgt_index,
        )

        results_scene_sequence = self._make_results_scene_sequence(
            query_scene_sequence,
            subsequence_frames,
            in_subsequence_src_index,
            in_subsequence_tgt_index,
        )

        if verbose:
            print(f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) end")

        return query_scene_sequence, results_scene_sequence

    def evaluator(self) -> Evaluator:
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.RAW_EPE:
            return PerClassRawEPEEvaluator(**self.eval_args)
        elif self.eval_type == EvalType.CLASS_THREEWAY_EPE:
            return PerClassThreewayEPEEvaluator(**self.eval_args)
        elif self.eval_type == EvalType.BUCKETED_EPE:
            if "meta_class_lookup" not in self.eval_args:
                self.eval_args["meta_class_lookup"] = METACATAGORIES
            return BucketedEPEEvaluator(**self.eval_args)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
