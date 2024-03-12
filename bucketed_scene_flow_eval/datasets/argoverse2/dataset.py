import copy
import enum
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from bucketed_scene_flow_eval.datasets.shared_datastructures import (
    RawItem,
    SceneFlowItem,
)
from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .argoverse_scene_flow import (
    CATEGORY_MAP,
    ArgoverseNoFlowSequence,
    ArgoverseNoFlowSequenceLoader,
    ArgoverseSceneFlowSequenceLoader,
)
from .av2_metacategories import BUCKETED_METACATAGORIES, THREEWAY_EPE_METACATAGORIES


class EvalType(enum.Enum):
    BUCKETED_EPE = 0
    THREEWAY_EPE = 1


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
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        use_cache=True,
        load_flow: bool = True,
    ) -> None:
        self.use_cache = use_cache
        self.with_ground = with_ground
        self.use_gt_flow = use_gt_flow
        if load_flow:
            self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
                root_dir, with_rgb=with_rgb, use_gt_flow=use_gt_flow, flow_data_path=flow_data_path
            )
        else:
            self.sequence_loader = ArgoverseNoFlowSequenceLoader(root_dir, with_rgb=with_rgb)
        self.subsequence_length = subsequence_length
        self.cache_path = self._cache_path(cache_root, root_dir)

        self.with_rgb = with_rgb
        self.flow_data_path = flow_data_path

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

    def _get_cache_file(self) -> Path:
        flow_data_path_name = "None"
        if not (self.flow_data_path is None):
            if isinstance(self.flow_data_path, list):
                flow_data_path_name = "_".join(
                    [Path(flow_data_path_part).stem for flow_data_path_part in self.flow_data_path]
                )
            else:
                flow_data_path_name = self.flow_data_path.stem
        flow_data_path_name = flow_data_path_name.strip()
        cache_file = (
            self.cache_path
            / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}_use_gt_{self.use_gt_flow}_with_rgb_{self.with_rgb}_with_ground_{self.with_ground}_flow_data_path_name_{flow_data_path_name}.pkl"
        )
        return cache_file

    def _load_dataset_to_sequence_subsequence_idx(self) -> list[tuple[int, int]]:
        cache_file = self._get_cache_file()
        if cache_file.exists() and self.use_cache:
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

    def _make_scene_sequence(
        self, subsequence_frames: list[SceneFlowItem], log_id: str
    ) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: dict[Timestamp, RawSceneItem] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            if not self.with_ground:
                entry.pc.mask = ~entry.is_ground_points
                entry.flowed_pc.mask = ~entry.is_ground_points

            rgb_frames = RGBFrameLookup.empty()
            if self.with_rgb:
                rgb_frames = entry.rgbs

            percept_lookup[dataset_idx] = RawSceneItem(pc_frame=entry.pc, rgb_frames=rgb_frames)

        return RawSceneSequence(percept_lookup, log_id)

    def _make_dummy_query_scene_sequence(
        self,
        scene_sequence: RawSceneSequence,
        subsequence_frames: Sequence[RawItem],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> QuerySceneSequence:
        query_timestamps: list[Timestamp] = [
            subsequence_src_index,
            subsequence_tgt_index,
        ]
        source_entry = subsequence_frames[subsequence_src_index]

        query_particles = QueryPointLookup(len(source_entry.pc.full_pc), subsequence_src_index)

        return QuerySceneSequence(scene_sequence, query_particles, query_timestamps)

    def _make_query_scene_sequence(
        self,
        scene_sequence: RawSceneSequence,
        subsequence_frames: Sequence[SceneFlowItem],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> QuerySceneSequence:
        query_scene_sequence = self._make_dummy_query_scene_sequence(
            scene_sequence, subsequence_frames, subsequence_src_index, subsequence_tgt_index
        )

        source_entry = subsequence_frames[subsequence_src_index]
        pc_points_array = source_entry.pc.full_global_pc.points
        is_valid_points_array = source_entry.in_range_mask & source_entry.pc.mask

        # Check that the in_range_points_array is the same size as the first dimension of pc_points_array
        assert len(is_valid_points_array) == len(
            pc_points_array
        ), f"Is valid points and pc points have different lengths. Is valid: {len(is_valid_points_array)}, pc points: {len(pc_points_array)}"

        particle_ids = np.arange(len(is_valid_points_array))
        query_scene_sequence.query_particles[particle_ids[is_valid_points_array]] = pc_points_array[
            is_valid_points_array
        ]
        return query_scene_sequence

    def _make_results_scene_sequence(
        self,
        query: QuerySceneSequence,
        subsequence_frames: list[SceneFlowItem],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> GroundTruthPointFlow:
        # Build query scene sequence. This requires enumerating all points in
        # the source frame and the associated flowed points.

        source_entry = subsequence_frames[subsequence_src_index]

        assert (
            source_entry.pc.mask == source_entry.flowed_pc.mask
        ).all(), f"Mask and flowed mask are different."

        source_pc = source_entry.pc.full_global_pc.points
        target_pc = source_entry.flowed_pc.full_global_pc.points

        is_valid_points_array = source_entry.in_range_mask & source_entry.pc.mask
        pc_class_ids = source_entry.pc_classes
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

        particle_trajectories[particle_ids[is_valid_points_array]] = (
            points[is_valid_points_array],
            pc_class_ids[is_valid_points_array],
            is_valids[is_valid_points_array],
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
                with_flow=(i != self.subsequence_length - 1),
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
        eval_args_copy = copy.deepcopy(self.eval_args)
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.BUCKETED_EPE:
            if "meta_class_lookup" not in eval_args_copy:
                eval_args_copy["meta_class_lookup"] = BUCKETED_METACATAGORIES
            return BucketedEPEEvaluator(**eval_args_copy)
        elif self.eval_type == EvalType.THREEWAY_EPE:
            if "meta_class_lookup" not in eval_args_copy:
                eval_args_copy["meta_class_lookup"] = THREEWAY_EPE_METACATAGORIES
            return ThreeWayEPEEvaluator(**eval_args_copy)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
