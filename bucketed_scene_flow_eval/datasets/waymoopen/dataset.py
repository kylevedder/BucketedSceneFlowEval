import enum
from pathlib import Path

import numpy as np

from bucketed_scene_flow_eval.datasets.shared_datastructures import SceneFlowItem
from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import BucketedEPEEvaluator, Evaluator
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .waymo_supervised_flow import CATEGORY_MAP, WaymoSupervisedSceneFlowSequenceLoader


class EvalType(enum.Enum):
    BUCKETED_EPE = 0


class WaymoOpenSceneFlow:
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        root_dir: Path,
        subsequence_length: int = 2,
        cache_path: Path = Path("/tmp/"),
        eval_type: str = "bucketed_epe",
        with_rgb: bool = False,
        eval_args=dict(),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(root_dir)
        self.subsequence_length = subsequence_length
        self.cache_path = cache_path

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx()

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _load_dataset_to_sequence_subsequence_idx(self):
        cache_file = (
            self.cache_path
            / "waymo"
            / self.root_dir.parent.name
            / self.root_dir.name
            / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}.pkl"
        )
        if cache_file.exists():
            return load_pickle(cache_file)

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

    def _make_scene_sequence(
        self, subsequence_frames: list[SceneFlowItem], seq_id: str
    ) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: dict[Timestamp, RawSceneItem] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            percept_lookup[dataset_idx] = RawSceneItem(pc_frame=entry.pc, rgb_frames=entry.rgbs)

        return RawSceneSequence(percept_lookup, seq_id)

    def _make_query_scene_sequence(
        self,
        scene_sequence: RawSceneSequence,
        subsequence_frames: list[SceneFlowItem],
        subsequence_src_index: int,
        subsequence_tgt_index: int,
    ) -> QuerySceneSequence:
        # Build query scene sequence. This requires enumerating all points in the source frame.
        query_timestamps: list[Timestamp] = [
            subsequence_src_index,
            subsequence_tgt_index,
        ]
        source_entry = subsequence_frames[subsequence_src_index]

        pc_points_array = source_entry.pc.global_pc.points

        query_particles = QueryPointLookup(len(pc_points_array), subsequence_src_index)
        return QuerySceneSequence(scene_sequence, query_particles, query_timestamps)

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
        source_pc = source_entry.pc.global_pc.points
        target_pc = source_entry.flowed_pc.global_pc.points
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
        # Stack the false false array len(source_pc) times.

        particle_ids = np.arange(len(source_pc))
        is_valids = np.ones((len(source_pc),), dtype=bool)

        particle_trajectories[particle_ids] = (
            points,
            pc_class_ids,
            is_valids,
        )

        return particle_trajectories

    def __getitem__(
        self, dataset_idx, verbose: bool = False
    ) -> tuple[QuerySceneSequence, GroundTruthPointFlow]:
        if verbose:
            print(f"Waymo Open Scene Flow dataset __getitem__({dataset_idx}) start")

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
            print(f"Waymo Open Scene Flow dataset __getitem__({dataset_idx}) end")

        return query_scene_sequence, results_scene_sequence

    def evaluator(self) -> Evaluator:
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.BUCKETED_EPE:
            return BucketedEPEEvaluator(**self.eval_args)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
