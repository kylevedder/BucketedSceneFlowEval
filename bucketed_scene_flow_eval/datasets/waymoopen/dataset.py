from bucketed_scene_flow_eval.datastructures import *
from pathlib import Path
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from typing import Tuple, Dict, List
import time
import numpy as np
import enum

from .waymo_supervised_flow import WaymoSupervisedSceneFlowSequenceLoader, CATEGORY_MAP
from bucketed_scene_flow_eval.eval import Evaluator


class EvalType(enum.Enum):
    RAW_EPE = 0
    SCALED_EPE = 1
    CLASS_THREEWAY_EPE = 2
    BUCKETED_EPE = 3


class WaymoOpenSceneFlow():
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """
    def __init__(self,
                 root_dir: Path,
                 subsequence_length: int = 2,
                 cache_path: Path = Path("/tmp/"),
                 eval_type: str = "bucketed_epe",
                 eval_args=dict()) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(root_dir)
        self.subsequence_length = subsequence_length
        self.cache_path = cache_path

        self.ego_pc_key = "ego_pc"
        self.ego_pc_flowed_key = "ego_flowed_pc"
        self.relative_pc_key = "relative_pc"
        self.relative_pc_flowed_key = "relative_flowed_pc"
        self.pc_classes_key = "pc_classes"

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx(
        )

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _load_dataset_to_sequence_subsequence_idx(self):
        cache_file = self.cache_path / "waymo" / self.root_dir.parent.name / self.root_dir.name / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}.pkl"
        if cache_file.exists():
            return load_pickle(cache_file)

        print("Building dataset index...")
        # Build map from dataset index to sequence and subsequence index.
        dataset_to_sequence_subsequence_idx = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(
                    len(sequence) - self.subsequence_length + 1):
                dataset_to_sequence_subsequence_idx.append(
                    (sequence_idx, subsequence_start_idx))

        print(
            f"Loaded {len(dataset_to_sequence_subsequence_idx)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_to_sequence_subsequence_idx)
        return dataset_to_sequence_subsequence_idx

    def __len__(self):
        return len(self.dataset_to_sequence_subsequence_idx)

    def _make_scene_sequence(
            self, subsequence_frames: List[Dict], seq_id : str) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: Dict[Timestamp, RawSceneItem] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            pc: PointCloud = entry[self.ego_pc_key]
            lidar_to_ego = SE3.identity()
            ego_to_world: SE3 = entry["relative_pose"]
            point_cloud_frame = PointCloudFrame(
                pc, PoseInfo(lidar_to_ego, ego_to_world))
            percept_lookup[dataset_idx] = RawSceneItem(
                pc_frame=point_cloud_frame, rgb_frame=None)

        return RawSceneSequence(percept_lookup, seq_id)

    def _make_query_scene_sequence(
            self, scene_sequence: RawSceneSequence,
            subsequence_frames: List[Dict], subsequence_src_index: int,
            subsequence_tgt_index: int) -> QuerySceneSequence:
        # Build query scene sequence. This requires enumerating all points in the source frame.
        query_timestamps: List[Timestamp] = [
            subsequence_src_index, subsequence_tgt_index
        ]
        source_entry = subsequence_frames[subsequence_src_index]

        pc_points_array = source_entry[self.relative_pc_key].points

        query_particles = QueryParticleLookup(len(pc_points_array),
                                              subsequence_src_index)
        return QuerySceneSequence(scene_sequence, query_particles,
                                  query_timestamps)

    def _make_results_scene_sequence(
            self, query: QuerySceneSequence, subsequence_frames: List[Dict],
            subsequence_src_index: int,
            subsequence_tgt_index: int) -> GroundTruthParticleTrajectories:
        # Build query scene sequence. This requires enumerating all points in
        # the source frame and the associated flowed points.

        source_entry = subsequence_frames[subsequence_src_index]
        source_pc = source_entry[self.relative_pc_key].points
        target_pc = source_entry[self.relative_pc_flowed_key].points
        pc_class_ids = source_entry[self.pc_classes_key]
        assert len(source_pc) == len(
            target_pc), "Source and target point clouds must be the same size."
        assert len(source_pc) == len(
            pc_class_ids
        ), f"Source point cloud and class ids must be the same size. Instead got {len(source_pc)} and {len(pc_class_ids)}."


        particle_trajectories = GroundTruthParticleTrajectories(
            len(source_pc),
            np.array([subsequence_src_index, subsequence_tgt_index]),
            query.query_particles.query_init_timestamp, CATEGORY_MAP)

        points = np.stack([source_pc, target_pc], axis=1)
        # Stack the false false array len(source_pc) times.
        is_occluded = np.tile([False, False], (len(source_pc), 1))

        particle_ids = np.arange(len(source_pc))
        is_valids = np.ones((len(source_pc), 2), dtype=bool)

        particle_trajectories[particle_ids] = (points, is_occluded,
                                               pc_class_ids, is_valids)

        return particle_trajectories

    def __getitem__(
        self,
        dataset_idx,
        verbose: bool = False
    ) -> Tuple[QuerySceneSequence, GroundTruthParticleTrajectories]:

        if verbose:
            print(
                f"Waymo Open Scene Flow dataset __getitem__({dataset_idx}) start"
            )

        sequence_idx, subsequence_start_idx = self.dataset_to_sequence_subsequence_idx[
            dataset_idx]

        # Load sequence
        sequence = self.sequence_loader[sequence_idx]

        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        # Load subsequence

        subsequence_frames = [
            sequence.load(subsequence_start_idx + i,
                          subsequence_start_idx + in_subsequence_tgt_index)
            for i in range(self.subsequence_length)
        ]

        scene_sequence = self._make_scene_sequence(subsequence_frames, sequence.log_id)

        query_scene_sequence = self._make_query_scene_sequence(
            scene_sequence, subsequence_frames, in_subsequence_src_index,
            in_subsequence_tgt_index)

        results_scene_sequence = self._make_results_scene_sequence(
            query_scene_sequence, subsequence_frames, in_subsequence_src_index,
            in_subsequence_tgt_index)

        if verbose:
            print(
                f"Waymo Open Scene Flow dataset __getitem__({dataset_idx}) end"
            )

        return query_scene_sequence, results_scene_sequence

    def evaluator(self) -> Evaluator:
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.RAW_EPE:
            return PerClassRawEPEEvaluator(**self.eval_args)
        elif self.eval_type == EvalType.SCALED_EPE:
            return PerClassScaledEPEEvaluator(**self.eval_args)
        elif self.eval_type == EvalType.CLASS_THREEWAY_EPE:
            return PerClassThreewayEPEEvaluator(**self.eval_args)
        elif self.eval_type == EvalType.BUCKETED_EPE:
            return BucketedEPEEvaluator(**self.eval_args)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
