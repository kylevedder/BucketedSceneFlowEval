from bucketed_scene_flow_eval.datastructures import *
from pathlib import Path
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from typing import Tuple, Dict, List
import numpy as np

from .argoverse_supervised_scene_flow import ArgoverseSupervisedSceneFlowSequenceLoader, CATEGORY_MAP
from .argoverse_unsupervised_scene_flow import ArgoverseUnsupervisedFlowSequenceLoader
from .av2_metacategories import METACATAGORIES
from bucketed_scene_flow_eval.eval import Evaluator, PerClassRawEPEEvaluator, PerClassThreewayEPEEvaluator, BucketedEPEEvaluator
import enum


class EvalType(enum.Enum):
    RAW_EPE = 0
    CLASS_THREEWAY_EPE = 1
    BUCKETED_EPE = 2


class Argoverse2SceneFlow():
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """
    def __init__(self,
                 root_dir: Path,
                 subsequence_length: int = 2,
                 with_ground: bool = True,
                 with_rgb: bool = True,
                 cache_path: Path = Path("/tmp/"),
                 use_gt_flow: bool = True,
                 eval_type: str = "bucketed_epe",
                 eval_args=dict()) -> None:
        self.root_dir = Path(root_dir)
        if use_gt_flow:
            self.sequence_loader = ArgoverseSupervisedSceneFlowSequenceLoader(
                root_dir, with_rgb=with_rgb)
        else:
            self.sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
                root_dir, with_rgb=with_rgb)
        self.subsequence_length = subsequence_length
        self.cache_path = cache_path
        if with_ground:
            self.ego_pc_key = "ego_pc_with_ground"
            self.ego_pc_flowed_key = "ego_flowed_pc_with_ground"
            self.relative_pc_key = "relative_pc_with_ground"
            self.relative_pc_flowed_key = "relative_flowed_pc_with_ground"
            self.pc_classes_key = "pc_classes_with_ground"
            self.in_range_mask_key = "in_range_mask_with_ground"
        else:
            self.ego_pc_key = "ego_pc"
            self.ego_pc_flowed_key = "ego_flowed_pc"
            self.relative_pc_key = "relative_pc"
            self.relative_pc_flowed_key = "relative_flowed_pc"
            self.pc_classes_key = "pc_classes"
            self.in_range_mask_key = "in_range_mask"

        self.with_rgb = with_rgb

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx(
        )
        self.sequence_subsequence_idx_to_dataset_idx = {
            value: key
            for key, value in enumerate(
                self.dataset_to_sequence_subsequence_idx)
        }

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _load_dataset_to_sequence_subsequence_idx(
            self) -> List[Tuple[int, int]]:
        cache_file = self.cache_path / "argo" / self.root_dir.parent.name / self.root_dir.name / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}.pkl"
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

    def _av2_sequence_id_and_timestamp_to_idx(self, av2_sequence_id: str,
                                              timestamp: int) -> int:

        sequence_loader_idx = self.sequence_loader._sequence_id_to_idx(
            av2_sequence_id)
        sequence = self.sequence_loader.load_sequence(av2_sequence_id)
        sequence_idx = sequence._timestamp_to_idx(timestamp)
        return self.sequence_subsequence_idx_to_dataset_idx[(
            sequence_loader_idx, sequence_idx)]

    def _make_scene_sequence(
            self, subsequence_frames: List[Dict], log_id: str) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: Dict[Timestamp, RawSceneItem] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            pc: PointCloud = entry[self.ego_pc_key]
            lidar_to_ego = SE3.identity()
            ego_to_world: SE3 = entry["relative_pose"]
            point_cloud_frame = PointCloudFrame(
                pc, PoseInfo(lidar_to_ego, ego_to_world))

            rgb_to_ego: SE3 = entry["rgb_camera_ego_pose"]
            rgb_camera_projection: CameraProjection = entry[
                "rgb_camera_projection"]
            if self.with_rgb:
                rgb_frame = RGBFrame(entry["rgb"],
                                     PoseInfo(rgb_to_ego, ego_to_world),
                                     rgb_camera_projection)
            else:
                rgb_frame = None
            percept_lookup[dataset_idx] = RawSceneItem(
                pc_frame=point_cloud_frame, rgb_frame=rgb_frame)

        return RawSceneSequence(percept_lookup, log_id)

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
        in_range_points_array = source_entry[self.in_range_mask_key]

        query_particles = QueryParticleLookup(len(pc_points_array),
                                              subsequence_src_index)
        particle_ids = np.arange(len(pc_points_array))
        query_particles[particle_ids[in_range_points_array]] = pc_points_array[
            in_range_points_array]
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
        in_range_points_array = source_entry[self.in_range_mask_key]
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

        particle_trajectories[particle_ids[in_range_points_array]] = (
            points[in_range_points_array], is_occluded[in_range_points_array],
            pc_class_ids[in_range_points_array],
            is_valids[in_range_points_array])

        return particle_trajectories

    def __getitem__(
        self,
        dataset_idx,
        verbose: bool = False
    ) -> Tuple[QuerySceneSequence, GroundTruthParticleTrajectories]:

        if verbose:
            print(
                f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) start"
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
                f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) end"
            )

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
