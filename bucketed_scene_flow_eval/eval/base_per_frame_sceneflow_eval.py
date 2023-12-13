from bucketed_scene_flow_eval.datastructures import (
    EstimatedParticleTrajectories,
    GroundTruthParticleTrajectories,
    Timestamp,
    ParticleClassId,
)
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Set, Any, Union
import numpy as np
import pickle
import json
import enum
from .eval import Evaluator
from bucketed_scene_flow_eval.utils import save_json, save_pickle
import copy


@dataclass(frozen=True, eq=True, order=True, repr=True)
class BaseSplitKey:
    name: str
    distance_threshold: float
    speed_thresholds: Tuple[float, float]

    def __eq__(self, __value: object) -> bool:
        # TODO: This is a hack because the hash function works but the autogen eq function doesn't.
        return hash(self) == hash(__value)


@dataclass(frozen=True, eq=True, repr=True)
class BaseSplitValue:
    avg_epe: float
    count: int
    avg_speed: float

    def __eq__(self, __value: object) -> bool:
        # TODO: This is a hack because the hash function works but the autogen eq function doesn't.
        return hash(self) == hash(__value)


class BaseEvalFrameResult:
    def __init__(
        self,
        gt_world_points: np.ndarray,
        gt_class_ids: np.ndarray,
        gt_flow: np.ndarray,
        pred_flow: np.ndarray,
        class_id_to_name=lambda e: e,
        distance_thresholds: List[float] = [35, np.inf],
        max_speed_thresholds: List[Tuple[float, float]] = [(0, np.inf)],
    ):

        self.distance_thresholds = distance_thresholds
        self.max_speed_thresholds = max_speed_thresholds

        assert (gt_world_points.ndim == 2
                ), f"gt_world_points must be 3D, got {gt_world_points.ndim}"
        assert (
            gt_world_points.shape == gt_flow.shape
        ), f"gt_world_points and gt_flow must have the same shape, got {gt_world_points.shape} and {gt_flow.shape}"
        assert (gt_class_ids.ndim == 1
                ), f"gt_class_ids must be 1D, got {gt_class_ids.ndim}"
        assert gt_flow.shape == pred_flow.shape, f"gt_flow and pred_flow must have the same shape, got {gt_flow.shape} and {pred_flow.shape}"

        gt_speeds = np.linalg.norm(gt_flow, axis=1)

        scaled_gt_flow, scaled_pred_flow = self._scale_flows(
            gt_flow, pred_flow)

        scaled_epe_errors = np.linalg.norm(scaled_gt_flow - scaled_pred_flow,
                                           axis=1)

        self.class_error_dict = {
            k: v
            for k, v in
            self.make_splits(gt_world_points, gt_speeds, gt_class_ids,
                             scaled_epe_errors, class_id_to_name)
        }

    def _get_gt_classes(self, gt_class_ids: np.ndarray) -> Set[int]:
        return np.unique(gt_class_ids)

    def _get_distance_thresholds(self) -> List[float]:
        return self.distance_thresholds

    def _get_max_speed_thresholds(self) -> List[Tuple[float, float]]:
        return self.max_speed_thresholds

    def _scale_flows(self, gt_flow: np.ndarray,
                     pred_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return gt_flow, pred_flow

    def make_splits(
            self, gt_world_points, gt_speeds, gt_class_ids, epe_errors,
            class_id_to_name) -> List[Tuple[BaseSplitKey, BaseSplitValue]]:
        unique_gt_classes = self._get_gt_classes(gt_class_ids)
        distance_thresholds = self._get_distance_thresholds()
        speed_threshold_tuples = self._get_max_speed_thresholds()

        for speed_threshold_tuple in speed_threshold_tuples:
            min_speed_threshold, max_speed_threshold = speed_threshold_tuple
            within_speed_mask = (gt_speeds >= min_speed_threshold) & (
                gt_speeds < max_speed_threshold)
            for class_id in unique_gt_classes:
                class_matched_mask = gt_class_ids == class_id
                class_and_speed_mask = class_matched_mask & within_speed_mask
                # Early exiting for improved eval performance
                if class_and_speed_mask.sum() == 0:
                    continue
                for distance_threshold in distance_thresholds:
                    within_distance_mask = (np.linalg.norm(
                        gt_world_points[:, :2], ord=np.inf, axis=1) <
                                            distance_threshold)

                    match_mask = class_and_speed_mask & within_distance_mask
                    count = match_mask.sum()
                    # Early exiting for improved eval performance
                    if count == 0:
                        continue

                    avg_epe = np.sum(epe_errors[match_mask]) / count
                    split_avg_speed = np.mean(gt_speeds[match_mask])
                    class_name = class_id_to_name(class_id)
                    yield BaseSplitKey(class_name, distance_threshold,
                                       speed_threshold_tuple), BaseSplitValue(
                                           avg_epe, count, split_avg_speed)


class PerFrameSceneFlowEvaluator(Evaluator):
    def __init__(self, output_path: Path = Path("/tmp/frame_results")):
        self.eval_frame_results: List[BaseEvalFrameResult] = []
        self.output_path = output_path
        # print(f"Saving results to {self.output_path}")
        # make the directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_evaluator_list(
            evaluator_list: List["PerFrameSceneFlowEvaluator"]):
        assert len(evaluator_list
                   ) > 0, "evaluator_list must have at least one evaluator"

        return sum(evaluator_list)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other: "PerFrameSceneFlowEvaluator"):
        if isinstance(other, int):
            if other == 0:
                return self

        # Concatenate the eval_frame_results
        evaluator = copy.copy(self)
        evaluator.eval_frame_results.extend(other.eval_frame_results)
        return evaluator

    def __len__(self):
        return len(self.eval_frame_results)

    def _validate_inputs(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
    ):
        assert isinstance(
            predictions, EstimatedParticleTrajectories
        ), f"predictions must be a EstimatedParticleTrajectories, got {type(predictions)}"

        assert isinstance(
            ground_truth, GroundTruthParticleTrajectories
        ), f"ground_truth must be a GroundTruthParticleTrajectories, got {type(ground_truth)}"

        # Validate that the predictions and ground truth have the same number of predictions.

        assert len(predictions) == len(
            ground_truth
        ), f"predictions and ground_truth must have the same number of predictions, got {len(predictions)} and {len(ground_truth)}"

        assert (
            len(predictions) > 0
        ), f"predictions must have at least one prediction, got {len(predictions)}"

        # All Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more trajectories than
        # the Ground Truth Particle Trajectories.

        predictions_intersection_ground_truth = (predictions.is_valid
                                                 & ground_truth.is_valid)
        predictions_match_ground_truth = (
            predictions_intersection_ground_truth == ground_truth.is_valid)
        vectors = ground_truth.world_points[~predictions_match_ground_truth]
        assert (predictions_match_ground_truth).all(
        ), f"all ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching points: {(~predictions_match_ground_truth).sum()}. Violating vectors: {vectors}"

        # All timestamps for the Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more timestamps than
        # the Ground Truth Particle Trajectories.
        assert set(ground_truth.trajectory_timestamps).issubset(
            set(predictions.trajectory_timestamps)
        ), f"all timestamps for the ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching timestamps: {set(ground_truth.trajectory_timestamps) - set(predictions.trajectory_timestamps)}"

    def _get_indices_of_timestamps(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
        query_timestamp: Timestamp,
    ):
        # create an numpy array
        pred_timestamps = predictions.trajectory_timestamps

        traj_timestamps = ground_truth.trajectory_timestamps

        # index of first occurrence of each value
        sorter = np.argsort(pred_timestamps)

        matched_idxes = sorter[np.searchsorted(pred_timestamps,
                                               traj_timestamps,
                                               sorter=sorter)]

        # find the index of the query timestamp in traj_timestamps
        query_idx = np.where(traj_timestamps == query_timestamp)[0][0]

        return matched_idxes, query_idx

    def eval(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
        query_timestamp: Timestamp,
    ):
        self._validate_inputs(predictions, ground_truth)

        # Extract the ground truth entires for the timestamps that are in both the predictions and ground truth.
        # It could be that the predictions have more timestamps than the ground truth.

        matched_time_axis_indices, query_idx = self._get_indices_of_timestamps(
            predictions, ground_truth, query_timestamp)

        # We only support Scene Flow
        if query_idx != 0:
            raise NotImplementedError(
                "TODO: Handle query_idx != 0 when computing speed bucketing.")

        eval_particle_ids = ground_truth.valid_particle_ids()

        gt_is_valids = ground_truth.is_valid[
            eval_particle_ids][:, matched_time_axis_indices]

        pred_is_valids = predictions.is_valid[
            eval_particle_ids][:, matched_time_axis_indices]

        # Make sure that all the pred_is_valids are true if gt_is_valids is true.
        assert ((gt_is_valids & pred_is_valids) == gt_is_valids).all(
        ), f"all gt_is_valids must be true if pred_is_valids is true."

        gt_world_points = ground_truth.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_world_points = predictions.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        gt_is_occluded = ground_truth.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_is_occluded = predictions.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]

        gt_class_ids = ground_truth.cls_ids[eval_particle_ids]

        assert (
            gt_world_points.shape[1] == 2
        ), f"gt_world_points must have 2 timestamps; we only support Scene Flow. Instead we got {gt_world_points.shape[1]} dimensions."
        assert (
            pred_world_points.shape[1] == 2
        ), f"pred_world_points must have 2 timestamps; we only support Scene Flow. Instead we got {pred_world_points.shape[1]} dimensions."

        # Query index should have roughly the same values.
        assert np.isclose(
            gt_world_points[:, query_idx], pred_world_points[:, query_idx]
        ).all(
        ), f"gt_world_points and pred_world_points should have the same values for the query index, got {gt_world_points[:, query_idx]} and {pred_world_points[:, query_idx]}"

        pc1 = gt_world_points[:, 0]
        gt_pc2 = gt_world_points[:, 1]
        pred_pc2 = pred_world_points[:, 1]

        gt_flow = gt_pc2 - pc1
        pred_flow = pred_pc2 - pc1

        eval_frame_result = self._build_eval_frame_results(
            pc1, gt_class_ids, gt_flow, pred_flow, ground_truth)

        self.eval_frame_results.append(eval_frame_result)

    def _build_eval_frame_results(
            self, pc1: np.ndarray, gt_class_ids: np.ndarray,
            gt_flow: np.ndarray, pred_flow: np.ndarray,
            ground_truth: GroundTruthParticleTrajectories
    ) -> BaseEvalFrameResult:
        return BaseEvalFrameResult(pc1,
                                   gt_class_ids,
                                   gt_flow,
                                   pred_flow,
                                   class_id_to_name=ground_truth.pretty_name)

    def _save_intermediary_results(self):
        save_path = self.output_path / "eval_frame_results.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(save_path, self.eval_frame_results)

    def _category_to_per_frame_stats(
            self) -> Dict[BaseSplitKey, List[BaseSplitValue]]:
        # From list of dicts to dict of lists
        merged_class_error_dict = dict()
        for eval_frame_result in self.eval_frame_results:
            for k, v in eval_frame_result.class_error_dict.items():
                if k not in merged_class_error_dict:
                    merged_class_error_dict[k] = [v]
                else:
                    merged_class_error_dict[k].append(v)
        return merged_class_error_dict

    def _category_to_average_stats(
        self, merged_class_error_dict: Dict[BaseSplitKey, List[BaseSplitValue]]
    ) -> Dict[BaseSplitKey, BaseSplitValue]:
        # Compute the average EPE for each key
        result_dict = dict()
        for k in sorted(merged_class_error_dict.keys()):
            values = merged_class_error_dict[k]
            epes = np.array([v.avg_epe for v in values])
            counts = np.array([v.count for v in values])
            # Average of the epes weighted by the counts

            weighted_split_avg_speed = np.NaN
            weighted_average_epe = np.NaN
            if counts.sum() > 0:
                valid_counts_mask = counts > 0

                weighted_average_epe = np.average(
                    epes[valid_counts_mask],
                    weights=(counts[valid_counts_mask]))
                weighted_split_avg_speed = np.average(
                    np.array([v.avg_speed for v in values])[valid_counts_mask],
                    weights=(counts[valid_counts_mask]),
                )

            result_dict[k] = BaseSplitValue(weighted_average_epe, counts.sum(),
                                            weighted_split_avg_speed)
        return result_dict

    def _save_dict(self, path: Path, data: Dict[Any, float]):
        str_data = {str(k): v for k, v in data.items()}
        save_json(path, str_data)

    def _save_stats_tables(self, average_stats: Dict[BaseSplitKey,
                                                     BaseSplitValue]):
        assert (
            len(average_stats) > 0
        ), f"average_stats must have at least one entry, got {len(average_stats)}"

        unique_speed_threshold_tuples = sorted(
            set([k.speed_thresholds for k in average_stats.keys()]))
        unique_distance_thresholds = sorted(
            set([k.distance_threshold for k in average_stats.keys()]))
        unique_category_names = sorted(
            set([k.name for k in average_stats.keys()]))

        for distance_threshold in unique_distance_thresholds:
            raw_table_save_path = self.output_path / f"metric_table_{distance_threshold}.json"
            speed_table_save_path = self.output_path / f"speed_table_{distance_threshold}.json"

            # Rows are category names, columns are for speed buckets

            epe_dict = dict()
            speed_dict = dict()

            for category_name in unique_category_names:
                for speed_threshold_tuple in unique_speed_threshold_tuples:
                    key = BaseSplitKey(category_name, distance_threshold,
                                       speed_threshold_tuple)
                    avg_epe = np.NaN
                    avg_speed = np.NaN
                    if key in average_stats:
                        avg_epe = average_stats[key].avg_epe
                        avg_speed = average_stats[key].avg_speed
                    epe_dict[key] = avg_epe
                    speed_dict[key] = avg_speed

            Path(raw_table_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(speed_table_save_path).parent.mkdir(parents=True,
                                                     exist_ok=True)
            self._save_dict(raw_table_save_path, epe_dict)
            self._save_dict(speed_table_save_path, speed_dict)

    def compute_results(
            self,
            save_results: bool = True) -> Dict[BaseSplitKey, BaseSplitValue]:
        assert (len(self.eval_frame_results) >
                0), "Must call eval at least once before calling compute"
        if save_results:
            self._save_intermediary_results()

        # From list of dicts to dict of lists
        category_to_per_frame_stats = self._category_to_per_frame_stats()
        category_to_average_stats = self._category_to_average_stats(
            category_to_per_frame_stats)
        self._save_stats_tables(category_to_average_stats)
        return category_to_average_stats
