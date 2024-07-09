import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Set, Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    PointCloud,
    SemanticClassId,
    SemanticClassIdArray,
    TimeSyncedSceneFlowFrame,
    VectorArray,
)
from bucketed_scene_flow_eval.utils import save_json, save_pickle

from .eval import Evaluator


@dataclass(frozen=True, eq=True, order=True, repr=True)
class BaseSplitKey:
    name: str
    distance_threshold: float
    speed_thresholds: tuple[float, float]

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
        gt_world_points: PointCloud,
        gt_class_ids: SemanticClassIdArray,
        gt_flow: VectorArray,
        pred_flow: VectorArray,
        class_id_to_name: dict[SemanticClassId, str],
        distance_thresholds: list[float] = [35, np.inf],
        max_speed_thresholds: list[tuple[float, float]] = [(0, np.inf)],
    ):
        self.distance_thresholds = distance_thresholds
        self.max_speed_thresholds = max_speed_thresholds

        assert (
            gt_world_points.shape == gt_flow.shape
        ), f"gt_world_points and gt_flow must have the same shape, got {gt_world_points.shape} and {gt_flow.shape}"
        assert gt_class_ids.ndim == 1, f"gt_class_ids must be 1D, got {gt_class_ids.ndim}"
        assert (
            gt_flow.shape == pred_flow.shape
        ), f"gt_flow and pred_flow must have the same shape, got {gt_flow.shape} and {pred_flow.shape}"

        scaled_gt_flow, scaled_pred_flow = self._scale_flows(gt_flow, pred_flow)

        scaled_epe_errors = np.linalg.norm(scaled_gt_flow - scaled_pred_flow, axis=1)

        self.class_error_dict = {
            k: v
            for k, v in self.make_splits(
                gt_world_points,
                gt_flow,
                gt_class_ids,
                scaled_epe_errors,
                class_id_to_name,
            )
        }

    def _get_gt_classes(self, gt_class_ids: SemanticClassIdArray) -> SemanticClassIdArray:
        return np.unique(gt_class_ids)

    def _get_distance_thresholds(self) -> list[float]:
        return self.distance_thresholds

    def _get_max_speed_thresholds(self) -> list[tuple[float, float]]:
        return self.max_speed_thresholds

    def _scale_flows(
        self, gt_flow: np.ndarray, pred_flow: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return gt_flow, pred_flow

    def make_splits(
        self,
        gt_world_points: PointCloud,
        gt_flow: VectorArray,
        gt_class_ids: SemanticClassIdArray,
        epe_errors: np.ndarray,
        class_id_to_name: dict[SemanticClassId, str],
    ) -> Iterable[tuple[BaseSplitKey, BaseSplitValue]]:
        gt_speeds = np.linalg.norm(gt_flow, axis=1)
        unique_gt_classes = self._get_gt_classes(gt_class_ids)
        distance_thresholds = self._get_distance_thresholds()
        speed_threshold_tuples = self._get_max_speed_thresholds()

        for speed_threshold_tuple in speed_threshold_tuples:
            min_speed_threshold, max_speed_threshold = speed_threshold_tuple
            within_speed_mask = (gt_speeds >= min_speed_threshold) & (
                gt_speeds < max_speed_threshold
            )
            for class_id in unique_gt_classes:
                class_matched_mask = gt_class_ids == class_id
                class_and_speed_mask = class_matched_mask & within_speed_mask
                # Early exiting for improved eval performance
                if class_and_speed_mask.sum() == 0:
                    continue
                for distance_threshold in distance_thresholds:
                    within_distance_mask = (
                        np.linalg.norm(gt_world_points.points[:, :2], ord=np.inf, axis=1)
                        < distance_threshold
                    )

                    match_mask = class_and_speed_mask & within_distance_mask
                    count = match_mask.sum()
                    # Early exiting for improved eval performance
                    if count == 0:
                        continue

                    avg_epe = np.sum(epe_errors[match_mask]) / count
                    split_avg_speed = np.mean(gt_speeds[match_mask])
                    class_name = class_id_to_name[class_id]
                    yield BaseSplitKey(
                        class_name, distance_threshold, speed_threshold_tuple
                    ), BaseSplitValue(avg_epe, count, split_avg_speed)


class PerFrameSceneFlowEvaluator(Evaluator):
    def __init__(
        self,
        class_id_to_name: dict[SemanticClassId, str],
        output_path: Path = Path("/tmp/frame_results"),
    ):
        output_path = Path(output_path)
        self.eval_frame_results: list[BaseEvalFrameResult] = []
        self.output_path = output_path
        # print(f"Saving results to {self.output_path}")
        # make the directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.class_id_to_name = class_id_to_name

    @staticmethod
    def from_evaluator_list(evaluator_list: list["PerFrameSceneFlowEvaluator"]):
        assert len(evaluator_list) > 0, "evaluator_list must have at least one evaluator"

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

    def _sanitize_and_validate_inputs(
        self,
        predicted_flow: EgoLidarFlow,
        gt_frame: TimeSyncedSceneFlowFrame,
    ):
        assert isinstance(
            predicted_flow, EgoLidarFlow
        ), f"predictions must be a EstimatedFlows, got {type(predicted_flow)}"

        assert isinstance(
            gt_frame, TimeSyncedSceneFlowFrame
        ), f"ground_truth must be a GroundTruthFlows, got {type(gt_frame)}"

        # Ensure that the predictions underlying array is the same shape as the gt
        assert len(predicted_flow.full_flow) == len(
            gt_frame.flow.full_flow
        ), f"predictions and ground_truth must have the same length, got {len(predicted_flow.full_flow)} and {len(gt_frame.flow.full_flow)}"

        # Validate that all valid gt flow vectors are considered valid in the predictions.
        if not np.all((predicted_flow.mask & gt_frame.flow.mask) == gt_frame.flow.mask):
            print(
                f"{gt_frame.log_id} index {gt_frame.log_idx} with timestamp {gt_frame.log_timestamp} missing {np.sum(gt_frame.flow.mask & ~predicted_flow.mask)} points marked valid."
            )

        # Set the prediction valid flow mask to be the gt flow so everything lines up
        predicted_flow.mask = gt_frame.flow.mask

    def eval(self, predicted_flow: EgoLidarFlow, gt_frame: TimeSyncedSceneFlowFrame):
        self._sanitize_and_validate_inputs(predicted_flow, gt_frame)

        is_valid_flow_mask = gt_frame.flow.mask

        global_pc = gt_frame.pc.full_global_pc.mask_points(is_valid_flow_mask)
        class_ids = gt_frame.pc.full_pc_classes[is_valid_flow_mask]
        gt_flow = gt_frame.flow.valid_flow
        pred_flow = predicted_flow.valid_flow

        assert (
            gt_flow.shape == pred_flow.shape
        ), f"gt_flow and pred_flow must have the same shape, got {gt_flow.shape} and {pred_flow.shape}"

        eval_frame_result = self._build_eval_frame_results(global_pc, class_ids, gt_flow, pred_flow)
        self.eval_frame_results.append(eval_frame_result)

    def _build_eval_frame_results(
        self, pc1: PointCloud, gt_class_ids: np.ndarray, gt_flow: np.ndarray, pred_flow: np.ndarray
    ) -> BaseEvalFrameResult:
        """
        Override this method to build a custom EvalFrameResult child construction
        """
        return BaseEvalFrameResult(
            pc1, gt_class_ids, gt_flow, pred_flow, class_id_to_name=self.class_id_to_name
        )

    def _save_intermediary_results(self):
        save_path = self.output_path / "eval_frame_results.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(save_path, self.eval_frame_results)

    def _category_to_per_frame_stats(self) -> dict[BaseSplitKey, list[BaseSplitValue]]:
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
        self, merged_class_error_dict: dict[BaseSplitKey, list[BaseSplitValue]]
    ) -> dict[BaseSplitKey, BaseSplitValue]:
        # Compute the average EPE for each key
        result_dict = dict()
        for k in sorted(merged_class_error_dict.keys()):
            values = merged_class_error_dict[k]
            epes = np.array([v.avg_epe for v in values])
            counts = np.array([v.count for v in values])
            # Average of the epes weighted by the counts

            weighted_split_avg_speed = np.nan
            weighted_average_epe = np.nan
            if counts.sum() > 0:
                valid_counts_mask = counts > 0

                weighted_average_epe = np.average(
                    epes[valid_counts_mask], weights=(counts[valid_counts_mask])
                )
                weighted_split_avg_speed = np.average(
                    np.array([v.avg_speed for v in values])[valid_counts_mask],
                    weights=(counts[valid_counts_mask]),
                )

            result_dict[k] = BaseSplitValue(
                weighted_average_epe, counts.sum(), weighted_split_avg_speed
            )
        return result_dict

    def _save_dict(self, path: Path, data: dict[Any, float]):
        str_data = {str(k): v for k, v in data.items()}
        save_json(path, str_data)

    def _save_stats_tables(self, average_stats: dict[BaseSplitKey, BaseSplitValue]):
        assert (
            len(average_stats) > 0
        ), f"average_stats must have at least one entry, got {len(average_stats)}"

        unique_speed_threshold_tuples = sorted(
            set([k.speed_thresholds for k in average_stats.keys()])
        )
        unique_distance_thresholds = sorted(
            set([k.distance_threshold for k in average_stats.keys()])
        )
        unique_category_names = sorted(set([k.name for k in average_stats.keys()]))

        for distance_threshold in unique_distance_thresholds:
            raw_table_save_path = self.output_path / f"metric_table_{distance_threshold}.json"
            speed_table_save_path = self.output_path / f"speed_table_{distance_threshold}.json"

            # Rows are category names, columns are for speed buckets

            epe_dict = dict()
            speed_dict = dict()

            for category_name in unique_category_names:
                for speed_threshold_tuple in unique_speed_threshold_tuples:
                    key = BaseSplitKey(category_name, distance_threshold, speed_threshold_tuple)
                    avg_epe = np.nan
                    avg_speed = np.nan
                    if key in average_stats:
                        avg_epe = average_stats[key].avg_epe
                        avg_speed = average_stats[key].avg_speed
                    epe_dict[key] = avg_epe
                    speed_dict[key] = avg_speed

            Path(raw_table_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(speed_table_save_path).parent.mkdir(parents=True, exist_ok=True)
            self._save_dict(raw_table_save_path, epe_dict)
            self._save_dict(speed_table_save_path, speed_dict)

    def compute_results(
        self, save_results: bool = True
    ) -> Union[dict[BaseSplitKey, BaseSplitValue], dict[str, Any]]:
        assert (
            len(self.eval_frame_results) > 0
        ), "Must call eval at least once before calling compute"
        if save_results:
            self._save_intermediary_results()

        # From list of dicts to dict of lists
        category_to_per_frame_stats = self._category_to_per_frame_stats()
        category_to_average_stats = self._category_to_average_stats(category_to_per_frame_stats)
        self._save_stats_tables(category_to_average_stats)
        return category_to_average_stats
