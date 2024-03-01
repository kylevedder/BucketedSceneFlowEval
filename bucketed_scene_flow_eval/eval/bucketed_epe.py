import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bucketed_scene_flow_eval.utils import save_json, save_txt

from .base_per_frame_sceneflow_eval import (
    BaseEvalFrameResult,
    BaseSplitKey,
    BaseSplitValue,
    PerFrameSceneFlowEvaluator,
)


class BucketedEvalFrameResult(BaseEvalFrameResult):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Pass through all arguments to super class
        super().__init__(*args, **kwargs)

    def _scale_flows(
        self, gt_flow: np.ndarray, pred_flow: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return gt_flow, pred_flow


@dataclass(frozen=True, eq=True, repr=True)
class OverallError:
    static_epe: float
    dynamic_error: float

    def __repr__(self) -> str:
        static_epe_val_str = (
            f"{self.static_epe:0.6f}" if np.isfinite(self.static_epe) else f"{self.static_epe}"
        )
        dynamic_error_val_str = (
            f"{self.dynamic_error:0.6f}"
            if np.isfinite(self.dynamic_error)
            else f"{self.dynamic_error}"
        )
        return f"({static_epe_val_str}, {dynamic_error_val_str})"

    def to_tuple(self) -> tuple[float, float]:
        return (self.static_epe, self.dynamic_error)


class BucketResultMatrix:
    def __init__(self, class_names: list[str], speed_buckets: list[tuple[float, float]]):
        self.class_names = class_names
        self.speed_buckets = speed_buckets

        assert (
            len(self.class_names) > 0
        ), f"class_names must have at least one entry, got {len(self.class_names)}"
        assert (
            len(self.speed_buckets) > 0
        ), f"speed_buckets must have at least one entry, got {len(self.speed_buckets)}"

        # By default, NaNs are not counted in np.nanmean
        self.epe_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.NaN
        self.speed_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.NaN
        self.count_storage_matrix = np.zeros(
            (len(class_names), len(self.speed_buckets)), dtype=np.int64
        )

    def has_class(self, class_name: str) -> bool:
        return class_name in self.class_names

    def accumulate_value(
        self,
        class_name: str,
        speed_bucket: tuple[float, float],
        average_epe: float,
        average_speed: float,
        count: int,
    ):
        assert count > 0, f"count must be greater than 0, got {count}"
        assert np.isfinite(average_epe), f"average_epe must be finite, got {average_epe}"
        assert np.isfinite(average_speed), f"average_speed must be finite, got {average_speed}"

        class_idx = self.class_names.index(class_name)
        speed_bucket_idx = self.speed_buckets.index(speed_bucket)

        prior_epe = self.epe_storage_matrix[class_idx, speed_bucket_idx]
        prior_speed = self.speed_storage_matrix[class_idx, speed_bucket_idx]
        prior_count = self.count_storage_matrix[class_idx, speed_bucket_idx]

        if np.isnan(prior_epe):
            self.epe_storage_matrix[class_idx, speed_bucket_idx] = average_epe
            self.speed_storage_matrix[class_idx, speed_bucket_idx] = average_speed
            self.count_storage_matrix[class_idx, speed_bucket_idx] = count
            return

        # Accumulate the average EPE and speed, weighted by the number of samples using np.mean
        self.epe_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_epe, average_epe], weights=[prior_count, count]
        )
        self.speed_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_speed, average_speed], weights=[prior_count, count]
        )
        self.count_storage_matrix[class_idx, speed_bucket_idx] += count

    def get_normalized_error_matrix(self) -> np.ndarray:
        error_matrix = self.epe_storage_matrix.copy()
        # For the 1: columns, normalize EPE entries by the speed
        error_matrix[:, 1:] = error_matrix[:, 1:] / self.speed_storage_matrix[:, 1:]
        return error_matrix

    def get_overall_class_errors(self, normalized: bool = True) -> dict[str, OverallError]:
        if normalized:
            error_matrix = self.get_normalized_error_matrix()
        else:
            error_matrix = self.epe_storage_matrix.copy()
        static_epes = error_matrix[:, 0]
        # Hide the warning about mean of empty slice
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dynamic_errors = np.nanmean(error_matrix[:, 1:], axis=1)

        return {
            class_name: OverallError(static_epe, dynamic_error)
            for class_name, static_epe, dynamic_error in zip(
                self.class_names, static_epes, dynamic_errors
            )
        }

    def get_class_entries(self, class_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_idx = self.class_names.index(class_name)

        epe = self.epe_storage_matrix[class_idx, :]
        speed = self.speed_storage_matrix[class_idx, :]
        count = self.count_storage_matrix[class_idx, :]
        return epe, speed, count

    def merge_matrix_classes(self, meta_class_lookup: dict[str, list[str]]) -> "BucketResultMatrix":
        assert meta_class_lookup is not None, f"meta_class_lookup must be set to merge classes"
        assert (
            len(meta_class_lookup) > 0
        ), f"meta_class_lookup must have at least one entry, got {len(meta_class_lookup)}"

        # Create a new matrix with the merged classes
        merged_matrix = BucketResultMatrix(
            class_names=sorted(meta_class_lookup.keys()),
            speed_buckets=self.speed_buckets,
        )

        for meta_class, child_classes in sorted(meta_class_lookup.items()):
            for child_class in child_classes:
                if not self.has_class(child_class):
                    continue
                epes, speeds, counts = self.get_class_entries(child_class)
                for speed_bucket, epe, speed, count in zip(
                    self.speed_buckets, epes, speeds, counts
                ):
                    if np.isnan(epe):
                        continue
                    merged_matrix.accumulate_value(meta_class, speed_bucket, epe, speed, count)

        return merged_matrix

    def get_mean_average_values(self, normalized: bool = True) -> OverallError:
        overall_errors = self.get_overall_class_errors(normalized=normalized)

        average_static_epe = np.nanmean([v.static_epe for v in overall_errors.values()])
        average_dynamic_error = np.nanmean([v.dynamic_error for v in overall_errors.values()])

        return OverallError(average_static_epe, average_dynamic_error)

    def to_full_latex(self, normalized: bool = True) -> str:
        if normalized:
            error_matrix = self.get_normalized_error_matrix()
        else:
            error_matrix = self.epe_storage_matrix.copy()
        # First, get the average class values
        average_class_values = self.get_overall_class_errors(normalized=normalized)

        # Define the header row with the speed buckets and the beginning of the tabular environment
        column_format = (
            "l" + "c" + "c" * len(self.speed_buckets)
        )  # 'l' for the first column (Class Name), 'c' for the average and bucket columns
        header_row = "Class Name & \\textbf{Overall Error}"  # Add 'Average' column header in bold
        for low, high in self.speed_buckets:
            # Multiply by 10 and format with two decimal places, then rotate the text vertically
            header_row += " & \\rotatebox{90}{" + f"[{low * 10:.2f}-{high * 10:.2f}]" + "}"
        header_row += " \\\\\n"  # End the header row with newline and line break for LaTeX

        latex_string = "\\begin{tabular}{" + column_format + "}\n\\hline\n"
        latex_string += header_row  # Add the header row
        latex_string += "\\hline\n"

        # Add the data rows for each class
        for class_name in self.class_names:
            # Escape underscores in class names
            class_name_escaped = class_name.replace("_", "\\_")
            # Get the average value for the class, make it bold, and replace NaN with a hyphen
            avg_val = average_class_values[class_name]
            average_value = f"\\textbf{{{avg_val}}}"
            # Format the data values with two decimal places or a hyphen if NaN
            row_data = " & ".join(
                [
                    f"{value:.6f}" if not np.isnan(value) else "-"
                    for value in error_matrix[self.class_names.index(class_name)]
                ]
            )
            latex_string += f"{class_name_escaped} & {average_value} & {row_data} \\\\\n"  # End the row with newline and line break for LaTeX

        # Finish the tabular environment
        latex_string += "\\hline\n\\end{tabular}"

        return latex_string


class BucketedEPEEvaluator(PerFrameSceneFlowEvaluator):
    def __init__(
        self,
        bucket_max_speed: float = 20.0 / 10.0,
        num_buckets: int = 51,
        output_path: Path = Path("/tmp/frame_results/bucketed_epe"),
        meta_class_lookup: Optional[dict[str, list[str]]] = None,
    ):
        # Bucket the speeds into num_buckets buckets. Add one extra bucket at the end to capture all
        # the speeds above bucket_max_speed_meters_per_second.
        bucket_edges = np.concatenate([np.linspace(0, bucket_max_speed, num_buckets), [np.inf]])
        self.speed_thresholds = list(zip(bucket_edges, bucket_edges[1:]))
        self.meta_class_lookup = meta_class_lookup
        super().__init__(output_path=output_path)

    def _build_eval_frame_results(
        self,
        pc1: np.ndarray,
        gt_class_ids: np.ndarray,
        gt_flow: np.ndarray,
        pred_flow: np.ndarray,
        ground_truth,
    ) -> BaseEvalFrameResult:
        return BucketedEvalFrameResult(
            pc1,
            gt_class_ids,
            gt_flow,
            pred_flow,
            class_id_to_name=ground_truth.pretty_name,
            max_speed_thresholds=self.speed_thresholds,
        )

    def _build_stat_table(
        self,
        average_stats: dict[BaseSplitKey, BaseSplitValue],
        distance_threshold: float,
    ) -> BucketResultMatrix:
        unique_category_names = sorted(set([k.name for k in average_stats.keys()]))

        matrix = BucketResultMatrix(unique_category_names, self.speed_thresholds)

        for category_name in unique_category_names:
            for speed_threshold_tuple in self.speed_thresholds:
                key = BaseSplitKey(category_name, distance_threshold, speed_threshold_tuple)
                if key not in average_stats:
                    continue
                avg_epe = average_stats[key].avg_epe
                avg_speed = average_stats[key].avg_speed
                count = average_stats[key].count
                matrix.accumulate_value(
                    category_name, speed_threshold_tuple, avg_epe, avg_speed, count
                )

        if self.meta_class_lookup is not None:
            matrix = matrix.merge_matrix_classes(self.meta_class_lookup)

        return matrix

    def _save_stats_tables(
        self, average_stats: dict[BaseSplitKey, BaseSplitValue], normalized: bool = True
    ):
        super()._save_stats_tables(average_stats)

        # Compute averages over the speed buckets

        assert (
            len(average_stats) > 0
        ), f"average_stats must have at least one entry, got {len(average_stats)}"

        unique_distance_thresholds = sorted(
            set([k.distance_threshold for k in average_stats.keys()])
        )

        for distance_threshold in unique_distance_thresholds:
            matrix = self._build_stat_table(average_stats, distance_threshold)

            full_table_save_path = self.output_path / f"full_table_{distance_threshold}.tex"
            per_class_save_path = self.output_path / f"per_class_results_{distance_threshold}.json"
            mean_average_save_path = (
                self.output_path / f"mean_average_results_{distance_threshold}.json"
            )

            # Save the raw table
            save_txt(full_table_save_path, matrix.to_full_latex(normalized=normalized))

            # Save the per-class results
            save_json(
                per_class_save_path,
                {
                    str(k): str(v)
                    for k, v in matrix.get_overall_class_errors(normalized=normalized).items()
                },
                indent=4,
            )

            # Save the mean average results
            save_json(
                mean_average_save_path,
                matrix.get_mean_average_values(normalized=normalized).to_tuple(),
                indent=4,
            )

    def compute_results(
        self, save_results: bool = True, return_distance_threshold: int = 35
    ) -> dict[str, tuple[float, float]]:
        super().compute_results(save_results)

        category_to_per_frame_stats = self._category_to_per_frame_stats()
        category_to_average_stats = self._category_to_average_stats(category_to_per_frame_stats)
        matrix = self._build_stat_table(category_to_average_stats, return_distance_threshold)
        return {
            str(k): (v.static_epe, v.dynamic_error)
            for k, v in matrix.get_overall_class_errors().items()
        }
