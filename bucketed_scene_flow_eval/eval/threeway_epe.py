from pathlib import Path

import numpy as np

from .bucketed_epe import BucketedEPEEvaluator


class ThreeWayEPEEvaluator(BucketedEPEEvaluator):
    def __init__(
        self,
        meta_class_lookup: dict[str, list[str]],
        dynamic_threshold_meters_per_frame=0.5 / 10,
        output_path: Path = Path("/tmp/frame_results/threeway_epe"),
    ) -> None:
        assert meta_class_lookup is not None, "meta_class_lookup must be provided"
        assert isinstance(meta_class_lookup, dict), "meta_class_lookup must be a dictionary"
        assert (
            len(meta_class_lookup.keys()) == 2
        ), f"Threeway EPE meta_class_lookup must have 2 keys, instead found {len(meta_class_lookup.keys())} keys: {meta_class_lookup.keys()}"
        super().__init__(output_path=output_path, meta_class_lookup=meta_class_lookup)
        bucket_edges = [0.0, dynamic_threshold_meters_per_frame, np.inf]
        self.speed_thresholds = list(zip(bucket_edges, bucket_edges[1:]))

    def _save_stats_tables(self, average_stats):
        super()._save_stats_tables(average_stats, normalized=False)

    def compute_results(
        self, save_results: bool = True, return_distance_threshold: int = 35
    ) -> dict[str, tuple[float, float]]:
        super().compute_results(save_results)

        category_to_per_frame_stats = self._category_to_per_frame_stats()
        category_to_average_stats = self._category_to_average_stats(category_to_per_frame_stats)
        matrix = self._build_stat_table(category_to_average_stats, return_distance_threshold)
        assert len(matrix.class_names) == 2, f"Expected 2 classes, found {len(matrix.class_names)}"
        return {
            str(k): (v.static_epe, v.dynamic_error)
            for k, v in matrix.get_overall_class_errors(normalized=False).items()
        }
