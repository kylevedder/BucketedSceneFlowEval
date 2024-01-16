from pathlib import Path

import numpy as np

from .base_per_frame_sceneflow_eval import (
    BaseEvalFrameResult,
    PerFrameSceneFlowEvaluator,
)


class ThreewayEPEEvalFrameResult(BaseEvalFrameResult):
    # Pass through all arguments to super class
    def __init__(self, *args, **kwargs):
        # Add additional 0.05 threshold (0.5m/s)
        super().__init__(*args, **kwargs, max_speed_thresholds=[(0, 0.05), (0.05, np.inf)])


class PerClassThreewayEPEEvaluator(PerFrameSceneFlowEvaluator):
    def __init__(self, output_path: Path = Path("/tmp/frame_results/threeway_epe")):
        super().__init__(output_path=output_path)
        print(">>>>>>>>>>>PerClassThreewayEPEEvaluator")

    def _build_eval_frame_results(
        self,
        pc1: np.ndarray,
        gt_class_ids: np.ndarray,
        gt_flow: np.ndarray,
        pred_flow: np.ndarray,
        ground_truth,
    ) -> BaseEvalFrameResult:
        return ThreewayEPEEvalFrameResult(
            pc1,
            gt_class_ids,
            gt_flow,
            pred_flow,
            class_id_to_name=ground_truth.pretty_name,
        )
