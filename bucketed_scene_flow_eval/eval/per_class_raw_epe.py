from pathlib import Path

from .base_per_frame_sceneflow_eval import PerFrameSceneFlowEvaluator


class PerClassRawEPEEvaluator(PerFrameSceneFlowEvaluator):
    def __init__(self, output_path: Path = Path("/tmp/frame_results/raw_per_class_epe")):
        super().__init__(output_path=output_path)
