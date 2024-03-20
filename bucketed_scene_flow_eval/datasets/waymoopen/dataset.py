import copy
from pathlib import Path

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.interfaces import BaseAstractSeqLoaderDataset, EvalType

from .waymo_supervised_flow import CATEGORY_MAP, WaymoSupervisedSceneFlowSequenceLoader


class WaymoOpenSceneFlow(BaseAstractSeqLoaderDataset):
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        root_dir: Path,
        subsequence_length: int = 2,
        cache_root: Path = Path("/tmp/"),
        eval_type: str = "bucketed_epe",
        with_rgb: bool = False,
        use_cache: bool = True,
        eval_args=dict(),
    ) -> None:
        self.sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(root_dir)
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=True,
            idx_lookup_cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
        )

    def evaluator(self) -> Evaluator:
        eval_args_copy = copy.deepcopy(self.eval_args)
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.BUCKETED_EPE:
            if "class_id_to_name" not in eval_args_copy:
                eval_args_copy["class_id_to_name"] = CATEGORY_MAP
            return BucketedEPEEvaluator(**eval_args_copy)
        elif self.eval_type == EvalType.THREEWAY_EPE:
            if "class_id_to_name" not in eval_args_copy:
                eval_args_copy["class_id_to_name"] = CATEGORY_MAP
            return ThreeWayEPEEvaluator(**eval_args_copy)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
