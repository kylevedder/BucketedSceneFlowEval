import copy
from pathlib import Path

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.interfaces import (
    CausalSeqLoaderDataset,
    EvalType,
    NonCausalSeqLoaderDataset,
)

from .waymo_supervised_flow import CATEGORY_MAP, WaymoSupervisedSceneFlowSequenceLoader

THREEWAY_EPE_METACATAGORIES = {
    "FOREGROUND": ["VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"],
    "BACKGROUND": ["BACKGROUND"],
}


def _make_waymo_evaluator(eval_type: EvalType, eval_args: dict) -> Evaluator:
    eval_args_copy = copy.deepcopy(eval_args)
    # Builds the evaluator object for this dataset.
    if eval_type == EvalType.BUCKETED_EPE:
        if "class_id_to_name" not in eval_args_copy:
            eval_args_copy["class_id_to_name"] = CATEGORY_MAP
        return BucketedEPEEvaluator(**eval_args_copy)
    elif eval_type == EvalType.THREEWAY_EPE:
        if "meta_class_lookup" not in eval_args_copy:
            eval_args_copy["meta_class_lookup"] = THREEWAY_EPE_METACATAGORIES
        if "class_id_to_name" not in eval_args_copy:
            eval_args_copy["class_id_to_name"] = CATEGORY_MAP
        return ThreeWayEPEEvaluator(**eval_args_copy)
    else:
        raise ValueError(f"Unknown eval type {eval_type}")


class WaymoOpenCausalSceneFlow(CausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Path,
        flow_folder: Path | None = None,
        subsequence_length: int = 2,
        cache_root: Path = Path("/tmp/"),
        eval_type: str = "bucketed_epe",
        with_rgb: bool = True,
        use_cache: bool = True,
        log_subset: list[str] | None = None,
        eval_args=dict(),
    ) -> None:
        self.sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(
            root_dir, log_subset=log_subset, with_rgb=with_rgb, flow_dir=flow_folder
        )
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
        return _make_waymo_evaluator(self.eval_type, self.eval_args)


class WaymoOpenNonCausalSceneFlow(NonCausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Path,
        subsequence_length: int = 2,
        cache_root: Path = Path("/tmp/"),
        eval_type: str = "bucketed_epe",
        with_rgb: bool = True,
        use_cache: bool = True,
        log_subset: list[str] | None = None,
        eval_args=dict(),
    ) -> None:
        self.sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(
            root_dir, log_subset=log_subset
        )
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
        return _make_waymo_evaluator(self.eval_type, self.eval_args)
