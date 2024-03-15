import copy
from pathlib import Path
from typing import Any, Optional, Union

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.interfaces import (
    AbstractAVLidarSequence,
    BaseDatasetForAbstractSeqLoader,
    EvalType,
)

from .argoverse_scene_flow import (
    CATEGORY_MAP,
    ArgoverseNoFlowSequenceLoader,
    ArgoverseSceneFlowSequenceLoader,
)
from .av2_metacategories import BUCKETED_METACATAGORIES, THREEWAY_EPE_METACATAGORIES


class Argoverse2SceneFlow(BaseDatasetForAbstractSeqLoader):
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        subsequence_length: int = 2,
        with_ground: bool = True,
        with_rgb: bool = False,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        use_cache=True,
        load_flow: bool = True,
    ) -> None:
        if load_flow:
            self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
                root_dir, with_rgb=with_rgb, use_gt_flow=use_gt_flow, flow_data_path=flow_data_path
            )
        else:
            self.sequence_loader = ArgoverseNoFlowSequenceLoader(root_dir, with_rgb=with_rgb)
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=with_ground,
            cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
        )

    def _load_from_sequence(
        self,
        sequence: AbstractAVLidarSequence,
        idx: int,
        subsequence_start_idx: int,
        other_load_args: dict[str, Any] = {},
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        other_load_args["with_flow"] = idx != self.subsequence_length - 1
        return super()._load_from_sequence(
            sequence,
            idx,
            subsequence_start_idx,
            other_load_args=other_load_args,
        )

    def evaluator(self) -> Evaluator:
        eval_args_copy = copy.deepcopy(self.eval_args)
        # Builds the evaluator object for this dataset.
        if self.eval_type == EvalType.BUCKETED_EPE:
            if "meta_class_lookup" not in eval_args_copy:
                eval_args_copy["meta_class_lookup"] = BUCKETED_METACATAGORIES
            if "class_id_to_name" not in eval_args_copy:
                eval_args_copy["class_id_to_name"] = CATEGORY_MAP
            return BucketedEPEEvaluator(**eval_args_copy)
        elif self.eval_type == EvalType.THREEWAY_EPE:
            if "meta_class_lookup" not in eval_args_copy:
                eval_args_copy["meta_class_lookup"] = THREEWAY_EPE_METACATAGORIES
            if "class_id_to_name" not in eval_args_copy:
                eval_args_copy["class_id_to_name"] = CATEGORY_MAP
            return ThreeWayEPEEvaluator(**eval_args_copy)
        else:
            raise ValueError(f"Unknown eval type {self.eval_type}")
