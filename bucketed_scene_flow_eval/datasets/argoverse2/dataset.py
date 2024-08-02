import copy
from pathlib import Path
from typing import Optional, Union

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

from .argoverse_box_annotations import ArgoverseBoxAnnotationSequenceLoader
from .argoverse_raw_data import DEFAULT_POINT_CLOUD_RANGE, PointCloudRange
from .argoverse_scene_flow import (
    CATEGORY_MAP,
    ArgoverseNoFlowSequenceLoader,
    ArgoverseSceneFlowSequenceLoader,
)
from .av2_metacategories import (
    BUCKETED_METACATAGORIES,
    BUCKETED_VOLUME_METACATAGORIES,
    THREEWAY_EPE_METACATAGORIES,
)


def _make_av2_evaluator(eval_type: EvalType, eval_args: dict) -> Evaluator:
    eval_args_copy = copy.deepcopy(eval_args)
    # Builds the evaluator object for this dataset.
    if eval_type == EvalType.BUCKETED_EPE:
        if "meta_class_lookup" not in eval_args_copy:
            eval_args_copy["meta_class_lookup"] = BUCKETED_METACATAGORIES
        if "class_id_to_name" not in eval_args_copy:
            eval_args_copy["class_id_to_name"] = CATEGORY_MAP
        return BucketedEPEEvaluator(**eval_args_copy)
    elif eval_type == EvalType.THREEWAY_EPE:
        if "meta_class_lookup" not in eval_args_copy:
            eval_args_copy["meta_class_lookup"] = THREEWAY_EPE_METACATAGORIES
        if "class_id_to_name" not in eval_args_copy:
            eval_args_copy["class_id_to_name"] = CATEGORY_MAP
        return ThreeWayEPEEvaluator(**eval_args_copy)
    elif eval_type == EvalType.BUCKETED_VOLUME_EPE:
        if "meta_class_lookup" not in eval_args_copy:
            eval_args_copy["meta_class_lookup"] = BUCKETED_VOLUME_METACATAGORIES
        if "class_id_to_name" not in eval_args_copy:
            eval_args_copy["class_id_to_name"] = {
                -1: "BACKGROUND",
                0: "SMALL",
                1: "MEDIUM",
                2: "LARGE",
            }
        return BucketedEPEEvaluator(**eval_args_copy)
    else:
        raise ValueError(f"Unknown eval type {eval_type}")


class Argoverse2CausalSceneFlow(CausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        subsequence_length: int = 2,
        sliding_window_step_size: int | None = 1,
        with_ground: bool = True,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        load_boxes: bool = False,
        load_flow: bool = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        if load_boxes:
            self.sequence_loader = ArgoverseBoxAnnotationSequenceLoader(
                root_dir,
                **kwargs,
            )
        elif load_flow:
            self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
                root_dir,
                use_gt_flow=use_gt_flow,
                flow_data_path=flow_data_path,
                **kwargs,
            )
        else:
            self.sequence_loader = ArgoverseNoFlowSequenceLoader(
                root_dir,
                **kwargs,
            )
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=with_ground,
            idx_lookup_cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
            sliding_window_step_size=sliding_window_step_size,
        )

    def evaluator(self) -> Evaluator:
        return _make_av2_evaluator(self.eval_type, self.eval_args)


class Argoverse2NonCausalSceneFlow(NonCausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        subsequence_length: int = 2,
        sliding_window_step_size: int | None = None,
        with_ground: bool = True,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        use_cache=True,
        load_boxes: bool = False,
        load_flow: bool = True,
        **kwargs,
    ) -> None:
        if load_boxes:
            self.sequence_loader = ArgoverseBoxAnnotationSequenceLoader(
                raw_data_path=root_dir,
                **kwargs,
            )
        elif load_flow:
            self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
                raw_data_path=root_dir,
                use_gt_flow=use_gt_flow,
                flow_data_path=flow_data_path,
                **kwargs,
            )
        else:
            self.sequence_loader = ArgoverseNoFlowSequenceLoader(
                raw_data_path=root_dir,
                **kwargs,
            )
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=with_ground,
            idx_lookup_cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
            sliding_window_step_size=sliding_window_step_size,
        )

    def evaluator(self) -> Evaluator:
        return _make_av2_evaluator(self.eval_type, self.eval_args)
