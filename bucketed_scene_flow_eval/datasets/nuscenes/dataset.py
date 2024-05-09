import copy
from pathlib import Path
from typing import Optional, Union

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_raw_data import (
    DEFAULT_POINT_CLOUD_RANGE,
    PointCloudRange,
)
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

from .nuscenes_metacategories import (
    BUCKETED_METACATAGORIES,
    THREEWAY_EPE_METACATAGORIES,
)
from .nuscenes_scene_flow import (
    CATEGORY_MAP,
    NuScenesNoFlowSequenceLoader,
    NuScenesSceneFlowSequenceLoader,
)


def _make_evaluator(eval_type: EvalType, eval_args: dict) -> Evaluator:
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
    else:
        raise ValueError(f"Unknown eval type {eval_type}")


class NuScenesCausalSceneFlow(CausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        nuscenes_version: str,
        split: str,
        subsequence_length: int = 2,
        with_ground: bool = True,
        with_rgb: bool = False,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
        point_cloud_range: Optional[PointCloudRange] = DEFAULT_POINT_CLOUD_RANGE,
        use_cache=True,
        load_flow: bool = True,
    ) -> None:
        if load_flow:
            self.sequence_loader = NuScenesSceneFlowSequenceLoader(
                raw_data_path=root_dir,
                nuscenes_version=nuscenes_version,
                split=split,
                with_rgb=with_rgb,
                use_gt_flow=use_gt_flow,
                flow_data_path=flow_data_path,
                expected_camera_shape=expected_camera_shape,
                point_cloud_range=point_cloud_range,
            )
        else:
            self.sequence_loader = NuScenesNoFlowSequenceLoader(
                raw_data_path=root_dir,
                nuscenes_version=nuscenes_version,
                split=split,
                with_rgb=with_rgb,
                expected_camera_shape=expected_camera_shape,
                point_cloud_range=point_cloud_range,
            )
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=with_ground,
            idx_lookup_cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
        )

    def evaluator(self) -> Evaluator:
        return _make_evaluator(self.eval_type, self.eval_args)


class NuScenesNonCausalSceneFlow(NonCausalSeqLoaderDataset):
    def __init__(
        self,
        root_dir: Union[Path, list[Path]],
        nuscenes_version: str,
        split: str,
        subsequence_length: int = 2,
        with_ground: bool = True,
        with_rgb: bool = False,
        cache_root: Path = Path("/tmp/"),
        use_gt_flow: bool = True,
        flow_data_path: Optional[Union[Path, list[Path]]] = None,
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        expected_camera_shape: tuple[int, int, int] = (1550, 2048, 3),
        use_cache=True,
        load_flow: bool = True,
    ) -> None:
        if load_flow:
            self.sequence_loader = NuScenesSceneFlowSequenceLoader(
                root_dir,
                nuscenes_version=nuscenes_version,
                split=split,
                with_rgb=with_rgb,
                use_gt_flow=use_gt_flow,
                flow_data_path=flow_data_path,
                expected_camera_shape=expected_camera_shape,
            )
        else:
            self.sequence_loader = NuScenesNoFlowSequenceLoader(
                root_dir, nuscenes_version=nuscenes_version, split=split, with_rgb=with_rgb, expected_camera_shape=expected_camera_shape
            )
        super().__init__(
            sequence_loader=self.sequence_loader,
            subsequence_length=subsequence_length,
            with_ground=with_ground,
            idx_lookup_cache_root=cache_root,
            eval_type=eval_type,
            eval_args=eval_args,
            use_cache=use_cache,
        )

    def evaluator(self) -> Evaluator:
        return _make_evaluator(self.eval_type, self.eval_args)
