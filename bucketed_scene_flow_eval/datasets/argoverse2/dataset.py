import copy
import enum
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.interfaces import AbstractDataset
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .argoverse_scene_flow import (
    CATEGORY_MAP,
    ArgoverseNoFlowSequence,
    ArgoverseNoFlowSequenceLoader,
    ArgoverseSceneFlowSequenceLoader,
)
from .av2_metacategories import BUCKETED_METACATAGORIES, THREEWAY_EPE_METACATAGORIES


class EvalType(enum.Enum):
    BUCKETED_EPE = 0
    THREEWAY_EPE = 1


class Argoverse2SceneFlow(AbstractDataset):
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
        self.use_cache = use_cache
        self.with_ground = with_ground
        self.use_gt_flow = use_gt_flow
        if load_flow:
            self.sequence_loader = ArgoverseSceneFlowSequenceLoader(
                root_dir, with_rgb=with_rgb, use_gt_flow=use_gt_flow, flow_data_path=flow_data_path
            )
        else:
            self.sequence_loader = ArgoverseNoFlowSequenceLoader(root_dir, with_rgb=with_rgb)
        self.subsequence_length = subsequence_length
        self.cache_path = self._cache_path(cache_root, root_dir)

        self.with_rgb = with_rgb
        self.flow_data_path = flow_data_path

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx()
        self.sequence_subsequence_idx_to_dataset_idx = {
            value: key for key, value in enumerate(self.dataset_to_sequence_subsequence_idx)
        }

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _cache_path(self, cache_root: Path, root_dir: Union[Path, list[Path]]) -> Path:
        if isinstance(root_dir, list):
            parent_name = "_".join([Path(root_dir_part).parent.name for root_dir_part in root_dir])
            folder_name = "_".join([Path(root_dir_part).name for root_dir_part in root_dir])
        else:
            parent_name = Path(root_dir).parent.name
            folder_name = Path(root_dir).name
        return cache_root / "argo" / parent_name / folder_name

    def _get_cache_file(self) -> Path:
        flow_data_path_name = "None"
        if not (self.flow_data_path is None):
            if isinstance(self.flow_data_path, list):
                flow_data_path_name = "_".join(
                    [Path(flow_data_path_part).stem for flow_data_path_part in self.flow_data_path]
                )
            else:
                flow_data_path_name = self.flow_data_path.stem
        flow_data_path_name = flow_data_path_name.strip()
        cache_file = (
            self.cache_path
            / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}_use_gt_{self.use_gt_flow}_with_rgb_{self.with_rgb}_with_ground_{self.with_ground}_flow_data_path_name_{flow_data_path_name}.pkl"
        )
        return cache_file

    def _load_dataset_to_sequence_subsequence_idx(self) -> list[tuple[int, int]]:
        cache_file = self._get_cache_file()
        if cache_file.exists() and self.use_cache:
            cache_pkl = load_pickle(cache_file)
            # Sanity check that the cache is the right length by ensuring that it
            # has the same length as the sequence loader.
            if len(cache_pkl) == len(self.sequence_loader):
                return cache_pkl

        print("Building dataset index...")
        # Build map from dataset index to sequence and subsequence index.
        dataset_to_sequence_subsequence_idx = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(len(sequence) - self.subsequence_length + 1):
                dataset_to_sequence_subsequence_idx.append((sequence_idx, subsequence_start_idx))

        print(
            f"Loaded {len(dataset_to_sequence_subsequence_idx)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_to_sequence_subsequence_idx)
        return dataset_to_sequence_subsequence_idx

    def __len__(self):
        return len(self.dataset_to_sequence_subsequence_idx)

    def _av2_sequence_id_and_timestamp_to_idx(self, av2_sequence_id: str, timestamp: int) -> int:
        sequence_loader_idx = self.sequence_loader._sequence_id_to_idx(av2_sequence_id)
        sequence = self.sequence_loader.load_sequence(av2_sequence_id)
        sequence_idx = sequence._timestamp_to_idx(timestamp)
        return self.sequence_subsequence_idx_to_dataset_idx[(sequence_loader_idx, sequence_idx)]

    def _process_with_metadata(
        self, item: TimeSyncedSceneFlowItem, metadata: TimeSyncedAVLidarData
    ) -> TimeSyncedSceneFlowItem:
        # Falsify PC mask for ground points.
        item.pc.mask = item.pc.mask & metadata.in_range_mask
        # Falsify Flow mask for ground points.
        item.flow.mask = item.flow.mask & metadata.in_range_mask

        if not self.with_ground:
            item.pc = item.pc.mask_points(~metadata.is_ground_points)
            item.flow = item.flow.mask_points(~metadata.is_ground_points)

        return item

    def __getitem__(self, dataset_idx, verbose: bool = False) -> list[TimeSyncedSceneFlowItem]:
        if verbose:
            print(f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) start")

        sequence_idx, subsequence_start_idx = self.dataset_to_sequence_subsequence_idx[dataset_idx]

        # Load sequence
        sequence = self.sequence_loader[sequence_idx]

        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        # Load subsequence

        subsequence_frames = [
            sequence.load(
                subsequence_start_idx + i,
                subsequence_start_idx + in_subsequence_tgt_index,
                with_flow=(i != self.subsequence_length - 1),
            )
            for i in range(self.subsequence_length)
        ]

        scene_flow_items = [item for item, _ in subsequence_frames]
        scene_flow_metadata = [metadata for _, metadata in subsequence_frames]

        scene_flow_items = [
            self._process_with_metadata(item, metadata)
            for item, metadata in zip(scene_flow_items, scene_flow_metadata)
        ]

        return scene_flow_items

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
