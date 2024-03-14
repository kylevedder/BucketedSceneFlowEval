import copy
import enum
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.eval import (
    BucketedEPEEvaluator,
    Evaluator,
    ThreeWayEPEEvaluator,
)
from bucketed_scene_flow_eval.interfaces import AbstractDataset
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .abstract_sequence_loader import (
    AbstractAVLidarSequence,
    AbstractSequence,
    AbstractSequenceLoader,
)


class EvalType(enum.Enum):
    BUCKETED_EPE = 0
    THREEWAY_EPE = 1


class BaseDatasetForAbstractSeqLoader(AbstractDataset):
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        sequence_loader: AbstractSequenceLoader,
        subsequence_length: int = 2,
        with_ground: bool = True,
        cache_root: Path = Path("/tmp/"),
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        use_cache=True,
    ) -> None:
        self.use_cache = use_cache
        self.with_ground = with_ground
        self.sequence_loader = sequence_loader
        self.subsequence_length = subsequence_length
        self.cache_path = cache_root / self.sequence_loader.config_string()

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx()
        self.sequence_subsequence_idx_to_dataset_idx = {
            value: key for key, value in enumerate(self.dataset_to_sequence_subsequence_idx)
        }

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    def _get_cache_file(self) -> Path:
        cache_file = self.cache_path / "dataset_index_cache.pkl"
        return cache_file

    def _load_dataset_to_sequence_subsequence_idx(self) -> list[tuple[int, int]]:
        cache_file = self._get_cache_file()
        if cache_file.exists() and self.use_cache:
            cache_pkl = load_pickle(cache_file)
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

    def _process_with_metadata(
        self, item: TimeSyncedSceneFlowItem, metadata: TimeSyncedAVLidarData
    ) -> TimeSyncedSceneFlowItem:
        # Typecheck
        assert isinstance(item, TimeSyncedSceneFlowItem), f"item is {type(item)}"
        assert isinstance(metadata, TimeSyncedAVLidarData), f"metadata is {type(metadata)}"
        # Falsify PC mask for ground points.
        item.pc.mask = item.pc.mask & metadata.in_range_mask
        # Falsify Flow mask for ground points.
        item.flow.mask = item.flow.mask & metadata.in_range_mask

        if not self.with_ground:
            item.pc = item.pc.mask_points(~metadata.is_ground_points)
            item.flow = item.flow.mask_points(~metadata.is_ground_points)

        return item

    def _load_from_sequence(
        self,
        sequence: AbstractAVLidarSequence,
        idx: int,
        subsequence_start_idx: int,
        other_load_args: dict[str, Any] = {},
    ) -> tuple[TimeSyncedSceneFlowItem, TimeSyncedAVLidarData]:
        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        # with_flow=(idx != self.subsequence_length - 1)
        return sequence.load(
            subsequence_start_idx + idx,
            subsequence_start_idx + in_subsequence_tgt_index,
            **other_load_args,
        )

    def __getitem__(self, dataset_idx, verbose: bool = False) -> list[TimeSyncedSceneFlowItem]:
        sequence_idx, subsequence_start_idx = self.dataset_to_sequence_subsequence_idx[dataset_idx]

        # Load sequence
        sequence = self.sequence_loader[sequence_idx]

        # Load subsequence

        subsequence_frames = [
            self._load_from_sequence(sequence, i, subsequence_start_idx)
            for i in range(self.subsequence_length)
        ]

        scene_flow_items = [item for item, _ in subsequence_frames]
        scene_flow_metadata = [metadata for _, metadata in subsequence_frames]

        scene_flow_items = [
            self._process_with_metadata(item, metadata)
            for item, metadata in zip(scene_flow_items, scene_flow_metadata)
        ]

        return scene_flow_items
