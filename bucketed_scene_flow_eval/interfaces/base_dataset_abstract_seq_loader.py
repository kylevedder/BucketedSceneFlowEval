import copy
import enum
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.interfaces import AbstractDataset, LoaderType
from bucketed_scene_flow_eval.utils import load_pickle, save_pickle

from .abstract_sequence_loader import (
    AbstractAVLidarSequence,
    AbstractSequence,
    AbstractSequenceLoader,
)


class EvalType(enum.Enum):
    BUCKETED_EPE = 0
    THREEWAY_EPE = 1


CacheLookup = list[tuple[int, tuple[int, int]]]


class BaseAbstractSeqLoaderDataset(AbstractDataset):
    def __init__(
        self,
        sequence_loader: AbstractSequenceLoader,
        subsequence_length: int = 2,
        with_ground: bool = True,
        idx_lookup_cache_root: Path = Path("/tmp/idx_lookup_cache/"),
        eval_type: str = "bucketed_epe",
        eval_args=dict(),
        use_cache=True,
    ) -> None:
        self.use_cache = use_cache
        self.with_ground = with_ground
        self.sequence_loader = sequence_loader
        self.subsequence_length = subsequence_length
        self.idx_lookup_cache_path = (
            idx_lookup_cache_root / self.sequence_loader.cache_folder_name()
        )

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx()
        self.sequence_subsequence_idx_to_dataset_idx = {
            value: key for key, value in enumerate(self.dataset_to_sequence_subsequence_idx)
        }

        self.eval_type = EvalType[eval_type.strip().upper()]
        self.eval_args = eval_args

    @abstractmethod
    def _get_idx_lookup_cache_file(self) -> Path:
        raise NotImplementedError

    def _load_existing_cache(self) -> Optional[CacheLookup]:
        cache_file = self._get_idx_lookup_cache_file()
        if cache_file.exists() and self.use_cache:
            cache_pkl = load_pickle(cache_file)
            return cache_pkl
        return None

    @abstractmethod
    def _build_new_cache(self) -> CacheLookup:
        raise NotImplementedError

    def _load_dataset_to_sequence_subsequence_idx(self) -> CacheLookup:
        existing_cache = self._load_existing_cache()
        if existing_cache is not None:
            return existing_cache

        return self._build_new_cache()

    def __len__(self):
        return len(self.dataset_to_sequence_subsequence_idx)

    def _process_frame_with_metadata(
        self, frame: TimeSyncedSceneFlowFrame, metadata: TimeSyncedAVLidarData
    ) -> TimeSyncedSceneFlowFrame:
        # Typecheck
        assert isinstance(frame, TimeSyncedSceneFlowFrame), f"item is {type(frame)}"
        assert isinstance(metadata, TimeSyncedAVLidarData), f"metadata is {type(metadata)}"
        # Falsify PC mask for ground points.
        frame.pc.mask = frame.pc.mask & metadata.in_range_mask
        # Falsify Flow mask for ground points.
        frame.flow.mask = frame.flow.mask & metadata.in_range_mask

        if not self.with_ground:
            frame.pc.mask = frame.pc.mask & ~metadata.is_ground_points
            frame.flow.mask = frame.flow.mask & ~metadata.is_ground_points

        return frame

    @abstractmethod
    def _load_from_sequence(
        self,
        sequence: AbstractAVLidarSequence,
        relative_idx: int,
        subsequence_start_idx: int,
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        raise NotImplementedError

    def __getitem__(self, dataset_idx, verbose: bool = False) -> list[TimeSyncedSceneFlowFrame]:
        sequence_idx, (
            subsequence_start_idx,
            _,
        ) = self.dataset_to_sequence_subsequence_idx[dataset_idx]

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
            self._process_frame_with_metadata(item, metadata)
            for item, metadata in zip(scene_flow_items, scene_flow_metadata)
        ]

        return scene_flow_items


class CausalSeqLoaderDataset(BaseAbstractSeqLoaderDataset):
    def _build_new_cache(self) -> CacheLookup:
        cache_file = self._get_idx_lookup_cache_file()
        # Build map from dataset index to sequence and subsequence range.
        # This is a causal loader, so we load self.subsequence_length frames at a time
        # WITH overlap between chunks.
        dataset_idx_to_sequence_subsequence_range: CacheLookup = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(len(sequence) - self.subsequence_length + 1):
                dataset_idx_to_sequence_subsequence_range.append(
                    (
                        sequence_idx,
                        (subsequence_start_idx, subsequence_start_idx + self.subsequence_length),
                    )
                )

        print(
            f"Loaded {len(dataset_idx_to_sequence_subsequence_range)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_idx_to_sequence_subsequence_range)
        return dataset_idx_to_sequence_subsequence_range

    def _load_from_sequence(
        self,
        sequence: AbstractAVLidarSequence,
        relative_idx: int,
        subsequence_start_idx: int,
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        assert isinstance(
            sequence, AbstractAVLidarSequence
        ), f"sequence is {type(sequence)}, not AbstractAVLidarSequence"
        # As a causal loader, the central frame is the last frame in the sequence.
        in_subsequence_tgt_index = self.subsequence_length - 1
        return sequence.load(
            subsequence_start_idx + relative_idx,
            subsequence_start_idx + in_subsequence_tgt_index,
            with_flow=relative_idx != self.subsequence_length - 1,
        )

    def _get_idx_lookup_cache_file(self) -> Path:
        cache_file = (
            self.idx_lookup_cache_path / f"causal_subsequence_{self.subsequence_length}_lookup.pkl"
        )
        return cache_file

    def loader_type(self) -> LoaderType:
        return LoaderType.CAUSAL


class NonCausalSeqLoaderDataset(BaseAbstractSeqLoaderDataset):
    def _build_new_cache(self) -> CacheLookup:
        cache_file = self._get_idx_lookup_cache_file()
        # Build map from dataset index to sequence and subsequence index.
        # This is a noncausal loader, so we load self.subsequence_length frames at a time
        # WITHOUT overlap between chunks.
        dataset_to_sequence_subsequence_idx = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(
                0, len(sequence) - self.subsequence_length + 1, self.subsequence_length
            ):
                dataset_to_sequence_subsequence_idx.append(
                    (
                        sequence_idx,
                        (subsequence_start_idx, subsequence_start_idx + self.subsequence_length),
                    )
                )

        print(
            f"Loaded {len(dataset_to_sequence_subsequence_idx)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_to_sequence_subsequence_idx)
        return dataset_to_sequence_subsequence_idx

    def _load_from_sequence(
        self,
        sequence: AbstractAVLidarSequence,
        relative_idx: int,
        subsequence_start_idx: int,
        other_load_args: dict[str, Any] = {},
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        assert isinstance(
            sequence, AbstractAVLidarSequence
        ), f"sequence is {type(sequence)}, not AbstractAVLidarSequence"
        # As a non-causal loader, the central frame is the middle frame in the sequence.
        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        return sequence.load(
            subsequence_start_idx + relative_idx,
            subsequence_start_idx + in_subsequence_tgt_index,
            **other_load_args,
        )

    def _get_idx_lookup_cache_file(self) -> Path:
        cache_file = (
            self.idx_lookup_cache_path
            / f"non_causal_subsequence_{self.subsequence_length}_lookup.pkl"
        )
        return cache_file

    def loader_type(self) -> LoaderType:
        return LoaderType.NON_CAUSAL
