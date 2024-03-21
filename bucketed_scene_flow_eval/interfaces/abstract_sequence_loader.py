# import abstract base class
from abc import ABC, abstractmethod

from bucketed_scene_flow_eval.datastructures import (
    TimeSyncedAVLidarData,
    TimeSyncedBaseAuxilaryData,
    TimeSyncedRawFrame,
    TimeSyncedSceneFlowFrame,
)


class AbstractSequence(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(
        self, idx: int, relative_to_idx: int
    ) -> tuple[TimeSyncedRawFrame, TimeSyncedBaseAuxilaryData]:
        pass

    @abstractmethod
    def __len__(self):
        pass


class AbstractAVLidarSequence(AbstractSequence):
    @abstractmethod
    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = True
    ) -> tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]:
        pass


class AbstractSequenceLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_sequence_ids(self) -> list:
        pass

    @abstractmethod
    def load_sequence(self, sequence_identifier) -> AbstractSequence:
        pass

    @abstractmethod
    def cache_folder_name(self) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        return self.load_sequence(self.sequence_id_lst[idx])


class CachedSequenceLoader(AbstractSequenceLoader):
    def __init__(self):
        self.last_loaded_sequence = None
        self.last_loaded_sequence_id = None

    @abstractmethod
    def _load_sequence_uncached(self, sequence_identifier) -> AbstractSequence:
        pass

    def load_sequence(self, sequence_identifier) -> AbstractSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_identifier:
            self.last_loaded_sequence = self._load_sequence_uncached(sequence_identifier)
            self.last_loaded_sequence_id = sequence_identifier

        return self.last_loaded_sequence
