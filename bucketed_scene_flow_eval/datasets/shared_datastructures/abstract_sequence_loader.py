# import abstract base class
from abc import ABC, abstractmethod

from .scene_representations import RawItem


class AbstractSequence(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, idx: int, relative_to_idx: int) -> RawItem:
        pass

    @abstractmethod
    def __len__(self):
        pass


class AbstractSequenceLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_sequence(self, sequence_identifier) -> AbstractSequence:
        pass
