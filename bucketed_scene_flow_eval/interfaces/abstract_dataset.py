# import abstract base class
import enum
from abc import ABC, abstractmethod

from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from bucketed_scene_flow_eval.eval import Evaluator


class LoaderType(enum.Enum):
    CAUSAL = 0
    NON_CAUSAL = 1


class AbstractDataset:
    @abstractmethod
    def __getitem__(self, idx: int) -> list[TimeSyncedSceneFlowFrame]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evaluator(self) -> Evaluator:
        raise NotImplementedError

    @abstractmethod
    def loader_type(self) -> LoaderType:
        raise NotImplementedError
