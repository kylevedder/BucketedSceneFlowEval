# import abstract base class
from abc import ABC, abstractmethod

from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowItem
from bucketed_scene_flow_eval.eval import Evaluator


class AbstractDataset:
    @abstractmethod
    def __getitem__(self, idx: int) -> list[TimeSyncedSceneFlowItem]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def evaluator(self) -> Evaluator:
        pass
