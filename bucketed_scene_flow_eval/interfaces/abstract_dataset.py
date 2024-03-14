# import abstract base class
from abc import ABC, abstractmethod

from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowItem


class AbstractDataset:
    @abstractmethod
    def __getitem__(self, idx: int) -> list[TimeSyncedSceneFlowItem]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
