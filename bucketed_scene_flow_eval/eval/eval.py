# Import abstract base class for evaluator
from abc import ABC, abstractmethod
from typing import Any

from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    TimeSyncedSceneFlowItem,
)


class Evaluator(ABC):
    @abstractmethod
    def eval(self, predictions: EgoLidarFlow, gt: TimeSyncedSceneFlowItem):
        pass

    @abstractmethod
    def compute_results(self, save_results: bool = True) -> dict[Any, Any]:
        pass
