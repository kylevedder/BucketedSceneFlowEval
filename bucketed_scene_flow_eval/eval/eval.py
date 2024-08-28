# Import abstract base class for evaluator
from abc import ABC, abstractmethod
from typing import Any

from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    TimeSyncedSceneFlowFrame,
)


class Evaluator(ABC):
    @abstractmethod
    def eval(self, predictions: EgoLidarFlow, gt: TimeSyncedSceneFlowFrame):
        raise NotImplementedError

    @abstractmethod
    def compute_results(self, save_results: bool = True) -> dict[Any, Any]:
        raise NotImplementedError


class EmptyEvaluator(Evaluator):
    def __init__(self):
        pass

    def eval(self, predictions: EgoLidarFlow, gt: TimeSyncedSceneFlowFrame):
        pass

    def compute_results(self, save_results: bool = True) -> dict[Any, Any]:
        return {}
