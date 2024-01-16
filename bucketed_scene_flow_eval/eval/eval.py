# Import abstract base class for evaluator
from abc import ABC, abstractmethod
from typing import Any

from bucketed_scene_flow_eval.datastructures import (
    EstimatedPointFlow,
    GroundTruthPointFlow,
    Timestamp,
)


class Evaluator(ABC):
    @abstractmethod
    def eval(
        self,
        predictions: EstimatedPointFlow,
        ground_truth: GroundTruthPointFlow,
        query_timestamp: Timestamp,
    ):
        pass

    @abstractmethod
    def compute_results(self, save_results: bool = True) -> dict[Any, Any]:
        pass
