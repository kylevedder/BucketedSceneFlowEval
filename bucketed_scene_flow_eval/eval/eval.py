from bucketed_scene_flow_eval.datastructures import (
    EstimatedPointFlow,
    GroundTruthPointFlow,
    Timestamp,
    ParticleClassId,
)
import time
import numpy as np
import pandas as pd

from typing import Tuple, Dict, List, Set, Any, Union

from dataclasses import dataclass
from pathlib import Path

# Import abstract base class for evaluator
from abc import ABC, abstractmethod


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
    def compute_results(self, save_results: bool = True) -> Dict[Any, Any]:
        pass
