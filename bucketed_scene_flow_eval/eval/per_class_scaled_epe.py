from bucketed_scene_flow_eval.datastructures import (
    EstimatedParticleTrajectories,
    GroundTruthParticleTrajectories,
    Timestamp,
    ParticleClassId,
)
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Set, Any, Union
import numpy as np
import pickle
import json
import enum
from .eval import Evaluator
import copy
from .base_per_frame_sceneflow_eval import PerFrameSceneFlowEvaluator, BaseEvalFrameResult


class ScalingType(enum.Enum):
    CONSTANT = "constant"
    FOUR_D = "4d"
    FOUR_D_01 = "4d_01"

    @staticmethod
    def from_str(s: str):
        # Iterate over members and check if match string by value
        for member in ScalingType:
            if member.value == s:
                return member
        raise ValueError(f"ScalingType {s} not found.")


class ScaledEvalFrameResult(BaseEvalFrameResult):
    def __init__(
        self,
        scaling_type: ScalingType,
        *args,
        **kwargs,
    ):
        self.scaling_type = scaling_type

        # Pass through all arguments to super class
        super().__init__(*args, **kwargs)

    def _scale_flows(self, gt_flow: np.ndarray,
                     pred_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self.scaling_type == ScalingType.CONSTANT:
            gt_speeds = np.linalg.norm(gt_flow, axis=1)
            scaled_gt_flow = gt_flow / (gt_speeds + 1)[:, None]
            scaled_pred_flow = pred_flow / (gt_speeds + 1)[:, None]
            return scaled_gt_flow, scaled_pred_flow

        elif self.scaling_type == ScalingType.FOUR_D or self.scaling_type == ScalingType.FOUR_D_01:

            def augment_flow(flow, value: float = 1):
                # Add a fourth dimension of ones to the flow
                return np.concatenate(
                    [flow, np.ones((flow.shape[0], 1)) * value], axis=1)

            def deaugment_flow(flow):
                # Remove the fourth dimension of ones from the flow
                return flow[:, :3]

            augmentation_value = 1.0
            if self.scaling_type == ScalingType.FOUR_D_01:
                augmentation_value = 0.1

            gt_flow_aug = augment_flow(gt_flow, value=augmentation_value)
            pred_flow_aug = augment_flow(pred_flow, value=augmentation_value)

            gt_aug_norm = np.linalg.norm(gt_flow_aug, axis=1)

            scaled_gt_flow_aug = gt_flow_aug / gt_aug_norm[:, None]
            scaled_pred_flow_aug = pred_flow_aug / gt_aug_norm[:, None]

            scaled_gt_flow = deaugment_flow(scaled_gt_flow_aug)
            scaled_pred_flow = deaugment_flow(scaled_pred_flow_aug)
            return scaled_gt_flow, scaled_pred_flow
        else:
            raise NotImplementedError(
                f"Scaling type {self.scaling_type} not implemented.")


class PerClassScaledEPEEvaluator(PerFrameSceneFlowEvaluator):
    def __init__(self,
                 scaling_type: str,
                 output_path: Path = Path("/tmp/frame_results/scaled_epe")):
        self.scaling_type = ScalingType.from_str(scaling_type.lower())
        super().__init__(output_path=output_path / self.scaling_type.value)

    def _build_eval_frame_results(self, pc1: np.ndarray,
                                  gt_class_ids: np.ndarray,
                                  gt_flow: np.ndarray, pred_flow: np.ndarray,
                                  ground_truth) -> BaseEvalFrameResult:
        return ScaledEvalFrameResult(self.scaling_type,
                                     pc1,
                                     gt_class_ids,
                                     gt_flow,
                                     pred_flow,
                                     class_id_to_name=ground_truth.pretty_name)
