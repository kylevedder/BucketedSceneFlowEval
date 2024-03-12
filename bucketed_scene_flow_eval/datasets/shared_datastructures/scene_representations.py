from dataclasses import dataclass
from typing import Optional

import numpy as np

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    CameraProjection,
    PointCloud,
    PointCloudFrame,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
    Timestamp,
)


@dataclass(kw_only=True)
class RawItem:
    pc: PointCloudFrame
    is_ground_points: np.ndarray
    in_range_mask: np.ndarray
    rgbs: RGBFrameLookup
    log_id: str
    log_idx: int
    log_timestamp: Timestamp


@dataclass(kw_only=True)
class SceneFlowItem(RawItem):
    pc_classes: np.ndarray
    flowed_pc: PointCloudFrame
