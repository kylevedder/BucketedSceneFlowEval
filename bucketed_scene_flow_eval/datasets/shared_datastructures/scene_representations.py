from dataclasses import dataclass
from typing import Optional

import numpy as np

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    CameraProjection,
    PointCloud,
    RGBImage,
    Timestamp,
)


@dataclass(kw_only=True)
class RawItem:
    ego_pc: PointCloud
    ego_pc_with_ground: PointCloud
    relative_pc: PointCloud
    relative_pc_with_ground: PointCloud
    is_ground_points: np.ndarray
    in_range_mask: np.ndarray
    in_range_mask_with_ground: np.ndarray
    rgb: Optional[RGBImage]
    rgb_camera_projection: Optional[CameraProjection]
    rgb_camera_ego_pose: Optional[SE3]
    relative_pose: SE3
    log_id: str
    log_idx: int
    log_timestamp: Timestamp


@dataclass(kw_only=True)
class SceneFlowItem(RawItem):
    ego_flowed_pc: PointCloud
    ego_flowed_pc_with_ground: PointCloud
    relative_flowed_pc: PointCloud
    relative_flowed_pc_with_ground: PointCloud
    pc_classes: np.ndarray
    pc_classes_with_ground: np.ndarray
