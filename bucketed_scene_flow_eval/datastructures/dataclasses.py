from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy._typing import NDArray

from .camera_projection import CameraProjection
from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3

SemanticClassId = np.int8
SemanticClassIdArray = NDArray[SemanticClassId]
MaskArray = NDArray[np.bool_]
VectorArray = NDArray[np.float32]


@dataclass
class PoseInfo:
    """Stores pose SE3 objects for a given frame.

    sensor_to_ego: the transformation from sensor frame to ego (vehicle) frame
    ego_to_global: the transformation from ego frame to a consistent global frame, which is defined as the ego pose at a different timestep in the sequence
    """
    sensor_to_ego: SE3
    ego_to_global: SE3

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PoseInfo):
            return False
        return (
            self.sensor_to_ego == __value.sensor_to_ego
            and self.ego_to_global == __value.ego_to_global
        )

    @property
    def sensor_to_global(self) -> SE3:
        return self.ego_to_global @ self.sensor_to_ego

    def __repr__(self) -> str:
        return f"PoseInfo(sensor_to_ego={self.sensor_to_ego}, ego_to_global={self.ego_to_global})"


@dataclass
class EgoLidarFlow:
    """
    Ego frame lidar flow from the ego frame of P0 to the relative frame of P1.
    """

    full_flow: VectorArray
    mask: MaskArray

    @staticmethod
    def make_no_flow(flow_dim: int) -> "EgoLidarFlow":
        return EgoLidarFlow(
            full_flow=np.zeros((flow_dim, 3), dtype=np.float32),
            mask=np.zeros(flow_dim, dtype=bool),
        )

    @staticmethod
    def make_no_flow_like(flow: "EgoLidarFlow") -> "EgoLidarFlow":
        return EgoLidarFlow.make_no_flow(flow.shape[0])

    def __post_init__(self):
        assert self.full_flow.ndim == 2, f"flow must be a 2D array, got {self.full_flow.ndim}"
        assert self.mask.ndim == 1, f"valid_flow_mask must be a 1D array, got {self.mask.ndim}"
        assert self.mask.dtype == bool, f"valid_flow_mask must be boolean, got {self.mask.dtype}"

        assert len(self.full_flow) == len(self.mask), (
            f"flow and valid_flow_mask must have the same length, got {len(self.full_flow)} and "
            f"{len(self.mask)}"
        )

        assert (
            self.full_flow.shape[1] == 3
        ), f"flow must have 3 columns, got {self.full_flow.shape[1]}"

    def __repr__(self) -> str:
        return f"LidarFlow(flow={self.full_flow}, valid_flow_mask={self.mask})"

    @property
    def valid_flow(self) -> VectorArray:
        return self.full_flow[self.mask]

    @property
    def shape(self) -> tuple[int, int]:
        return self.full_flow.shape

    def mask_points(self, mask: MaskArray) -> "EgoLidarFlow":
        assert isinstance(mask, np.ndarray), f"mask must be an ndarray, got {type(mask)}"
        assert mask.ndim == 1, f"mask must be a 1D array, got {mask.ndim}"
        assert mask.dtype == bool, f"mask must be a boolean array, got {mask.dtype}"
        return EgoLidarFlow(full_flow=self.full_flow[mask], mask=self.mask[mask])


@dataclass
class PointCloudFrame:
    """A Point Cloud Frame.

    full_pc: the point cloud as captured in the sensors coordinate frame
    pose: a PoseInfo object that specifies the transformations from sensor -> ego as well as ego -> global.
    mask: a mask for validity in the point cloud, validity is determined by the dataloader and could be any of the following:
        if the point is valid for the purpose of computing scene flow, if the point is ground or not ground, or any other criteria enforced by the dataloader
    """
    full_pc: PointCloud
    pose: PoseInfo
    mask: MaskArray

    @property
    def pc(self) -> PointCloud:
        return self.full_pc.mask_points(self.mask)

    @property
    def full_ego_pc(self) -> PointCloud:
        return self.full_pc.transform(self.pose.sensor_to_ego)

    @property
    def ego_pc(self) -> PointCloud:
        return self.pc.transform(self.pose.sensor_to_ego)

    @property
    def global_pose(self) -> SE3:
        return self.pose.sensor_to_global

    @property
    def full_global_pc(self) -> PointCloud:
        return self.full_pc.transform(self.global_pose)

    @property
    def global_pc(self) -> PointCloud:
        return self.pc.transform(self.global_pose)

    def mask_points(self, mask: MaskArray) -> "PointCloudFrame":
        assert isinstance(mask, np.ndarray), f"mask must be an ndarray, got {type(mask)}"
        assert mask.ndim == 1, f"mask must be a 1D array, got {mask.ndim}"
        assert mask.dtype == bool, f"mask must be a boolean array, got {mask.dtype}"
        return PointCloudFrame(
            full_pc=self.full_pc.mask_points(mask),
            pose=self.pose,
            mask=self.mask[mask],
        )

    def flow(self, flow: EgoLidarFlow) -> "PointCloudFrame":
        return PointCloudFrame(
            full_pc=self.full_pc.flow_masked(flow.valid_flow, flow.mask),
            pose=self.pose,
            mask=self.mask,
        )


@dataclass
class SupervisedPointCloudFrame(PointCloudFrame):
    full_pc_classes: SemanticClassIdArray

    def __post_init__(self):
        # Check pc_classes
        assert isinstance(
            self.full_pc_classes, np.ndarray
        ), f"pc_classes must be an ndarray, got {type(self.full_pc_classes)}"
        assert (
            self.full_pc_classes.ndim == 1
        ), f"pc_classes must be a 1D array, got {self.full_pc_classes.ndim}"
        assert (
            self.full_pc_classes.dtype == SemanticClassId
        ), f"pc_classes must be a SemanticClassId array, got {self.full_pc_classes.dtype}"
        assert len(self.full_pc_classes) == len(
            self.full_pc
        ), f"pc_classes must be the same length as pc, got {len(self.full_pc_classes)} and {len(self.full_pc)}"

    @property
    def pc_classes(self) -> SemanticClassIdArray:
        return self.full_pc_classes[self.mask]

    def mask_points(self, mask: MaskArray) -> "SupervisedPointCloudFrame":
        assert isinstance(mask, np.ndarray), f"mask must be an ndarray, got {type(mask)}"
        assert mask.ndim == 1, f"mask must be a 1D array, got {mask.ndim}"
        assert mask.dtype == bool, f"mask must be a boolean array, got {mask.dtype}"
        return SupervisedPointCloudFrame(
            full_pc=self.full_pc.mask_points(mask),
            pose=self.pose,
            mask=self.mask[mask],
            full_pc_classes=self.full_pc_classes[mask],
        )


@dataclass
class RGBFrame:
    rgb: RGBImage
    pose: PoseInfo
    camera_projection: CameraProjection

    def __repr__(self) -> str:
        return f"RGBFrame(rgb={self.rgb},\npose={self.pose},\ncamera_projection={self.camera_projection})"

    def rescale(self, factor: int) -> "RGBFrame":
        return RGBFrame(
            rgb=self.rgb.rescale(factor),
            pose=self.pose,
            camera_projection=self.camera_projection.rescale(factor),
        )


@dataclass
class RGBFrameLookup:
    lookup: dict[str, RGBFrame]
    entries: list[str]

    @staticmethod
    def empty() -> "RGBFrameLookup":
        return RGBFrameLookup({}, [])

    def __contains__(self, key: str) -> bool:
        return key in self.lookup

    def items(self) -> list[tuple[str, RGBFrame]]:
        return [(key, self.lookup[key]) for key in self.entries]

    def values(self) -> list[RGBFrame]:
        return [self.lookup[key] for key in self.entries]

    def __getitem__(self, key: str) -> RGBFrame:
        return self.lookup[key]

    def __len__(self) -> int:
        return len(self.lookup)


@dataclass(kw_only=True)
class TimeSyncedBaseAuxilaryData:
    pass


@dataclass(kw_only=True)
class TimeSyncedAVLidarData(TimeSyncedBaseAuxilaryData):
    is_ground_points: MaskArray
    in_range_mask: MaskArray


@dataclass(kw_only=True)
class TimeSyncedRawFrame:
    pc: PointCloudFrame
    rgbs: RGBFrameLookup
    log_id: str
    log_idx: int
    log_timestamp: int


@dataclass(kw_only=True)
class TimeSyncedSceneFlowFrame(TimeSyncedRawFrame):
    pc: SupervisedPointCloudFrame
    flow: EgoLidarFlow

    def __post_init__(self):
        assert isinstance(
            self.pc, SupervisedPointCloudFrame
        ), f"pc must be a SupervisedPointCloudFrame, got {type(self.pc)}"
        assert isinstance(
            self.flow, EgoLidarFlow
        ), f"flow must be an EgoLidarFlow, got {type(self.flow)}"

        # Ensure full flow is the same shape as full pc
        assert len(self.flow.full_flow) == len(
            self.pc.full_pc
        ), f"flow and pc must have the same length, got {len(self.flow.full_flow)} and {len(self.pc.full_pc)}"
