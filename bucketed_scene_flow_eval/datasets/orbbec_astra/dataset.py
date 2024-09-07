from pathlib import Path

import numpy as np
import open3d as o3d
import tqdm

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    ColoredSupervisedPointCloudFrame,
    EgoLidarFlow,
    PointCloud,
    PoseInfo,
    RGBFrameLookup,
    SemanticClassId,
    SupervisedPointCloudFrame,
    TimeSyncedSceneFlowFrame,
)
from bucketed_scene_flow_eval.eval import EmptyEvaluator, Evaluator
from bucketed_scene_flow_eval.interfaces import AbstractDataset, LoaderType
from bucketed_scene_flow_eval.utils import load_feather, load_pickle


class OrbbecAstra(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        flow_dir: Path | None,
        subsequence_length: int = 2,
        extension_name: str = ".pkl",
    ) -> None:
        root_dir = Path(root_dir)
        self.data_dir = root_dir
        self.flow_dir = Path(flow_dir) if flow_dir is not None else None
        self.subsequence_length = subsequence_length
        self.pointclouds = [
            self._load_file(pcd_file)
            for pcd_file in tqdm.tqdm(
                sorted(root_dir.glob(f"*{extension_name}")), desc="Loading ORBBEC pointclouds"
            )
        ]
        assert (
            len(self.pointclouds) >= subsequence_length
        ), f"Need at least {subsequence_length} frames, found {len(self.pointclouds)} in {root_dir}/*{extension_name}"

        # Magic numbers to scale the pointclouds to be in the same range as the argoverse data we are training on
        # These numbers are derived from looking at the pointclouds themselves

        self.scale = 35
        self.center_translation = -np.array([1.13756592, 0.21126675, -1.04425789])
        self.bg_delete_x_max = (1.7 + self.center_translation[0]) * self.scale
        self.bg_delete_z_min = (-1.2 + self.center_translation[2]) * self.scale

    def __len__(self):
        return len(self.pointclouds) - self.subsequence_length + 1

    def _load_file(self, data_file: Path) -> tuple[PointCloud, np.ndarray]:
        data = load_pickle(data_file, verbose=False)
        points = data[:, :3]
        colors = data[:, 3:]
        return PointCloud(points), colors

    def evaluator(self) -> Evaluator:
        return EmptyEvaluator()

    def loader_type(self):
        return LoaderType.NON_CAUSAL

    def _load_pose_info(self) -> PoseInfo:
        # Convert from standard sensor coordinate system to right hand coordinate system we use.
        sensor_to_right_hand = SE3(
            # fmt: off
            rotation_matrix=np.array([[0, 0, 1],
                                      [-1, 0, 0],
                                      [0, -1, 0]]),
            # fmt: on
            translation=np.array([0.0, 0.0, 0.0]),
        )

        # The sensor is rotated down 30 degrees, so we need to rotate it back up so the table is level.
        theta_degrees = 30
        theta_radians = np.radians(theta_degrees)
        rotation_matrix_y = np.array(
            [
                [np.cos(theta_radians), 0, np.sin(theta_radians)],
                [0, 1, 0],
                [-np.sin(theta_radians), 0, np.cos(theta_radians)],
            ]
        )
        right_hand_to_ego = SE3(
            rotation_matrix=rotation_matrix_y,
            translation=np.array([0.0, 0.0, 0.0]),
        )

        return PoseInfo(
            sensor_to_ego=right_hand_to_ego.compose(sensor_to_right_hand)
            .translate(self.center_translation)
            .scale(self.scale),
            ego_to_global=SE3.identity(),
        )

    def _load_flow(self, idx: int, pc_frame: SupervisedPointCloudFrame) -> EgoLidarFlow:
        if self.flow_dir is None:
            return EgoLidarFlow(full_flow=np.zeros_like(pc_frame.full_pc), mask=pc_frame.mask)
        flow_file = self.flow_dir / f"{idx:010d}.feather"
        if not flow_file.exists():
            return EgoLidarFlow(full_flow=np.zeros_like(pc_frame.full_pc), mask=pc_frame.mask)
        flow_feather = load_feather(flow_file, verbose=False)
        flow_x = flow_feather["flow_tx_m"].to_numpy()
        flow_y = flow_feather["flow_ty_m"].to_numpy()
        flow_z = flow_feather["flow_tz_m"].to_numpy()
        flow = np.stack([flow_x, flow_y, flow_z], axis=-1)
        flow_mask = flow_feather["is_valid"].to_numpy()
        assert len(flow) == len(
            pc_frame.full_pc
        ), f"Expected {len(pc_frame.full_pc)} points, found {len(flow)}"
        assert np.all(
            flow_mask == pc_frame.mask
        ), f"Founds {np.sum(flow_mask)} masked points, expected {np.sum(pc_frame.mask)}"
        return EgoLidarFlow(full_flow=flow, mask=pc_frame.mask)

    def _get_sequence_frame(self, idx: int) -> TimeSyncedSceneFlowFrame:
        pc, color = self.pointclouds[idx]

        semantics = np.zeros(pc.shape[0], dtype=SemanticClassId)
        pose_info = self._load_pose_info()

        ego_pc = pc.transform(pose_info.sensor_to_ego)

        is_valid_mask = (ego_pc.points[:, 0] < self.bg_delete_x_max) & (
            ego_pc.points[:, 2] > self.bg_delete_z_min
        )
        pc_frame = ColoredSupervisedPointCloudFrame(pc, pose_info, is_valid_mask, semantics, color)
        rgb_frame_lookup = RGBFrameLookup.empty()
        gt_flow = self._load_flow(idx, pc_frame)

        return TimeSyncedSceneFlowFrame(
            pc=pc_frame,
            auxillary_pc=None,
            rgbs=rgb_frame_lookup,
            log_id=self.data_dir.name,
            log_idx=idx,
            log_timestamp=idx,
            flow=gt_flow,
        )

    def __getitem__(self, idx: int) -> list[TimeSyncedSceneFlowFrame]:
        # Minibatch logic
        return [self._get_sequence_frame(idx + i) for i in range(self.subsequence_length)]
