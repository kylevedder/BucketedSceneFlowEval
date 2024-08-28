from pathlib import Path

import numpy as np
import open3d as o3d
import tqdm

from bucketed_scene_flow_eval.datastructures import (
    SE3,
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
from bucketed_scene_flow_eval.utils import load_by_extension


class OrbbecAstra(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        flow_dir: Path | None,
        subsequence_length: int = 2,
        extension_name: str = ".pcd",
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

    def __len__(self):
        return len(self.pointclouds) - self.subsequence_length + 1

    def _load_file(self, pcd_file: Path) -> PointCloud:
        data = load_by_extension(pcd_file, verbose=False)
        return PointCloud(data)

    def evaluator(self) -> Evaluator:
        return EmptyEvaluator()

    def loader_type(self):
        return LoaderType.NON_CAUSAL

    def _load_pose_info(self) -> PoseInfo:
        sensor_to_right_hand = SE3(
            # fmt: off
            rotation_matrix=np.array([[0, 0, 1],
                                      [-1, 0, 0],
                                      [0, -1, 0]]),
            # fmt: on
            translation=np.array([0.0, 0.0, 0.0]),
        )

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
            sensor_to_ego=right_hand_to_ego.compose(sensor_to_right_hand),
            ego_to_global=SE3.identity(),
        )

    def _get_sequence_frame(self, idx: int) -> TimeSyncedSceneFlowFrame:
        pc = self.pointclouds[idx]

        semantics = np.zeros(pc.shape[0], dtype=SemanticClassId)
        pose_info = self._load_pose_info()

        ego_pc = pc.transform(pose_info.sensor_to_ego)

        is_valid_mask = (ego_pc.points[:, 0] < 1.7) & (ego_pc.points[:, 2] > -1.2)
        pc_frame = SupervisedPointCloudFrame(pc, pose_info, is_valid_mask, semantics)
        rgb_frame_lookup = RGBFrameLookup.empty()
        gt_flow = EgoLidarFlow(full_flow=np.zeros_like(pc.points), mask=is_valid_mask)

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
