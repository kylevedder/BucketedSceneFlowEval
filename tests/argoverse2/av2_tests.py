from pathlib import Path

import numpy as np
import pytest

from bucketed_scene_flow_eval.datasets.argoverse2 import (
    ArgoverseSceneFlowSequenceLoader,
)
from bucketed_scene_flow_eval.datasets.shared_datastructures import RawItem
from bucketed_scene_flow_eval.datastructures import SE3, PoseInfo


@pytest.fixture
def av2_loader() -> ArgoverseSceneFlowSequenceLoader:
    return ArgoverseSceneFlowSequenceLoader(
        raw_data_path=Path("/tmp/argoverse2_tiny/val"),
        flow_data_path=Path("/tmp/argoverse2_tiny/val_sceneflow_feather/"),
        with_rgb=True,
    )


def _are_poses_close(pose1: SE3, pose2: SE3, tol: float = 1e-6) -> bool:
    # Use PyTest's approx to compare the poses
    return pose1.translation == pytest.approx(
        pose2.translation, abs=tol
    ) and pose1.rotation_matrix == pytest.approx(pose2.rotation_matrix, abs=tol)


def _are_poseinfos_close(pose1: PoseInfo, pose2: PoseInfo, tol: float = 1e-6) -> bool:
    return _are_poses_close(pose1.sensor_to_ego, pose2.sensor_to_ego, tol) and _are_poses_close(
        pose1.ego_to_global, pose2.ego_to_global, tol
    )


def _load_reference_sequence(av2_loader: ArgoverseSceneFlowSequenceLoader) -> RawItem:
    sequence_id = "02678d04-cc9f-3148-9f95-1ba66347dff9"
    assert sequence_id in av2_loader.get_sequence_ids(), f"sequence_id {sequence_id} not found"
    sequence = av2_loader.load_sequence(sequence_id)
    assert len(sequence) > 0, f"no frames found in {sequence_id}"
    return sequence.load(0, 0)


def test_rgb_sizes(av2_loader: ArgoverseSceneFlowSequenceLoader):
    first_frame = _load_reference_sequence(av2_loader)
    assert len(first_frame.rgbs) == 5, f"expected 5 cameras, got {len(first_frame.rgbs)}"

    # Expect the shapes
    expected_img_shapes = {
        "ring_side_left": (1550, 2048, 3),
        "ring_front_left": (1550, 2048, 3),
        "ring_front_center": (2048, 1550, 3),
        "ring_front_right": (1550, 2048, 3),
        "ring_side_right": (1550, 2048, 3),
    }

    items = first_frame.rgbs.items()
    assert len(items) == len(
        expected_img_shapes
    ), f"expected {len(expected_img_shapes)} items, got {len(items)}"
    for name, rgb_frame in items:
        expected_shape = expected_img_shapes[name]
        assert (
            rgb_frame.rgb.shape == expected_shape
        ), f"expected shape {expected_shape} for {name}, got {rgb_frame.rgb.shape}"


def test_rgb_poses(av2_loader: ArgoverseSceneFlowSequenceLoader):
    first_frame = _load_reference_sequence(av2_loader)
    assert len(first_frame.rgbs) == 5, f"expected 5 cameras, got {len(first_frame.rgbs)}"
    # fmt: off
    expected_poses = {
        "ring_side_left": PoseInfo(sensor_to_ego=SE3(rotation_matrix=np.array([[-1.64434371e-01,  9.84926460e-01, -5.36768667e-02],
                                                                               [-9.86344525e-01, -1.64694543e-01, -4.29812899e-04],
                                                                               [-9.26362111e-03,  5.28732076e-02,  9.98558265e-01]]),
                                                     translation=np.array([ 0.01907694,  1.32876607, -1.39770751])),
                                   ego_to_global=SE3(rotation_matrix=np.array([[ 1.00000000e+00, -1.82959117e-17,  4.33680869e-19],
                                                                               [-1.82959117e-17,  1.00000000e+00,  3.46944695e-18],
                                                                               [ 4.33680869e-19,  3.46944695e-18,  1.00000000e+00]]),
                                                     translation=np.array([0., 0., 0.,]))),
        "ring_front_left": PoseInfo(sensor_to_ego=SE3(rotation_matrix=np.array([[7.04684427e-01, 7.07471628e-01, -5.38864882e-02],
                                                                                [-7.08494837e-01, 7.05715976e-01, 1.62409907e-04],
                                                                                [3.81434560e-02, 3.80638510e-02, 9.98547054e-01]]),
                                                       translation=np.array([-1.15679509, 0.95010363, -1.45731286])),
                                    ego_to_global=SE3(rotation_matrix=np.array([[ 1.00000000e+00, -1.82959117e-17,  4.33680869e-19],
                                                                                [-1.82959117e-17,  1.00000000e+00,  3.46944695e-18],
                                                                                [ 4.33680869e-19,  3.46944695e-18,  1.00000000e+00]]),
                                                      translation=np.array([0., 0., 0.]))),
        "ring_front_center": PoseInfo(sensor_to_ego=SE3(rotation_matrix=np.array([[0.99999065, 0.00342785, -0.00263813],
                                                                                  [-0.0034088, 0.99996833, 0.00719143],
                                                                                  [0.0026627, -0.00718237, 0.99997066]]),
                                                        translation=np.array([-1.6242827, -0.01104278, -1.40095728])),
                                      ego_to_global=SE3(rotation_matrix=np.array([[ 1.00000000e+00, -1.82959117e-17,  4.33680869e-19],
                                                                                  [-1.82959117e-17,  1.00000000e+00,  3.46944695e-18],
                                                                                  [ 4.33680869e-19,  3.46944695e-18,  1.00000000e+00]]),
                                                        translation=np.array([0., 0., 0.]))),
        "ring_front_right": PoseInfo(sensor_to_ego=SE3(rotation_matrix=np.array([[0.70691132, -0.70462986, -0.06142598],
                                                                                 [0.70606495, 0.70814306, 0.00238596],
                                                                                 [0.04181716, -0.04505739, 0.99810879]]),
                                                       translation=np.array([-1.1517093, -0.95880766, -1.46495846])),
                                     ego_to_global=SE3(rotation_matrix=np.array([[ 1.00000000e+00, -1.82959117e-17,  4.33680869e-19],
                                                                                 [-1.82959117e-17,  1.00000000e+00,  3.46944695e-18],
                                                                                 [ 4.33680869e-19,  3.46944695e-18,  1.00000000e+00]]),
                                                       translation=np.array([0., 0., 0.]))),
        "ring_side_right": PoseInfo(sensor_to_ego=SE3(rotation_matrix=np.array([[-0.15424876, -0.9859215, -0.06454548],
                                                                                [0.98800738, -0.15437699, -0.00302607],
                                                                                [-0.00698087, -0.06423818, 0.99791018]]),
                                                      translation=np.array([0.01941238, -1.33103954, -1.3939121])),
                                    ego_to_global=SE3(rotation_matrix=np.array([[ 1.00000000e+00, -1.82959117e-17,  4.33680869e-19],
                                                                                [-1.82959117e-17,  1.00000000e+00,  3.46944695e-18],
                                                                                [ 4.33680869e-19,  3.46944695e-18,  1.00000000e+00]]),
                                    translation=np.array([0., 0., 0.])))
    }
    # fmt: on
    for name, pose in expected_poses.items():
        assert name in first_frame.rgbs, f"expected {name} to be in the frame"
        assert _are_poseinfos_close(
            first_frame.rgbs[name].pose, pose
        ), f"expected pose for {name} to match"
