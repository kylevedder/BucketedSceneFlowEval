import pytest
import bucketed_scene_flow_eval
from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.datasets import construct_dataset
from typing import Tuple, Dict, List
import numpy as np


@pytest.fixture
def waymo_dataset_gt():
    return construct_dataset(
        "waymoopensceneflow",
        dict(root_dir="/tmp/waymo_open_processed_flow_tiny/training"),
    )


@pytest.fixture
def argo_dataset_gt():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=True,
        ),
    )


@pytest.fixture
def argo_dataset_pseudo():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=False,
        ),
    )


def _process_query(
    query: QuerySceneSequence,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[SE3, SE3], List[np.ndarray], List[SE3]]:
    assert (
        len(query.query_flow_timestamps) == 2
    ), f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
    scene = query.scene_sequence

    # These contain the full problem percepts, not just the ones in the query.
    full_percept_pc_arrays: List[np.ndarray] = []
    full_percept_poses: List[SE3] = []
    # These contain only the percepts in the query.0
    problem_pc_arrays: List[np.ndarray] = []
    problem_poses: List[SE3] = []

    for timestamp in scene.get_percept_timesteps():
        pc_frame = scene[timestamp].pc_frame
        pc_array = pc_frame.global_pc.points.astype(np.float32)
        pose = pc_frame.global_pose

        full_percept_pc_arrays.append(pc_array)
        full_percept_poses.append(pose)

        if timestamp in query.query_flow_timestamps:
            problem_pc_arrays.append(pc_array)
            problem_poses.append(pose)

    assert len(full_percept_pc_arrays) == len(
        full_percept_poses
    ), f"Percept arrays and poses have different lengths."
    assert len(problem_pc_arrays) == len(
        problem_poses
    ), f"Percept arrays and poses have different lengths."
    assert len(problem_pc_arrays) == len(
        query.query_flow_timestamps
    ), f"Percept arrays and poses have different lengths."

    return problem_pc_arrays, problem_poses, full_percept_pc_arrays, full_percept_poses


def _process_gt(result: GroundTruthPointFlow):
    flowed_source_pc = result.world_points[:, 1].astype(np.float32)
    point_cls_array = result.cls_ids
    return flowed_source_pc, point_cls_array


def _validate_dataloader(
    query: QuerySceneSequence, gt: GroundTruthPointFlow, expected_pc_size: int
):
    assert isinstance(
        query, QuerySceneSequence
    ), f"Expected QuerySceneSequence, got {type(query)}"
    assert isinstance(
        gt, GroundTruthPointFlow
    ), f"Expected GroundTruthParticleTrajectories, got {type(gt)}"

    t1, t2 = query.scene_sequence.get_percept_timesteps()
    pc_frame = query.scene_sequence[t1].pc_frame

    assert (
        len(pc_frame.global_pc) == expected_pc_size
    ), f"Expected {expected_pc_size} points, got {len(pc_frame.global_pc)} for WaymoOpen"

    (
        (source_pc, target_pc),
        (source_pose, target_pose),
        full_pc_points_list,
        full_pc_poses_list,
    ) = _process_query(query)

    gt_flowed_source_pc, gt_point_classes = _process_gt(gt)

    assert (
        source_pc.shape == gt_flowed_source_pc.shape
    ), f"Source PC shape mismatch: {source_pc.shape} vs {gt_flowed_source_pc.shape}"


def test_waymo_dataset(waymo_dataset_gt):
    assert len(waymo_dataset_gt) == 1, f"Expected 1 scene, got {len(waymo_dataset_gt)}"
    for query, gt in waymo_dataset_gt:
        _validate_dataloader(query, gt, 124364)


def test_argo_dataset_gt(argo_dataset_gt):
    assert len(argo_dataset_gt) == 1, f"Expected 1 scene, got {len(argo_dataset_gt)}"
    for query, gt in argo_dataset_gt:
        _validate_dataloader(query, gt, 90430)


def test_argo_dataset_pseudo(argo_dataset_pseudo):
    assert (
        len(argo_dataset_pseudo) == 1
    ), f"Expected 1 scene, got {len(argo_dataset_pseudo)}"
    for query, gt in argo_dataset_pseudo:
        _validate_dataloader(query, gt, 90430)
