from typing import Optional

import numpy as np
import pytest
import tqdm

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import *


@pytest.fixture
def waymo_dataset_gt():
    return construct_dataset(
        "waymoopensceneflow",
        dict(root_dir="/tmp/waymo_open_processed_flow_tiny/training"),
    )


@pytest.fixture
def argo_dataset_gt_with_ground():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=True,
            with_ground=True,
        ),
    )


@pytest.fixture
def argo_dataset_pseudo_with_ground():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=False,
            with_ground=True,
        ),
    )


@pytest.fixture
def argo_dataset_gt_no_ground():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=True,
            with_ground=False,
        ),
    )


@pytest.fixture
def argo_dataset_pseudo_no_ground():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=False,
            with_ground=False,
        ),
    )


def _process_query(
    query: QuerySceneSequence,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[SE3, SE3], list[np.ndarray], list[SE3]]:
    assert (
        len(query.query_flow_timestamps) == 2
    ), f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
    scene = query.scene_sequence

    # These contain the all problem percepts, not just the ones in the query.
    all_percept_pc_arrays: list[np.ndarray] = []
    all_percept_poses: list[SE3] = []
    # These contain only the percepts in the query.
    query_pc_arrays: list[np.ndarray] = []
    query_poses: list[SE3] = []

    for timestamp in scene.get_percept_timesteps():
        pc_frame = scene[timestamp].pc_frame
        pc_array = pc_frame.full_global_pc.points.astype(np.float32)
        pose = pc_frame.global_pose

        all_percept_pc_arrays.append(pc_array)
        all_percept_poses.append(pose)

        if timestamp in query.query_flow_timestamps:
            query_pc_arrays.append(pc_array)
            query_poses.append(pose)

    assert len(all_percept_pc_arrays) == len(
        all_percept_poses
    ), f"Percept arrays and poses have different lengths."
    assert len(query_pc_arrays) == len(
        query_poses
    ), f"Percept arrays and poses have different lengths."
    assert len(query_pc_arrays) == len(
        query.query_flow_timestamps
    ), f"Percept arrays and poses have different lengths."

    return (
        query_pc_arrays,
        query_poses,
        all_percept_pc_arrays,
        all_percept_poses,
    )


def _process_gt(result: GroundTruthPointFlow):
    flowed_source_pc = result.world_points[:, 1].astype(np.float32)
    is_valid_mask = result.is_valid_flow
    point_cls_array = result.cls_ids
    return flowed_source_pc, is_valid_mask, point_cls_array


def _validate_dataloader_elements(
    query: QuerySceneSequence,
    gt: GroundTruthPointFlow,
    expected_pc_size: Optional[int],
):
    assert isinstance(query, QuerySceneSequence), f"Expected QuerySceneSequence, got {type(query)}"
    assert isinstance(
        gt, GroundTruthPointFlow
    ), f"Expected GroundTruthParticleTrajectories, got {type(gt)}"

    t1, t2 = query.scene_sequence.get_percept_timesteps()
    pc_frame = query.scene_sequence[t1].pc_frame

    if expected_pc_size is not None:
        assert (
            len(pc_frame.global_pc) == expected_pc_size
        ), f"Expected {expected_pc_size} points, got {len(pc_frame.global_pc)}"

    (
        (source_pc, target_pc),
        (source_pose, target_pose),
        full_pc_points_list,
        full_pc_poses_list,
    ) = _process_query(query)

    gt_flowed_source_pc, is_valid_flow_mask, gt_point_classes = _process_gt(gt)

    assert (
        source_pc.shape == gt_flowed_source_pc.shape
    ), f"Source PC shape mismatch: {source_pc.shape} vs {gt_flowed_source_pc.shape}"

    assert source_pc.shape[0] == is_valid_flow_mask.shape[0], (
        f"Source PC and is_valid_flow_mask shape mismatch: "
        f"{source_pc.shape[0]} vs {is_valid_flow_mask.shape[0]}"
    )

    assert gt_point_classes.shape[0] == is_valid_flow_mask.shape[0], (
        f"Point classes and is_valid_flow_mask shape mismatch: "
        f"{gt_point_classes.shape[0]} vs {is_valid_flow_mask.shape[0]}"
    )


def _validate_dataloader(dataloader, pc_size: Optional[int], expected_len: int = 1):
    assert len(dataloader) == expected_len, f"Expected {expected_len} scene, got {len(dataloader)}"

    num_iteration_entries = 0
    for query, gt in tqdm.tqdm(dataloader):
        _validate_dataloader_elements(query, gt, pc_size)
        num_iteration_entries += 1

    # Check that we actually iterated over the dataset.
    assert (
        num_iteration_entries == expected_len
    ), f"Expected {expected_len} iteration, got {num_iteration_entries}"


def test_waymo_dataset(waymo_dataset_gt):
    _validate_dataloader(waymo_dataset_gt, 124364)


def test_argo_dataset_gt_with_ground(argo_dataset_gt_with_ground):
    _validate_dataloader(argo_dataset_gt_with_ground, 90430)


def test_argo_dataset_pseudo_with_ground(argo_dataset_pseudo_with_ground):
    _validate_dataloader(argo_dataset_pseudo_with_ground, 90430)


def test_argo_dataset_gt_no_ground(argo_dataset_gt_no_ground):
    _validate_dataloader(argo_dataset_gt_no_ground, 80927)


def test_argo_dataset_pseudo_no_ground(argo_dataset_pseudo_no_ground):
    _validate_dataloader(argo_dataset_pseudo_no_ground, 80927)
