from typing import Optional

import numpy as np
import pytest
import tqdm

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.interfaces import AbstractDataset


@pytest.fixture
def waymo_dataset_gt():
    return construct_dataset(
        "waymoopencausalsceneflow",
        dict(root_dir="/tmp/waymo_open_processed_flow_tiny/training"),
    )


@pytest.fixture
def argo_dataset_gt_with_ground():
    return construct_dataset(
        "argoverse2causalsceneflow",
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
        "argoverse2causalsceneflow",
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
        "argoverse2causalsceneflow",
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
        "argoverse2causalsceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=False,
            with_ground=False,
        ),
    )


@pytest.fixture
def argo_dataset_test_no_flow_no_ground():
    return construct_dataset(
        "argoverse2causalsceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/test",
            with_rgb=False,
            with_ground=False,
            load_flow=False,
        ),
    )


@pytest.fixture
def argo_dataset_test_no_flow_with_ground():
    return construct_dataset(
        "argoverse2causalsceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/test",
            with_rgb=False,
            with_ground=True,
            load_flow=False,
        ),
    )


@pytest.fixture
def argo_box_dataset():
    return construct_dataset(
        "argoverse2noncausalsceneflow",
        dict(
            root_dir="/tmp/argoverse2_small/val",
            subsequence_length=150,
            load_boxes=True,
            range_crop_type="ego",
            log_subset=["02678d04-cc9f-3148-9f95-1ba66347dff9"],
        ),
    )


def _validate_dataloader(
    dataloader: AbstractDataset,
    full_pc_size: int,
    masked_pc_size: int,
    expected_len: int = 1,
    expected_num_frames: int = 2,
):
    assert len(dataloader) == expected_len, f"Expected {expected_len} scene, got {len(dataloader)}"

    dataloader_entries = dataloader[0]
    assert (
        len(dataloader_entries) == expected_num_frames
    ), f"Expected list of length {expected_num_frames}, got {len(dataloader_entries)}"
    # Failure of the following line indicates that the __getitem__ method is broken.

    num_iteration_entries = 0
    for entry in dataloader:
        assert isinstance(entry, list), f"Expected list, got {type(entry)}"
        assert (
            len(entry) == expected_num_frames
        ), f"Expected list of length {expected_num_frames}, got {len(entry)}"
        item_t1 = entry[0]

        assert (
            full_pc_size == item_t1.pc.full_pc.shape[0]
        ), f"Expected full pc to be of size {full_pc_size}, got {item_t1.pc.full_pc.shape[0]}"

        assert (
            masked_pc_size == item_t1.pc.pc.shape[0]
        ), f"Expected masked pc to be of size {masked_pc_size}, got {item_t1.pc.pc.shape[0]}"

        num_iteration_entries += 1

    # Check that we actually iterated over the dataset.
    assert (
        num_iteration_entries == expected_len
    ), f"Expected {expected_len} iteration, got {num_iteration_entries}"


def test_argo_box_dataset(argo_box_dataset):
    _validate_dataloader(argo_box_dataset, 95381, 69600, 1, 150)


def test_waymo_dataset(waymo_dataset_gt):
    _validate_dataloader(waymo_dataset_gt, 124364, 124364)


def test_argo_dataset_gt_with_ground(argo_dataset_gt_with_ground):
    _validate_dataloader(argo_dataset_gt_with_ground, 90430, 74218)


def test_argo_dataset_pseudo_with_ground(argo_dataset_pseudo_with_ground):
    _validate_dataloader(argo_dataset_pseudo_with_ground, 90430, 74218)


def test_argo_dataset_test_no_flow_with_ground(argo_dataset_test_no_flow_with_ground):
    _validate_dataloader(argo_dataset_test_no_flow_with_ground, 90430, 74218)


def test_argo_dataset_gt_no_ground(argo_dataset_gt_no_ground):
    _validate_dataloader(argo_dataset_gt_no_ground, 90430, 65225)


def test_argo_dataset_pseudo_no_ground(argo_dataset_pseudo_no_ground):
    _validate_dataloader(argo_dataset_pseudo_no_ground, 90430, 65225)


def test_argo_dataset_test_no_flow_no_ground(argo_dataset_test_no_flow_no_ground):
    _validate_dataloader(argo_dataset_test_no_flow_no_ground, 90430, 65225)
