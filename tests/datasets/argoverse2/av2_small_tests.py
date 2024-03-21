from pathlib import Path

import numpy as np
import pytest

from bucketed_scene_flow_eval.datasets import (
    Argoverse2CausalSceneFlow,
    Argoverse2NonCausalSceneFlow,
)
from bucketed_scene_flow_eval.datasets.argoverse2 import (
    ArgoverseSceneFlowSequenceLoader,
)
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    PoseInfo,
    TimeSyncedAVLidarData,
    TimeSyncedSceneFlowFrame,
)


@pytest.fixture
def av2_sequence_loader() -> ArgoverseSceneFlowSequenceLoader:
    return ArgoverseSceneFlowSequenceLoader(
        raw_data_path=Path("/tmp/argoverse2_small/val"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        with_rgb=True,
        expected_camera_shape=(194, 256, 3),
    )


@pytest.fixture
def av2_dataset_seq_2_causal() -> Argoverse2CausalSceneFlow:
    return Argoverse2CausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=2,
        expected_camera_shape=(194, 256, 3),
    )


@pytest.fixture
def av2_dataset_seq_5_causal() -> Argoverse2CausalSceneFlow:
    return Argoverse2CausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=5,
        expected_camera_shape=(194, 256, 3),
    )


@pytest.fixture
def av2_dataset_seq_2_noncausal() -> Argoverse2NonCausalSceneFlow:
    return Argoverse2NonCausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=2,
        expected_camera_shape=(194, 256, 3),
    )


@pytest.fixture
def av2_dataset_seq_5_noncausal() -> Argoverse2NonCausalSceneFlow:
    return Argoverse2NonCausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=5,
        expected_camera_shape=(194, 256, 3),
    )


def test_load_full_sequence_size_causal(av2_sequence_loader: ArgoverseSceneFlowSequenceLoader):
    sequence = av2_sequence_loader.load_sequence("02678d04-cc9f-3148-9f95-1ba66347dff9")
    assert len(sequence) == 157, f"expected 157 frames, got {len(sequence)}"
    sequence = av2_sequence_loader.load_sequence("02a00399-3857-444e-8db3-a8f58489c394")
    assert len(sequence) == 159, f"expected 159 frames, got {len(sequence)}"


def test_av2_dataset_seq_2_size_noncausal(
    av2_dataset_seq_2_noncausal: Argoverse2NonCausalSceneFlow,
):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = (157 - 1) // 2 + (159 - 1) // 2
    assert (
        len(av2_dataset_seq_2_noncausal) == expected_len
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_2_noncausal)}"

    for frame_list in av2_dataset_seq_2_noncausal:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 2, f"expected 2 entries, got {len(frame_list)}"


def test_av2_dataset_seq_5_size_noncausal(
    av2_dataset_seq_5_noncausal: Argoverse2NonCausalSceneFlow,
):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = (157 - 1) // 5 + (159 - 1) // 5
    assert (
        len(av2_dataset_seq_5_noncausal) == expected_len
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_5_noncausal)}"

    for frame_list in av2_dataset_seq_5_noncausal:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 5, f"expected 5 entries, got {len(frame_list)}"


def test_av2_dataset_seq_2_size_causal(av2_dataset_seq_2_causal: Argoverse2CausalSceneFlow):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = 157 - 1 + 159 - 1
    assert (
        len(av2_dataset_seq_2_causal) == expected_len
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_2_causal)}"

    for frame_list in av2_dataset_seq_2_causal:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 2, f"expected 2 entries, got {len(frame_list)}"


def test_av2_dataset_seq_5_size_causal(av2_dataset_seq_5_causal: Argoverse2CausalSceneFlow):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = 157 - 4 + 159 - 4
    assert (
        len(av2_dataset_seq_5_causal) == 157 - 4 + 159 - 4
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_5_causal)}"

    for frame_list in av2_dataset_seq_5_causal:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 5, f"expected 5 entries, got {len(frame_list)}"
