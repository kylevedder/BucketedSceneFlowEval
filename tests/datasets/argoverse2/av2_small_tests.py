from pathlib import Path

import numpy as np
import pytest

from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow
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
def av2_dataset_seq_2() -> Argoverse2SceneFlow:
    return Argoverse2SceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=2,
        expected_camera_shape=(194, 256, 3),
    )


@pytest.fixture
def av2_dataset_seq_5() -> Argoverse2SceneFlow:
    return Argoverse2SceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val"),
        with_rgb=True,
        use_gt_flow=True,
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=5,
        expected_camera_shape=(194, 256, 3),
    )


def test_load_full_sequence_size(av2_sequence_loader: ArgoverseSceneFlowSequenceLoader):
    sequence = av2_sequence_loader.load_sequence("02678d04-cc9f-3148-9f95-1ba66347dff9")
    assert len(sequence) == 157, f"expected 157 frames, got {len(sequence)}"
    sequence = av2_sequence_loader.load_sequence("02a00399-3857-444e-8db3-a8f58489c394")
    assert len(sequence) == 159, f"expected 159 frames, got {len(sequence)}"


def test_av2_dataset_seq_2_size(av2_dataset_seq_2: Argoverse2SceneFlow):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = 157 - 1 + 159 - 1
    assert (
        len(av2_dataset_seq_2) == expected_len
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_2)}"

    for frame_list in av2_dataset_seq_2:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 2, f"expected 2 entries, got {len(frame_list)}"


def test_av2_dataset_seq_5_size(av2_dataset_seq_5: Argoverse2SceneFlow):
    # Length of the two subsequences, minus 1 because of flow between frames.
    expected_len = 157 - 4 + 159 - 4
    assert (
        len(av2_dataset_seq_5) == 157 - 4 + 159 - 4
    ), f"expected {expected_len} frames, got {len(av2_dataset_seq_5)}"

    for frame_list in av2_dataset_seq_5:
        assert isinstance(frame_list, list)
        assert len(frame_list) == 5, f"expected 5 entries, got {len(frame_list)}"
