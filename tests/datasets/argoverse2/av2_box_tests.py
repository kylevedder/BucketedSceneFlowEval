from pathlib import Path

import pytest

from bucketed_scene_flow_eval.datasets.argoverse2 import (
    ArgoverseBoxAnnotationSequenceLoader,
    ArgoverseSceneFlowSequenceLoader,
)


@pytest.fixture
def av2_box_sequence_loader() -> ArgoverseBoxAnnotationSequenceLoader:
    return ArgoverseBoxAnnotationSequenceLoader(
        raw_data_path=Path("/tmp/argoverse2_small/val"),
    )


def test_load_box_sequence_length(
    av2_box_sequence_loader: ArgoverseBoxAnnotationSequenceLoader,
):
    sequence = av2_box_sequence_loader.load_sequence("02678d04-cc9f-3148-9f95-1ba66347dff9")
    assert len(sequence) == 157, f"expected 157 frames, got {len(sequence)}"
    first_frame, lidar_data = sequence.load(0, 0)
    assert len(first_frame.boxes) == 23, f"expected 23 boxes, got {len(first_frame.boxes)}"

    sequence = av2_box_sequence_loader.load_sequence("02a00399-3857-444e-8db3-a8f58489c394")
    assert len(sequence) == 159, f"expected 159 frames, got {len(sequence)}"
    first_frame, lidar_data = sequence.load(0, 0)
    assert len(first_frame.boxes) == 10, f"expected 10 boxes, got {len(first_frame.boxes)}"
