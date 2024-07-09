from pathlib import Path

import pytest

from bucketed_scene_flow_eval.datasets.nuscenes import NuScenesRawSequenceLoader


@pytest.fixture
def nuscenes_loader() -> NuScenesRawSequenceLoader:
    return NuScenesRawSequenceLoader(
        sequence_dir=Path("/tmp/nuscenes"),
        version="v1.0-mini",
        split="mini_train",
        verbose=False,
    )


def test_nuscenes_loader_basic_load_and_len_check(nuscenes_loader: NuScenesRawSequenceLoader):
    assert len(nuscenes_loader) > 0, f"no sequences found in {nuscenes_loader}"
    expected_lens= [236, 236, 236, 233, 223, 239, 231, 231]
    assert len(nuscenes_loader) == len(
        expected_lens
    ), f"expected {len(expected_lens)} sequences, got {len(nuscenes_loader)}"

    num_loop_iterations = 0
    for sequence_id, expected_len in zip(nuscenes_loader.get_sequence_ids(), expected_lens):
        num_loop_iterations += 1
        nusc_seq = nuscenes_loader.load_sequence(sequence_id)
        assert (
            len(nusc_seq) == expected_len
        ), f"expected {expected_len} frames, got {len(nusc_seq)} for {sequence_id}"

    assert num_loop_iterations == len(
        expected_lens
    ), f"expected {len(expected_lens)} loop iterations, got {num_loop_iterations}"
