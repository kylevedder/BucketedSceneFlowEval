import numpy as np
import pytest

from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow, construct_dataset
from bucketed_scene_flow_eval.datastructures import GroundTruthPointFlow


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


def _run_eval_on_target_and_gt_datasets(
    gt_dataset: Argoverse2SceneFlow,
    target_dataset: Argoverse2SceneFlow,
    EXPECTED_RESULTS_DICT: dict[str, tuple[float, float]],
):
    assert len(gt_dataset) == len(target_dataset), (
        f"Ground truth and pseudo datasets must have the same number of samples. "
        f"Found {len(gt_dataset)} ground truth samples and "
        f"{len(target_dataset)} pseudo samples."
    )

    evaluator = gt_dataset.evaluator()

    # Iterate over both datasets, treating the pseudo dataset as the "prediction"
    # and the ground truth dataset as the "target"
    for (_, est_gt), (_, target_gt) in zip(target_dataset, gt_dataset):
        est_gt: GroundTruthPointFlow
        target_gt: GroundTruthPointFlow
        assert all(est_gt.trajectory_timestamps == target_gt.trajectory_timestamps), (
            f"Timestamps must match between the ground truth and pseudo datasets. "
            f"Found {est_gt.trajectory_timestamps} and {target_gt.trajectory_timestamps}."
        )
        assert (
            len(target_gt.trajectory_timestamps) == 2
        ), f"Timestamps must be a pair of timestamps. Found {target_gt.trajectory_timestamps}."

        evaluator.eval(est_gt, target_gt, target_gt.trajectory_timestamps[0])

    out_results_dict: dict[str, tuple[float, float]] = evaluator.compute_results()

    # Ensure that output results are a dictionary of the expected type
    assert isinstance(
        out_results_dict, dict
    ), f"Results must be a dictionary. Found {out_results_dict}."

    # Check overlap of keys
    assert out_results_dict.keys() == EXPECTED_RESULTS_DICT.keys(), (
        f"Results must be computed for the same classes. "
        f"Found {out_results_dict.keys()} and {EXPECTED_RESULTS_DICT.keys()}."
    )

    for key in EXPECTED_RESULTS_DICT:
        out_static_epe, out_dynamic_epe = out_results_dict[key]
        exp_static_epe, exp_dynamic_epe = EXPECTED_RESULTS_DICT[key]

        # Check that floats are equal, but be aware of NaNs (which are not equal to anything)
        assert np.isnan(out_static_epe) == np.isnan(
            exp_static_epe
        ), f"Static EPEs must both be NaN or not NaN. Found {out_static_epe} and {exp_static_epe}."

        assert np.isnan(out_dynamic_epe) == np.isnan(
            exp_dynamic_epe
        ), f"Dynamic EPEs must both be NaN or not NaN. Found {out_dynamic_epe} and {exp_dynamic_epe}."

        if not np.isnan(exp_static_epe):
            assert out_static_epe == pytest.approx(
                exp_static_epe, rel=1e-6
            ), f"Static EPEs must be equal. Found {out_static_epe} and {exp_static_epe}."
        if not np.isnan(exp_dynamic_epe):
            assert out_dynamic_epe == pytest.approx(
                exp_dynamic_epe, rel=1e-6
            ), f"Dynamic EPEs must be equal. Found {out_dynamic_epe} and {exp_dynamic_epe}."


def test_bucketed_eval_av2_with_ground(
    argo_dataset_gt_with_ground: Argoverse2SceneFlow,
    argo_dataset_pseudo_with_ground: Argoverse2SceneFlow,
):
    EXPECTED_RESULTS_DICT = {
        "BACKGROUND": (0.017420833175797096, float("nan")),
        "CAR": (0.00715087565425712, 0.9549859068245323),
        "OTHER_VEHICLES": (float("nan"), float("nan")),
        "PEDESTRIAN": (float("nan"), 0.8860363751576089),
        "WHEELED_VRU": (float("nan"), 0.9848588530322282),
    }
    _run_eval_on_target_and_gt_datasets(
        argo_dataset_gt_with_ground, argo_dataset_pseudo_with_ground, EXPECTED_RESULTS_DICT
    )


def test_bucketed_eval_av2_no_ground(
    argo_dataset_gt_no_ground: Argoverse2SceneFlow,
    argo_dataset_pseudo_no_ground: Argoverse2SceneFlow,
):
    EXPECTED_RESULTS_DICT = {
        "BACKGROUND": (0.01975785995262935, float("nan")),
        "CAR": (0.008681314962881582, 0.9460171305709397),
        "OTHER_VEHICLES": (float("nan"), float("nan")),
        "PEDESTRIAN": (float("nan"), 0.8834896978129233),
        "WHEELED_VRU": (float("nan"), 0.9758072524985107),
    }
    _run_eval_on_target_and_gt_datasets(
        argo_dataset_gt_no_ground, argo_dataset_pseudo_no_ground, EXPECTED_RESULTS_DICT
    )
