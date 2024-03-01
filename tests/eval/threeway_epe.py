import pytest

from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow, construct_dataset
from tests.eval.bucketed_epe import _run_eval_on_target_and_gt_datasets


@pytest.fixture
def argo_dataset_gt_with_ground():
    return construct_dataset(
        "argoverse2sceneflow",
        dict(
            root_dir="/tmp/argoverse2_tiny/val",
            with_rgb=False,
            use_gt_flow=True,
            with_ground=True,
            eval_type="threeway_epe",
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
            eval_type="threeway_epe",
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
            eval_type="threeway_epe",
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
            eval_type="threeway_epe",
        ),
    )


def test_bucketed_eval_av2_with_ground(
    argo_dataset_gt_with_ground: Argoverse2SceneFlow,
    argo_dataset_pseudo_with_ground: Argoverse2SceneFlow,
):
    EXPECTED_RESULTS_DICT = {
        "BACKGROUND": (0.017420833175797096, float("nan")),
        "FOREGROUND": (0.00715087565425712, 0.5442804620019708),
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
        "FOREGROUND": (0.008681314962881582, 0.5248476027085919),
    }
    _run_eval_on_target_and_gt_datasets(
        argo_dataset_gt_no_ground, argo_dataset_pseudo_no_ground, EXPECTED_RESULTS_DICT
    )
