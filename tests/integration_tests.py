import bucketed_scene_flow_eval
from bucketed_scene_flow_eval.datastructures import *
from bucketed_scene_flow_eval.datasets import construct_dataset


waymo_dataset_gt = construct_dataset(
    "waymoopensceneflow",
    dict(root_dir="/tmp/waymo_open_processed_flow_tiny/training"),
)

assert len(waymo_dataset_gt) == 1, f"Expected 1 scene, got {len(waymo_dataset_gt)}"


argo_dataset_gt = construct_dataset(
    "argoverse2sceneflow",
    dict(
        root_dir="/tmp/argoverse2_tiny/val",
        with_rgb=False,
        use_gt_flow=True,
    ),
)

assert len(argo_dataset_gt) == 1, f"Expected 1 scene, got {len(argo_dataset_gt)}"

argo_dataset_pseudo = construct_dataset(
    "argoverse2sceneflow",
    dict(
        root_dir="/tmp/argoverse2_tiny/val",
        with_rgb=False,
        use_gt_flow=False,
    ),
)

assert (
    len(argo_dataset_pseudo) == 1
), f"Expected 1 scene, got {len(argo_dataset_pseudo)}"


for query, gt in waymo_dataset_gt:
    assert isinstance(
        query, QuerySceneSequence
    ), f"Expected QuerySceneSequence, got {type(query)}"
    assert isinstance(
        gt, GroundTruthPointFlow
    ), f"Expected GroundTruthParticleTrajectories, got {type(gt)}"

for query, gt in argo_dataset_gt:
    assert isinstance(
        query, QuerySceneSequence
    ), f"Expected QuerySceneSequence, got {type(query)}"
    assert isinstance(
        gt, GroundTruthPointFlow
    ), f"Expected GroundTruthParticleTrajectories, got {type(gt)}"

for query, gt in argo_dataset_pseudo:
    assert isinstance(
        query, QuerySceneSequence
    ), f"Expected QuerySceneSequence, got {type(query)}"
    assert isinstance(
        gt, GroundTruthPointFlow
    ), f"Expected GroundTruthParticleTrajectories, got {type(gt)}"
