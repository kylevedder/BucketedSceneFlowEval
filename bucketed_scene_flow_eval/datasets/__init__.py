from bucketed_scene_flow_eval.datasets.argoverse2 import (
    Argoverse2CausalSceneFlow,
    Argoverse2NonCausalSceneFlow,
)
from bucketed_scene_flow_eval.datasets.orbbec_astra import OrbbecAstra
from bucketed_scene_flow_eval.datasets.waymoopen import (
    WaymoOpenCausalSceneFlow,
    WaymoOpenNonCausalSceneFlow,
)

# from bucketed_scene_flow_eval.datasets.nuscenes import (
#     NuScenesCausalSceneFlow,
#     NuScenesNonCausalSceneFlow,
# )
from bucketed_scene_flow_eval.interfaces import AbstractDataset

importable_classes = [
    Argoverse2CausalSceneFlow,
    Argoverse2NonCausalSceneFlow,
    # NuScenesCausalSceneFlow,
    # NuScenesNonCausalSceneFlow,
    WaymoOpenCausalSceneFlow,
    WaymoOpenNonCausalSceneFlow,
    OrbbecAstra,
]
name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_classes}


def construct_dataset(name: str, args: dict) -> AbstractDataset:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)
