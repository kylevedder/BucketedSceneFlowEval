from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2SceneFlow
from bucketed_scene_flow_eval.datasets.waymoopen import WaymoOpenSceneFlow

importable_classes = [Argoverse2SceneFlow, WaymoOpenSceneFlow]
name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_classes}


def construct_dataset(name: str, args: dict):
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)
