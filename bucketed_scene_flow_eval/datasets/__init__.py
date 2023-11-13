from typing import Dict
from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2SceneFlow
from bucketed_scene_flow_eval.datasets.waymoopen import WaymoOpenSceneFlow

__all__ = ["Argoverse2SceneFlow", "WaymoOpenSceneFlow"]
dataset_names = [cls.lower() for cls in __all__]


def construct_dataset(name: str, args: dict):
    name = name.lower()
    all_lookup: Dict[str, str] = {cls.lower(): cls for cls in __all__}
    if name not in all_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls_name = all_lookup[name]
    # Convert cls_name string to class object using getattr
    print("Importing: ", __import__(__name__), cls_name)
    cls = getattr(__import__(__name__), cls_name)
    return cls(**args)
