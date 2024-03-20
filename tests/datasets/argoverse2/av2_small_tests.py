from pathlib import Path

import numpy as np
import pytest

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
def av2_loader() -> ArgoverseSceneFlowSequenceLoader:
    return ArgoverseSceneFlowSequenceLoader(
        raw_data_path=Path("/tmp/argoverse2_small/val"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        with_rgb=True,
    )
