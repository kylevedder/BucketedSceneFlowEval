import bucketed_scene_flow_eval
from bucketed_scene_flow_eval.datastructures import PointCloud
from bucketed_scene_flow_eval.datasets import construct_dataset

argo_dataset = construct_dataset('argoverse2sceneflow', dict(root_dir='/tmp/argoverse2_tiny/val', with_rgb=False))


import numpy as np

pc = PointCloud(np.random.rand(100, 3))
print(pc)