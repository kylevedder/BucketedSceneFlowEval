import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import O3DVisualizer, QuerySceneSequence


def visualize_lidar_3d(query: QuerySceneSequence):
    scene_timestamp = query.query_particles.query_init_timestamp

    rgb_frames = query.scene_sequence[scene_timestamp].rgb_frames
    pc_frame = query.scene_sequence[scene_timestamp].pc_frame

    o3d_vis = O3DVisualizer()
    o3d_vis.add_pointcloud(pc_frame.global_pc)
    for rgb_frame in rgb_frames.values():
        image_plane_pc, colors = rgb_frame.camera_projection.image_to_image_plane_pc(
            rgb_frame.rgb, depth=20
        )
        image_plane_pc = image_plane_pc.transform(rgb_frame.pose.sensor_to_ego.inverse())
        o3d_vis.add_pointcloud(image_plane_pc, color=colors)
    o3d_vis.run()
    del o3d_vis


if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Argoverse2SceneFlow")
    parser.add_argument("--root_dir", type=str, default="/efs/argoverse2/val")
    parser.add_argument("--skip_rgb", action="store_true")
    args = parser.parse_args()

    dataset = construct_dataset(
        args.dataset, dict(root_dir=args.root_dir, with_rgb=not args.skip_rgb)
    )

    print("Dataset contains", len(dataset), "samples")

    for idx, (query, gt) in enumerate(dataset):
        visualize_lidar_3d(query)
