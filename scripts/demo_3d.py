import argparse
from pathlib import Path

import numpy as np
import tqdm

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    O3DVisualizer,
    TimeSyncedSceneFlowFrame,
)


def visualize_lidar_3d(frame_list: list[TimeSyncedSceneFlowFrame], downscale_rgb_factor: int):
    o3d_vis = O3DVisualizer()

    print("Visualizing", len(frame_list), "frames")

    for frame_idx, frame in enumerate(frame_list):
        rgb_frames = frame.rgbs
        pc_frame = frame.pc
        flow_frame = frame.flow

        # Set constant flow for debug
        # flow_frame.full_flow = np.ones_like(flow_frame.full_flow) * 0.1

        o3d_vis.add_global_pc_frame(pc_frame)
        o3d_vis.add_global_flow(pc_frame, flow_frame)
        for name, rgb_frame in rgb_frames.items():
            print(f"Adding RGB frame {frame_idx} {name}")
            rgb_frame = rgb_frame.rescale(downscale_rgb_factor)
            # print("RGB Frame ego pose:", rgb_frame.pose.ego_to_global.translation)
            o3d_vis.add_pose(rgb_frame.pose.ego_to_global)
            o3d_vis.add_global_rgb_frame(rgb_frame)
    o3d_vis.run()
    del o3d_vis


if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Argoverse2CausalSceneFlow")
    parser.add_argument("--root_dir", type=Path, default="/efs/argoverse2/val")
    parser.add_argument("--flow_dir", type=Path, default="/efs/argoverse2/val_sceneflow_feather/")
    parser.add_argument("--with_rgb", action="store_true")
    parser.add_argument("--no_ground", action="store_true")
    parser.add_argument("--sequence_length", type=int, default=2)
    parser.add_argument("--downscale_rgb_factor", type=int, default=8)
    args = parser.parse_args()

    dataset = construct_dataset(
        args.dataset,
        dict(
            root_dir=args.root_dir,
            flow_data_path=args.flow_dir,
            with_rgb=args.with_rgb,
            subsequence_length=args.sequence_length,
            use_gt_flow=False,
            with_ground=not args.no_ground,
        ),
    )
    assert len(dataset) > 0, "Dataset is empty"
    print("Dataset contains", len(dataset), "samples")

    vis_index = 0

    print("Loading sequence idx", vis_index)
    frame_list = dataset[vis_index]
    visualize_lidar_3d(frame_list, args.downscale_rgb_factor)
