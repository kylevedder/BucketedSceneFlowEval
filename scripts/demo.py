import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    GroundTruthPointFlow,
    O3DVisualizer,
    PointCloud,
    PointCloudFrame,
    QuerySceneSequence,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
)


def color_by_distance(distances: np.ndarray, max_distance: float = 10.0, cmap: str = "viridis"):
    # Use distance to color points, normalized to [0, 1].
    colors = distances.copy()

    # Normalize to [0, 1]
    colors = colors / max_distance
    colors[colors > 1] = 1.0

    colormap = plt.get_cmap(cmap)
    colors = colormap(colors)[:, :3]
    return colors


def process_lidar_only(o3d_vis: O3DVisualizer, pc_frame: PointCloudFrame):
    print(f"Adding Lidar pointcloud with {len(pc_frame.global_pc)} points")
    o3d_vis.add_pointcloud(pc_frame.global_pc)
    o3d_vis.run()


def project_lidar_into_rgb(
    pc_frame: PointCloudFrame, rgb_frame: RGBFrame, reduction_factor: int = 4
) -> RGBImage:
    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.inverse().compose(
        rgb_frame.pose.sensor_to_ego
    )
    cam_frame_pc = pc_frame.full_pc.transform(pc_into_cam_frame_se3)
    cam_frame_pc = PointCloud(cam_frame_pc.points[cam_frame_pc.points[:, 0] >= 0])

    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)
    projected_points = projected_points.astype(np.int32)

    # Use distance to color points, normalized to [0, 1].
    colors = color_by_distance(cam_frame_pc.points[:, 0], max_distance=30)
    valid_points_mask = (
        (projected_points[:, 0] >= 0)
        & (projected_points[:, 0] < rgb_frame.rgb.image.shape[1])
        & (projected_points[:, 1] >= 0)
        & (projected_points[:, 1] < rgb_frame.rgb.image.shape[0])
    )
    projected_points = projected_points[valid_points_mask]
    colors = colors[valid_points_mask]

    scaled_rgb = rgb_frame.rgb.rescale(reduction_factor)
    scaled_projected_points = projected_points // reduction_factor

    projected_rgb_image = scaled_rgb.image
    projected_rgb_image[scaled_projected_points[:, 1], scaled_projected_points[:, 0], :] = colors
    return RGBImage(projected_rgb_image)


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


def visualize_rgb(frame_idx: int, query: QuerySceneSequence, save_dir: Optional[Path] = None):
    # The query specifies the raw scene and query points at a particular timestamp
    # These query points can be thought of as the specification of the valid points for
    # scene flow in the pointcloud at `t` for prediction to timestamp `t+1`

    scene_timestamp = query.query_particles.query_init_timestamp

    # The scene contains RGB image and pointcloud data for each timestamp.
    # These are stored as "frames" with pose and intrinsics information.
    # This enables the raw percepts to be projected into desired coordinate frames across time.
    rgb_frames = query.scene_sequence[scene_timestamp].rgb_frames
    pc_frame = query.scene_sequence[scene_timestamp].pc_frame

    items = rgb_frames.items()
    for plot_idx, (name, rgb_frame) in enumerate(items):
        plt.subplot(1, len(items), plot_idx + 1)
        plt.imshow(project_lidar_into_rgb(pc_frame, rgb_frame).image)
        # Disable axis ticks
        plt.xticks([])
        plt.yticks([])
        # Set padding between subplots to 0
        plt.tight_layout(pad=0)
        # Get rid of black border
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Get rid of white space
        plt.margins(0)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if save_dir is None:
        plt.show()
    else:
        save_location = save_dir / f"{query.scene_sequence.log_id}" / f"{frame_idx:010d}.png"
        save_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_location, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.clf()


if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Argoverse2SceneFlow")
    parser.add_argument("--root_dir", type=str, default="/efs/argoverse2/val")
    parser.add_argument("--skip_rgb", action="store_true")
    parser.add_argument("--mode", choices=["lidar", "rgb"], default="lidar")
    parser.add_argument("--save_dir", type=Path, default=None)
    args = parser.parse_args()

    dataset = construct_dataset(
        args.dataset, dict(root_dir=args.root_dir, with_rgb=not args.skip_rgb)
    )

    print("Dataset contains", len(dataset), "samples")

    for idx, (query, gt) in enumerate(dataset):
        if args.mode == "rgb":
            visualize_rgb(idx, query, args.save_dir)
        else:
            visualize_lidar_3d(query)
