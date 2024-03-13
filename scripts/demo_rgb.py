import argparse
import enum
import itertools
from collections import namedtuple
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
    PoseInfo,
    QuerySceneSequence,
    RGBFrame,
    RGBFrameLookup,
    RGBImage,
)


class Mode(enum.Enum):
    PROJECT_LIDAR = "project_lidar"
    PROJECT_FLOW = "project_flow"


DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, itertools.accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def _color_by_distance(distances: np.ndarray, max_distance: float = 10.0, cmap: str = "viridis"):
    # Use distance to color points, normalized to [0, 1].
    colors = distances.copy()

    # Normalize to [0, 1]
    colors = colors / max_distance
    colors[colors > 1] = 1.0

    colormap = plt.get_cmap(cmap)
    colors = colormap(colors)[:, :3]
    return colors


def _flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float] = 2.0,
    background: Optional[str] = "bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )
    wheel = _make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional)
        + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs", ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"]
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - np.expand_dims(factors, -1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float32)
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float32)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def _insert_into_image(
    rgb_image: RGBImage, projected_points: np.ndarray, colors: np.ndarray, reduction_factor: int
) -> RGBImage:
    # Suppress RuntimeWarning: invalid value encountered in cast
    with np.errstate(invalid="ignore"):
        projected_points = projected_points.astype(np.int32)

    valid_points_mask = (
        (projected_points[:, 0] >= 0)
        & (projected_points[:, 0] < rgb_image.shape[1])
        & (projected_points[:, 1] >= 0)
        & (projected_points[:, 1] < rgb_image.shape[0])
    )

    projected_points = projected_points[valid_points_mask]
    colors = colors[valid_points_mask]

    scaled_rgb = rgb_image.rescale(reduction_factor)
    scaled_projected_points = projected_points // reduction_factor
    projected_rgb_image = scaled_rgb.image
    projected_rgb_image[scaled_projected_points[:, 1], scaled_projected_points[:, 0], :] = colors
    return RGBImage(projected_rgb_image)


def project_lidar_into_rgb(
    pc_frame: PointCloudFrame, rgb_frame: RGBFrame, reduction_factor: int
) -> RGBImage:
    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.inverse().compose(
        rgb_frame.pose.sensor_to_ego
    )
    cam_frame_pc = pc_frame.full_pc.transform(pc_into_cam_frame_se3)
    cam_frame_pc = PointCloud(cam_frame_pc.points[cam_frame_pc.points[:, 0] >= 0])

    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)

    # Use distance to color points, normalized to [0, 1].
    colors = _color_by_distance(cam_frame_pc.points[:, 0], max_distance=30)
    return _insert_into_image(rgb_frame.rgb, projected_points, colors, reduction_factor)


def project_flow_into_rgb(
    pc_frame: PointCloudFrame,
    flowed_pc_frame: PointCloudFrame,
    rgb_frame: RGBFrame,
    color_pose: PoseInfo,  # Pose used to compute color of flow
    reduction_factor: int,
) -> RGBImage:
    assert len(pc_frame.full_pc) == len(
        flowed_pc_frame.full_pc
    ), f"Pointclouds must be the same size, got {len(pc_frame.full_pc)} and {len(flowed_pc_frame.full_pc)}"

    # Ensure that all valid flowed_pc_frame points are valid in pc_frame
    assert np.all(
        pc_frame.mask & flowed_pc_frame.mask == flowed_pc_frame.mask
    ), f"Flow mask must be subset of pc mask but it's not"

    # Set the pc_frame mask to be the same as the flowed_pc_frame mask
    pc_frame.mask = flowed_pc_frame.mask

    assert len(pc_frame.pc) == len(
        flowed_pc_frame.pc
    ), f"Pointclouds must be the same size, got {len(pc_frame.pc)} and {len(flowed_pc_frame.pc)}"

    assert (
        pc_frame.pose == flowed_pc_frame.pose
    ), f"Poses must be the same, got {pc_frame.pose} and {flowed_pc_frame.pose}"

    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.inverse().compose(
        rgb_frame.pose.sensor_to_ego
    )

    cam_frame_pc = pc_frame.pc.transform(pc_into_cam_frame_se3)
    cam_frame_flowed_pc = flowed_pc_frame.pc.transform(pc_into_cam_frame_se3)

    in_front_of_cam_mask = cam_frame_pc.points[:, 0] >= 0

    # Don't use points behind the camera to describe flow.
    cam_frame_pc = PointCloud(cam_frame_pc.points[in_front_of_cam_mask])
    cam_frame_flowed_pc = PointCloud(cam_frame_flowed_pc.points[in_front_of_cam_mask])

    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)

    # Convert the cam_frame_pc and cam_frame_flowed_pc to the color_pose frame
    cam_frame_to_color_pose_se3 = rgb_frame.pose.sensor_to_ego.inverse().compose(
        color_pose.sensor_to_ego
    )
    cam_frame_pc = cam_frame_pc.transform(cam_frame_to_color_pose_se3)
    cam_frame_flowed_pc = cam_frame_flowed_pc.transform(cam_frame_to_color_pose_se3)
    flow_vectors = cam_frame_flowed_pc.points - cam_frame_pc.points

    flow_colors = _flow_to_rgb(flow_vectors) / 255.0
    blank_image = RGBImage.white_image_like(rgb_frame.rgb)
    return _insert_into_image(blank_image, projected_points, flow_colors, reduction_factor)


def visualize(
    frame_idx: int,
    query: QuerySceneSequence,
    gt: GroundTruthPointFlow,
    save_dir: Path,
    mode: Mode,
    reduction_factor: int,
):
    # The query specifies the raw scene and query points at a particular timestamp
    # These query points can be thought of as the specification of the valid points for
    # scene flow in the pointcloud at `t` for prediction to timestamp `t+1`

    scene_timestamp = query.query_particles.query_init_timestamp
    src_timestamp, target_timestamp = gt.trajectory_timestamps
    assert scene_timestamp == src_timestamp, f"Scene timestamp {scene_timestamp} != {src_timestamp}"

    # The scene contains RGB image and pointcloud data for each timestamp.
    # These are stored as "frames" with pose and intrinsics information.
    # This enables the raw percepts to be projected into desired coordinate frames across time.
    rgb_frames = query.scene_sequence[scene_timestamp].rgb_frames
    pc_frame = query.scene_sequence[scene_timestamp].pc_frame
    global_frame_flow, is_valid_flow_mask = gt.get_flow(src_timestamp, target_timestamp)
    assert len(pc_frame.full_pc) == len(gt.is_valid_flow), (
        f"Pointcloud and flow must be the same size, got {len(pc_frame.full_pc)} and "
        f"{len(gt.is_valid_flow)}"
    )
    flowed_pc_frame = pc_frame.add_global_flow(global_frame_flow, is_valid_flow_mask)

    if mode == Mode.PROJECT_LIDAR:
        rgb_images = [
            project_lidar_into_rgb(pc_frame, rgb_frame, reduction_factor)
            for rgb_frame in rgb_frames.values()
        ]
    elif mode == Mode.PROJECT_FLOW:
        middle_frame = rgb_frames.values()[len(rgb_frames) // 2]
        rgb_images = [
            project_flow_into_rgb(
                pc_frame, flowed_pc_frame, rgb_frame, middle_frame.pose, reduction_factor
            )
            for rgb_frame in rgb_frames.values()
        ]

    for plot_idx, rgb_image in enumerate(rgb_images):
        plt.subplot(1, len(rgb_images), plot_idx + 1)
        plt.imshow(rgb_image.image)
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
        # Set the background to be black
    fig = plt.gcf()
    fig.set_facecolor("black")
    save_location = save_dir / f"{query.scene_sequence.log_id}" / f"{frame_idx:010d}.png"
    save_location.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_location, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.clf()


if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Argoverse2SceneFlow")
    parser.add_argument("--root_dir", type=Path, default="/efs/argoverse2/val")
    parser.add_argument("--flow_dir", type=Path, default="/efs/argoverse2/val_sceneflow_feather")
    # use modes from the Mode enum
    parser.add_argument(
        "--mode", type=str, choices=[mode.value for mode in Mode], default=Mode.PROJECT_LIDAR.value
    )
    parser.add_argument("--save_dir", type=Path, default=Path("./vis_save_dir/"))
    parser.add_argument("--reduction_factor", type=int, default=4)
    args = parser.parse_args()

    dataset = construct_dataset(
        args.dataset,
        dict(
            root_dir=args.root_dir, with_rgb=True, flow_data_path=args.flow_dir, use_gt_flow=False
        ),
    )

    print("Dataset contains", len(dataset), "samples")
    mode = Mode(args.mode)

    save_dir = (
        args.save_dir
        / f"{args.dataset}"
        / f"{args.root_dir.stem}"
        / f"{mode.value}"
        / f"{args.flow_dir.stem}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, (query, gt) in enumerate(dataset):
        visualize(idx, query, gt, save_dir, mode, args.reduction_factor)
