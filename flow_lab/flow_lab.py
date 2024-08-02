import argparse
import shutil
from pathlib import Path

import open3d as o3d
from annotation_saver import AnnotationSaver
from vis_classes import *

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    PoseInfo,
    TimeSyncedSceneFlowBoxFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractSequence
from bucketed_scene_flow_eval.utils.glfw_key_ids import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Argoverse2NonCausalSceneFlow")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--sequence_id", type=str, required=True)
    parser.add_argument("--save_dir", type=Path, required=False)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--rolling_window_size", type=int, default=5)
    args = parser.parse_args()

    if args.preprocess:
        if not args.save_dir:
            raise ValueError("The --save_dir argument is required when --preprocess is specified.")
    else:
        if not args.save_dir:
            args.save_dir = args.root_dir

    return args


def load_box_frames(
    root_dir: Path, dataset_name: str, sequence_length, sequence_id: str
) -> list[TimeSyncedSceneFlowBoxFrame]:
    # input_path = root_dir / 'val'
    dataset = construct_dataset(
        name=dataset_name,
        args=dict(
            root_dir=root_dir,
            subsequence_length=sequence_length,
            with_ground=False,
            range_crop_type="ego",
            load_boxes=True,
            log_subset=[sequence_id],
        ),
    )
    assert len(dataset) == 1, f"Expected 1 sequence, got {len(dataset)}"
    return dataset[0]


def preprocess_box_frames(
    frames: list[TimeSyncedSceneFlowBoxFrame],
) -> list[TimeSyncedSceneFlowBoxFrame]:

    last_frame_poses = {}

    for frame in frames:
        for i, (box, pose, is_valid) in enumerate(
            zip(frame.boxes.full_boxes, frame.boxes.full_poses, frame.boxes.mask)
        ):
            if is_valid:
                if box.track_uuid in last_frame_poses:
                    # If the box reappears, keep the global pose from the last frame
                    last_global_se3 = last_frame_poses[box.track_uuid].sensor_to_global

                    frame.boxes.full_poses[i].sensor_to_ego = pose.ego_to_global.inverse().compose(
                        last_global_se3
                    )
                last_frame_poses[box.track_uuid] = frame.boxes.full_poses[i]

    return frames


def setup_visualizer(state_manager, annotation_saver, frames):
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.register_mouse_move_callback(state_manager.on_mouse_move)
    vis.register_mouse_scroll_callback(state_manager.on_mouse_scroll)
    vis.register_mouse_button_callback(state_manager.on_mouse_button)

    # Register key callbacks
    # Use WASD keys for translation, Q and E keys for yaw
    vis.register_key_callback(ord("W"), state_manager.forward_press)
    vis.register_key_callback(ord("S"), state_manager.backward_press)
    vis.register_key_callback(ord("A"), state_manager.left_press)
    vis.register_key_callback(ord("D"), state_manager.right_press)
    vis.register_key_callback(ord("Z"), state_manager.down_press)
    vis.register_key_callback(ord("X"), state_manager.up_press)
    vis.register_key_callback(ord("Q"), state_manager.yaw_clockwise_press)
    vis.register_key_callback(ord("E"), state_manager.yaw_counterclockwise_press)
    # Use arrow keys for pitch and roll
    vis.register_key_callback(GLFW_KEY_UP, state_manager.pitch_up_press)
    vis.register_key_callback(GLFW_KEY_DOWN, state_manager.pitch_down_press)
    vis.register_key_callback(GLFW_KEY_RIGHT, state_manager.roll_clockwise_press)
    vis.register_key_callback(GLFW_KEY_LEFT, state_manager.roll_counterclockwise_press)
    # Use , and . keys for going forward and backward through the frames
    vis.register_key_callback(ord(","), lambda vis: state_manager.backward_frame_press(vis))
    vis.register_key_callback(ord("."), lambda vis: state_manager.forward_frame_press(vis))
    # Press Shift to fine tune
    vis.register_key_action_callback(GLFW_KEY_LEFT_SHIFT, state_manager.shift_actions)
    # Use the 'Ctrl+X' to save
    vis.register_key_action_callback(
        ord("X"),
        lambda vis, action, mods: annotation_saver.save_callback(vis, action, mods, frames),
    )

    vis.create_window()
    return vis


def main():
    args = parse_arguments()

    frames = load_box_frames(
        args.root_dir, args.dataset_name, args.sequence_length, args.sequence_id
    )

    annotation_saver = AnnotationSaver(args.save_dir / args.sequence_id)
    if args.preprocess:
        print("Preprocessing data...")
        frames = preprocess_box_frames(frames)
        annotation_saver.save(frames)

    state_manager = ViewStateManager(frames, args.rolling_window_size)
    vis = setup_visualizer(state_manager, annotation_saver, frames)
    state_manager.render_pc_and_boxes(vis, reset_bounding_box=True)

    # render_option = vis.get_render_option()
    # render_option.mesh_show_wireframe = True
    # render_option.light_on = False
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    vis.run()


if __name__ == "__main__":

    print("Customized visualization with mouse action.")
    main()
