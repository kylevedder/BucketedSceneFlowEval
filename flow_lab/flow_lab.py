import argparse
import json
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
from bucketed_scene_flow_eval.utils.glfw_key_ids import *


def parse_arguments():
    # Define the default path for the lookup table inside the flow_lab folder
    flow_lab_dir = Path(__file__).resolve().parent
    default_lookup_table_path = flow_lab_dir / "av2_small_sequence_length.json"

    parser = argparse.ArgumentParser(description="Scene flow data visualization and annotation.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Argoverse2NonCausalSceneFlow",
        help="Name of the dataset to use. Default is 'Argoverse2NonCausalSceneFlow'.",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Path to the root directory containing the dataset.",
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="The specific sequence name."
    )
    parser.add_argument(
        "--lookup_table",
        type=Path,
        default=default_lookup_table_path,
        help="Path to JSON lookup table for sequence lengths.",
    )
    # parser.add_argument(
    #     "--save_dir",
    #     type=Path,
    #     required=False,
    #     help="Directory where processed data will be saved. ",
    # )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether the data need to be preprocessed. If speciffied, it will save the processed data to save_dir",
    )
    parser.add_argument(
        "--rolling_window_size",
        type=int,
        default=5,
        help="Number of frames to display around the current frame.",
    )

    args = parser.parse_args()

    return args


def load_box_frames(
    root_dir: Path, dataset_name: str, sequence_length, sequence_id: str
) -> list[TimeSyncedSceneFlowBoxFrame]:
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
    vis.register_key_action_callback(ord("W"), state_manager.forward_press)
    vis.register_key_action_callback(ord("A"), state_manager.left_press)
    vis.register_key_action_callback(ord("D"), state_manager.right_press)
    vis.register_key_action_callback(ord("Z"), state_manager.down_press)
    vis.register_key_action_callback(ord("X"), state_manager.up_press)
    vis.register_key_action_callback(ord("Q"), state_manager.yaw_clockwise_press)
    vis.register_key_action_callback(ord("E"), state_manager.yaw_counterclockwise_press)
    # Use arrow keys for pitch and roll
    vis.register_key_action_callback(GLFW_KEY_UP, state_manager.pitch_up_press)
    vis.register_key_action_callback(GLFW_KEY_DOWN, state_manager.pitch_down_press)
    vis.register_key_action_callback(GLFW_KEY_RIGHT, state_manager.roll_clockwise_press)
    vis.register_key_action_callback(GLFW_KEY_LEFT, state_manager.roll_counterclockwise_press)
    # Use , and . keys for going forward and backward through the frames
    vis.register_key_callback(ord(","), state_manager.backward_frame_press)
    vis.register_key_callback(ord("."), state_manager.forward_frame_press)
    # Use 'V' to toggle postion adjustment mode, "L" to toggle z axis lock
    vis.register_key_callback(ord("V"), state_manager.toggle_propagate_with_velocity)
    vis.register_key_callback(ord("L"), state_manager.toggle_lock_z_axis)
    vis.register_key_callback(ord("T"), state_manager.toggle_trajectory_visibility)

    # Use the 'Ctrl+S' to save, 'S' to translate
    vis.register_key_action_callback(ord("S"), state_manager.key_S_actions)
    # Use 'Enter' to zoom in
    vis.register_key_action_callback(GLFW_KEY_ENTER, state_manager.zoom_press)
    # Use 'Space' to toggle box
    vis.register_key_action_callback(GLFW_KEY_SPACE, state_manager.toggle_box)

    initial_title = f"Frame: {state_manager.current_frame_index} | Mode: Position"
    vis.create_window(window_name=initial_title)
    # vis.create_window()
    # render_option = vis.get_render_option()
    # render_option.mesh_show_wireframe = True
    # render_option.light_on = False
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    return vis


def load_sequence_length(sequence_id: str, lookup_table: Path) -> int:
    with open(lookup_table, "r") as f:
        data = json.load(f)
    return data.get(sequence_id, 0)  # Default to 0 if the sequence_id is not found


def setup_save_path(root_dir: Path, sequence_id: str) -> Path:
    """
    Set up the path to the folder where processed annotations should be saved.
    """
    parent_dir = root_dir.parent
    processed_dir = parent_dir / f"{root_dir.name}_processed"
    sequence_save_dir = processed_dir / "val" / sequence_id

    return sequence_save_dir


def setup_load_path(root_dir: Path, sequence_id: str) -> Path:
    """
    Set up the path to load data when preprocess is required.
    """
    parent_dir = root_dir.parent
    processed_dir = parent_dir / f"{root_dir.name}_processed"
    sequence_save_dir = processed_dir / "val"

    return sequence_save_dir


def main():
    args = parse_arguments()

    # load scequence length
    sequence_length = load_sequence_length(args.sequence_id, args.lookup_table)
    if sequence_length == 0:
        raise ValueError(f"Sequence ID {args.sequence_id} not found in lookup table.")

    # this is used to save edited data
    output_path = setup_save_path(args.root_dir, args.sequence_id)
    annotation_saver = AnnotationSaver(output_path)

    # Check if preprocessing is required and load frames
    annotation_file_path = output_path / "annotations.feather"
    if args.preprocess or not annotation_file_path.exists():
        print("Preprocessing data...")
        input_path = args.root_dir / "val"
        frames = load_box_frames(input_path, args.dataset_name, sequence_length, args.sequence_id)
        frames = preprocess_box_frames(frames)
        annotation_saver.save(frames)
    else:
        input_path = setup_load_path(args.root_dir, args.sequence_id)
        frames = load_box_frames(input_path, args.dataset_name, sequence_length, args.sequence_id)

    # declare the view state manager and display the window
    state_manager = ViewStateManager(frames, annotation_saver, args.rolling_window_size)
    vis = setup_visualizer(state_manager, annotation_saver, frames)
    state_manager.render_pc_and_boxes(vis, reset_bounding_box=True)
    vis.run()


if __name__ == "__main__":

    print("Customized visualization with mouse action.")
    main()
