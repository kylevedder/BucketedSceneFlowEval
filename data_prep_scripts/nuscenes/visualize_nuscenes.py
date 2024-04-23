from bucketed_scene_flow_eval.datasets.nuscenes import NuScenesRawSequenceLoader
import numpy as np
import open3d as o3d
from pathlib import Path

from bucketed_scene_flow_eval.datasets.nuscenes.nuscenes_metacategories import BUCKETED_METACATAGORIES
from bucketed_scene_flow_eval.utils.loaders import load_feather

raw_sequence_loader = NuScenesRawSequenceLoader(version='v1.0-mini', sequence_dir="/efs/nuscenes_mini")
sequence = raw_sequence_loader[0]

starter_idx = 0
timestamps = range(len(sequence))

def increase_starter_idx(vis):
    global starter_idx
    starter_idx += 1
    if starter_idx >= len(timestamps) - 1:
        starter_idx = 0
    # print("Index: ", starter_idx)
    vis.clear_geometries()
    draw_frames(vis, reset_view=False)


def decrease_starter_idx(vis):
    global starter_idx
    starter_idx -= 1
    if starter_idx < 0:
        starter_idx = len(timestamps) - 2
    # print("Index: ", starter_idx)
    vis.clear_geometries()
    draw_frames(vis, reset_view=False)


def setup_vis():
    # # make open3d visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().background_color = (0.1, 0.1, 0.1)
    # vis.get_render_option().show_coordinate_frame = True
    # set up vector
    vis.get_view_control().set_up([0, 0, 1])
    # left arrow decrease starter_idx
    vis.register_key_callback(263, decrease_starter_idx)
    # right arrow increase starter_idx
    vis.register_key_callback(262, increase_starter_idx)

    return vis

def _colorize_pc(pc: o3d.geometry.PointCloud, color_tuple: tuple[float, float, float]):
    pc_color = np.ones_like(pc.points) * np.array(color_tuple)
    return o3d.utility.Vector3dVector(pc_color)

def draw_frames_cuboids(vis, reset_view=False):
    ts = timestamps[starter_idx]
    pc_object = sequence.load(ts, 0)[0].pc
    lidar_pc = pc_object.full_ego_pc
    pose = pc_object.pose.ego_to_global
    lidar_sensor_token = sequence.synced_sensors[ts].lidar_top['token']
    cuboids = sequence.nusc.get_boxes_with_instance_token(lidar_sensor_token)
    # Add base point cloud
    pcd = lidar_pc.to_o3d()
    pcd.colors = _colorize_pc(pcd, (1, 1, 1))
    vis.add_geometry(pcd, reset_bounding_box=reset_view)
    # # Draw the cuboids
    cuboids = [c.transform(pose.inverse()) for c in cuboids]
    draw_cuboids(vis, cuboids)

def draw_frames(vis, reset_view=False):
    ts = timestamps[starter_idx:starter_idx+2]
    color_list = [(0, 0, 1), (0, 1, 0)]
    pc_objects = [sequence.load(t, 0)[0].pc for t in ts]

    lidar_pc = [pc_obj.full_ego_pc for pc_obj in pc_objects]

    groundish_points_mask = lidar_pc[0].within_region_mask(-1000, 1000, -1000, 1000, -2, 0.5)
    groundish_points = lidar_pc[0].mask_points(groundish_points_mask)
    actual_ground_from_groundish_mask = np.zeros((groundish_points.shape[0]), dtype=bool)

    o3d.utility.random.seed(42)
    _, inliers0 = groundish_points.to_o3d().segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100, probability=1.0)
    actual_ground_from_groundish_mask[inliers0] = 1

    ground_mask = np.zeros_like(groundish_points_mask, dtype=bool)
    ground_mask[groundish_points_mask] = actual_ground_from_groundish_mask

    final_rendered_pc = lidar_pc[0].to_o3d()
    final_pc_color = np.tile([0, 0, 1], (ground_mask.shape[0], 1))
    final_pc_color[ground_mask] = [0, 1, 1]
    final_rendered_pc.colors = o3d.utility.Vector3dVector(final_pc_color)

    vis.add_geometry(final_rendered_pc, reset_bounding_box=reset_view)

    # flow_data = load_feather(Path(f"/efs/nuscenes_mini_sceneflow_feather/{sequence.log_id}/{ts[0]}.feather"))
    flow_data = load_feather(Path(f"/efs/nuscenes_mini_nsfp_flow/sequence_len_002/{sequence.log_id}/{ts[0]:010d}.feather"))
    # flow_data = load_feather(Path(f"/efs/nuscenes_mini_fast_nsf_flow/sequence_len_002/{sequence.log_id}/{ts[0]:010d}.feather"))
    # is_valid_arr = flow_data["is_valid"].values

    # The flow data is stored as 3 1D arrays, one for each dimension.
    xs = flow_data["flow_tx_m"].values
    ys = flow_data["flow_ty_m"].values
    zs = flow_data["flow_tz_m"].values
    flow_0_1 = np.stack([xs, ys, zs], axis=1)
    # ego_frame_flow = flow_0_1
    ego_frame_flow = pc_objects[0].pose.sensor_to_ego.transform_flow(flow_0_1)

    pc = lidar_pc[0].to_array()
    flowed_pc = pc + ego_frame_flow

    line_set = o3d.geometry.LineSet()
    line_set_points = np.concatenate([pc, flowed_pc], axis=0)
    lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    draw_color = (1, 0, 0)
    line_set.colors = o3d.utility.Vector3dVector(
        [draw_color for _ in range(len(lines))]
    )
    vis.add_geometry(line_set, reset_bounding_box=reset_view)


def draw_cuboids(vis, box_list, reset_view=False):
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [4, 5],
        [5, 6],
        [6, 7],
        [4, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Use the same color for all lines
    red = [[1, 0, 0] for _ in range(len(lines))]
    green = [[0, 1, 0] for _ in range(len(lines))]
    blue = [[0, 0.8, 0.8] for _ in range(len(lines))]
    magenta = [[1, 0, 1] for _ in range(len(lines))]

    for bbox in box_list:
        corner_box = bbox.corners().T

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        if bbox.name in BUCKETED_METACATAGORIES["PEDESTRIAN"]:
            colors = red
        elif bbox.name in BUCKETED_METACATAGORIES["CAR"]:
            colors = blue
        elif bbox.name in BUCKETED_METACATAGORIES["WHEELED_VRU"]:
            colors = green
        elif bbox.name in BUCKETED_METACATAGORIES["OTHER_VEHICLES"]:
            colors = magenta
        else:  # Background/static
            colors = [[1, 1, 0] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Display the bounding boxes:
        vis.add_geometry(line_set, reset_bounding_box=reset_view)

if __name__ == "__main__":
    vis = setup_vis()
    draw_frames(vis, reset_view=True)
    vis.run()
