import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.cuboid import Cuboid, CuboidList
from av2.structures.sweep import Sweep

# TRACKER_RESULTS_PATH = "/efs/lt3d_weights/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_av2/val_tracker_full_results/track_predictions.pkl"
# DATA_DIR = "/efs/argoverse2/val/"
# LOG_ID = "02678d04-cc9f-3148-9f95-1ba66347dff9"
# t_0 = 315969915260100000
# t_1 = 315969915359633000
# t_0 = 315969915359633000
# timestamps = [t_0, t_1]

# TRACKER_RESULTS_PATH = "/efs/bevfusion_av2_leaderboard_submissions/bevfusion_track_predictions.pkl"
TRACKER_RESULTS_PATH = "/efs/le3de2e_av2_leaderboard_submissions/le3de2e_track_predictions.pkl"
DATA_DIR = "/efs/argoverse2/test/"
LOG_ID_LIST = [
    "a7f532a3-87de-3129-8864-258396fd0b50",
    "a9a3d5d7-e0c6-3f24-af35-2acadc1aa2d9",
    "a69fa035-5121-3a39-a3ce-e33e9f54b506",
    "a86ee261-b86b-34f7-92ab-be8367d1fc4c",
    "a315b370-623a-3e19-8ffb-ba62661286ae",
    "a674e2e5-3dfd-3dd5-8503-192357b0e96c",
    "a1358c59-b28d-3ddb-af1c-3a5d1c394ef5",
    "a4400a38-bc38-391c-b102-ba385d7e475e",
    "98e7f0eb-4676-3120-94f1-8a790581e6a4",
]

# Meta classes
BACKGROUND = ["BACKGROUND"]
CAR = ["REGULAR_VEHICLE"]
OTHER_VEHICLES = [
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "ARTICULATED_BUS",
    "BUS",
    "SCHOOL_BUS",
]
PEDESTRIAN = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]
WHEELED_VRU = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]

CONFIDENCE_THRESHOLD = 0.2
LOG_NUM = 8
starter_idx = 0

# Get the point clouds:
dataset = AV2SensorDataLoader(data_dir=Path(DATA_DIR), labels_dir=Path(DATA_DIR))
timestamps = dataset.get_ordered_log_lidar_timestamps(LOG_ID_LIST[LOG_NUM])
with open(TRACKER_RESULTS_PATH, "rb") as openfile:
    track_data = pickle.load(openfile)


def load_cuboid_list(track_data: dict[str, Any], log_id: str, timestamp: int):
    tracks_for_log = track_data[log_id]

    cuboid_list: CuboidList = CuboidList(cuboids=[])

    for track in tracks_for_log:
        if track["timestamp_ns"] == timestamp:
            score_mask = np.nonzero(track["score"] > CONFIDENCE_THRESHOLD)

            for bbox in range(track["track_id"][score_mask].shape[0]):
                numpy_cuboid = np.array(
                    [
                        track["translation_m"][score_mask][bbox, 0],
                        track["translation_m"][score_mask][bbox, 1],
                        track["translation_m"][score_mask][bbox, 2]
                        + (0.5 * track["size"][score_mask][bbox, 2]),
                        track["size"][score_mask][bbox, 0],
                        track["size"][score_mask][bbox, 1],
                        track["size"][score_mask][bbox, 2],
                        np.cos(track["yaw"][score_mask][bbox] / 2),
                        0,
                        0,
                        np.sin(track["yaw"][score_mask][bbox] / 2),
                    ]
                )
                cuboid = Cuboid.from_numpy(
                    numpy_cuboid,
                    category=track["name"][score_mask][bbox],
                    timestamp_ns=track["timestamp_ns"],
                )
                cuboid_list.cuboids.append(cuboid)
    return cuboid_list


def matplotlib_bev(sweep, cuboid_list):
    pc0 = sweep.xyz
    plt.scatter(pc0[:, 0], pc0[:, 1], s=1, c="b", marker="o")
    for cuboid in cuboid_list.cuboids:
        plt.plot(cuboid.vertices_m[:, 0], cuboid.vertices_m[:, 1], ":", c="red")
    plt.show()


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


def draw_frames(vis, reset_view=False):
    ts = timestamps[starter_idx]
    lidar_pc = Sweep.from_feather(dataset.get_lidar_fpath(LOG_ID_LIST[LOG_NUM], ts))
    pose = dataset.get_city_SE3_ego(LOG_ID_LIST[LOG_NUM], ts)
    cuboids = load_cuboid_list(track_data, LOG_ID_LIST[LOG_NUM], ts)
    # Add base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pc.xyz)
    pc_color = np.ones_like(lidar_pc.xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    vis.add_geometry(pcd, reset_bounding_box=reset_view)
    # Draw the cuboids
    cuboids = cuboids.transform(pose.inverse())
    draw_cuboids(vis, cuboids)


def draw_cuboids(vis, cuboid_list, reset_view=False):
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

    for bbox in cuboid_list.cuboids:
        corner_box = bbox.vertices_m

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        if bbox.category in PEDESTRIAN:
            colors = red
        elif bbox.category in CAR:
            colors = blue
        elif bbox.category in WHEELED_VRU:
            colors = green
        elif bbox.category in OTHER_VEHICLES:
            colors = magenta
        else:  # Background shouldn't ever be a box
            colors = [[0, 1, 1] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Display the bounding boxes:
        vis.add_geometry(line_set, reset_bounding_box=reset_view)


if __name__ == "__main__":
    vis = setup_vis()
    draw_frames(vis, reset_view=True)
    vis.run()
