import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.cuboid import Cuboid, CuboidList
from av2.structures.sweep import Sweep

from bucketed_scene_flow_eval.utils.loaders import load_feather

# DETECTOR_RESULTS_PATH = "/efs/le3de2e_av2_leaderboard_submissions/LE3DE2E_Detections_Test.feather"
DETECTOR_RESULTS_PATH = (
    "/efs/bevfusion_av2_leaderboard_submissions/BEVFusion_Detections_Test.feather"
)
DATA_DIR = "/efs/argoverse2/test/"
LOG_ID = "0c6e62d7-bdfa-3061-8d3d-03b13aa21f68"
t_0 = 315971436059707000
t_1 = 315971437859906000
timestamps = [t_0, t_1]

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

detections_df = load_feather(Path(DETECTOR_RESULTS_PATH))

cuboid_list_df = detections_df.loc[
    (detections_df["log_id"] == LOG_ID)
    & (detections_df["timestamp_ns"] == t_0)
    & (detections_df["score"] > CONFIDENCE_THRESHOLD)
]

# print(cuboid_list_df.info)

cuboid_list = CuboidList.from_dataframe(cuboid_list_df)

# Get the point clouds:

dataset = AV2SensorDataLoader(data_dir=Path(DATA_DIR), labels_dir=Path(DATA_DIR))

sweeps = [Sweep.from_feather(dataset.get_lidar_fpath(LOG_ID, ts)) for ts in timestamps]
# cuboids = get_ids_and_cuboids_at_lidar_timestamps(dataset, log_id, timestamps)
poses = [dataset.get_city_SE3_ego(LOG_ID, ts) for ts in timestamps]

"""To plot bboxes

gt_corners = boxes to corners
for index in range(gt_corners.shape[0])
    plt.plot(gt_corners[index, :, 0], gt_corners[index, :, 1], ":", c="red")
"""

pc0 = sweeps[0].xyz
# plt.scatter(pc0[:, 0], pc0[:, 1], s=1, c="b", marker="o")

pose0 = poses[0].inverse()
# cuboid_list = cuboid_list.transform(pose0)

# for cuboid in cuboid_list.cuboids:
#     plt.plot(cuboid.vertices_m[:, 0], cuboid.vertices_m[:, 1], ":", c="red")

# plt.show()

# make open3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1.5
vis.get_render_option().background_color = (0.8, 0.8, 0.8)
# vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

# Add base point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc0)
pc_color = np.ones_like(pc0)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
vis.add_geometry(pcd, reset_bounding_box=True)

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
    vis.add_geometry(line_set, reset_bounding_box=True)

vis.run()
