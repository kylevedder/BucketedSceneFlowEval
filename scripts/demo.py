from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import  QuerySceneSequence, GroundTruthParticleTrajectories, O3DVisualizer, PointCloud, PointCloudFrame, RGBFrame
from matplotlib import pyplot as plt
import numpy as np
import argparse

def color_threshold_distance(distances: np.ndarray, max_distance: float = 10.0):
    # Use distance to color points, normalized to [0, 1].
    colors = distances.copy()
    beyond_max_distance = colors > max_distance

    colors[beyond_max_distance] = 0
    colors[~beyond_max_distance] = 1
    # Make from grayscale to RGB
    colors = np.stack([colors, colors, colors], axis=1)
    return colors


def process_lidar_only(o3d_vis : O3DVisualizer, pc_frame : PointCloudFrame):
    o3d_vis.add_pointcloud(pc_frame.global_pc)
    o3d_vis.run()

def process_lidar_rgb(o3d_vis : O3DVisualizer, pc_frame : PointCloudFrame, rgb_frame : RGBFrame):
    image_plane_pc, colors = rgb_frame.camera_projection.image_to_image_plane_pc(rgb_frame.rgb, depth=10)

    pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.compose(rgb_frame.pose.sensor_to_ego.inverse())
    cam_frame_pc = pc_frame.pc.transform(pc_into_cam_frame_se3)

    # To prevent points behind the camera from being projected into the image, we had to remove them from the pointcloud.
    # These points have a negative X value in the camera frame.

    cam_frame_pc = PointCloud(cam_frame_pc.points[cam_frame_pc.points[:, 0] >= 0])

    o3d_vis.add_pointcloud(cam_frame_pc)
    o3d_vis.add_pointcloud(image_plane_pc, color=colors)
    o3d_vis.run()


    projected_points = rgb_frame.camera_projection.camera_frame_to_pixels(cam_frame_pc.points)
    projected_points = projected_points.astype(np.int32)

    # Use distance to color points, normalized to [0, 1]. Let points more than 10m away be black.
    colors = color_threshold_distance(cam_frame_pc.points[:, 0], max_distance=10)
    
    # Mask out points that are out of bounds
    
    valid_points_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < rgb_frame.rgb.image.shape[1]) & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < rgb_frame.rgb.image.shape[0])
    projected_points = projected_points[valid_points_mask]
    colors = colors[valid_points_mask]

    projected_rgb_image = rgb_frame.rgb.image
    projected_rgb_image[projected_points[:, 1], projected_points[:, 0], :] = colors
    plt.imshow(projected_rgb_image)
    plt.show()

def process_entry(query: QuerySceneSequence, gt: GroundTruthParticleTrajectories):
    # The query specifies the raw scene and query points at a particular timestamp
    # These query points can be thought of as the specification of the valid points for 
    # scene flow in the pointcloud at `t` for prediction to timestamp `t+1`

    scene_timestamps = query.scene_sequence.get_percept_timesteps()
    print("Scene timestamps:", scene_timestamps)

    query_timestamp = query.query_particles.query_init_timestamp
    print("Query timestamp:", query_timestamp)

    gt_timestamps = gt.trajectory_timestamps
    print("GT timestamps:", gt_timestamps)

    # The scene contains RGB image and pointcloud data for each timestamp.
    # These are stored as "frames" with pose and intrinsics information. 
    # This enables the raw percepts to be projected into desired coordinate frames across time.
    for scene_timestamp in scene_timestamps:
        rgb_frame = query.scene_sequence[scene_timestamp].rgb_frame
        pc_frame = query.scene_sequence[scene_timestamp].pc_frame

        o3d_vis = O3DVisualizer(point_size=0.5)
        gt.visualize(o3d_vis)

        if rgb_frame is None:
            print("No RGB frame for timestamp", scene_timestamp, "using lidar only")
            process_lidar_only(o3d_vis, pc_frame)
        else:
            print("RGB frame for timestamp", scene_timestamp, "using lidar and rgb")
            process_lidar_rgb(o3d_vis, pc_frame, rgb_frame)

        del o3d_vis

        



if __name__ == "__main__":
    # Take arguments to specify dataset and root directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Argoverse2SceneFlow')
    parser.add_argument('--root_dir', type=str, default='/efs/argoverse2/val')
    args = parser.parse_args()


    dataset = construct_dataset(args.dataset, dict(root_dir = args.root_dir))

    print("Dataset contains", len(dataset), "samples")

    query, gt = dataset[0]
    process_entry(query, gt)
