from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation as R

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    PoseInfo,
    TimeSyncedSceneFlowBoxFrame,
)


class AnnotationSaver:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def format_annotations(self, frames: list[TimeSyncedSceneFlowBoxFrame]) -> list[dict]:
        formatted_annotations = []

        for frame_num, frame in enumerate(frames):
            timestamp_ns = frame.log_timestamp
            for box, pose_info in zip(frame.boxes.full_boxes, frame.boxes.full_poses):
                pose = pose_info.sensor_to_ego
                rotation_matrix = pose.transform_matrix[:3, :3]
                rotation = R.from_matrix(rotation_matrix)
                qw, qx, qy, qz = rotation.as_quat()

                tx, ty, tz = pose.transform_matrix[:3, 3]

                # Format the annotation
                formatted_annotation = {
                    "timestamp_ns": timestamp_ns,
                    "track_uuid": box.track_uuid,
                    "category": box.category,
                    "length_m": box.length,
                    "width_m": box.width,
                    "height_m": box.height,
                    "qw": qw,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "tx_m": tx,
                    "ty_m": ty,
                    "tz_m": tz,
                    "num_interior_pts": 0,  # Placeholder for num_interior_pts
                }

                formatted_annotations.append(formatted_annotation)
        return formatted_annotations

    def save_callback(self, vis, action, mods, frames):
        mods_name = ["shift", "ctrl", "alt", "cmd"]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]
        if mods == ["ctrl"]:
            self.save(frames)

    def save(self, frames: list[TimeSyncedSceneFlowBoxFrame]):
        formatted_annotations = self.format_annotations(frames)
        df = pd.DataFrame(formatted_annotations)
        feather_file_path = Path(self.save_dir) / "annotations.feather"
        feather.write_feather(df, feather_file_path)
        print(f"Annotations saved to {feather_file_path}")
