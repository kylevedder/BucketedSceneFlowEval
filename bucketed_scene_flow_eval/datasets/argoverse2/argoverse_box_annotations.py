from dataclasses import dataclass
from pathlib import Path

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    TimeSyncedAVLidarData,
    TimeSyncedSceneFlowBoxFrame,
    TimeSyncedSceneFlowFrame,
)
from bucketed_scene_flow_eval.utils import load_feather

from .argoverse_scene_flow import ArgoverseNoFlowSequence, ArgoverseNoFlowSequenceLoader


class ArgoverseBoxAnnotationSequence(ArgoverseNoFlowSequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp_to_boxes = self._prep_bbox_annotations()

    def _prep_bbox_annotations(self) -> dict[int, list[BoundingBox]]:
        annotations_file = self.dataset_dir / "annotations.feather"
        assert annotations_file.exists(), f"Annotations file {annotations_file} does not exist"
        annotation_df = load_feather(annotations_file)
        # Index(['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m',
        #         'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m',
        #         'num_interior_pts'],
        #         dtype='object')

        # Convert to dictionary keyed by timestamp_ns int
        timestamp_to_annotations: dict[int, list[BoundingBox]] = {}
        for _, row in annotation_df.iterrows():
            timestamp_ns = row["timestamp_ns"]
            if timestamp_ns not in timestamp_to_annotations:
                timestamp_to_annotations[timestamp_ns] = []
            pose = SE3.from_rot_w_x_y_z_translation_x_y_z(
                row["qw"],
                row["qx"],
                row["qy"],
                row["qz"],
                row["tx_m"],
                row["ty_m"],
                row["tz_m"],
            )
            timestamp_to_annotations[timestamp_ns].append(
                BoundingBox(
                    pose=pose,
                    length=row["length_m"],
                    width=row["width_m"],
                    height=row["height_m"],
                    track_uuid=row["track_uuid"],
                    category=row["category"],
                )
            )
        return timestamp_to_annotations

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = False
    ) -> tuple[TimeSyncedSceneFlowBoxFrame, TimeSyncedAVLidarData]:
        scene_flow_frame, lidar_data = super().load(idx, relative_to_idx, with_flow)
        timestamp = self.timestamp_list[idx]
        boxes = self.timestamp_to_boxes.get(timestamp, [])
        return TimeSyncedSceneFlowBoxFrame(**vars(scene_flow_frame), boxes=boxes), lidar_data


class ArgoverseBoxAnnotationSequenceLoader(ArgoverseNoFlowSequenceLoader):

    def _load_sequence_uncached(self, sequence_id: str) -> ArgoverseBoxAnnotationSequence:
        assert (
            sequence_id in self.sequence_id_to_raw_data
        ), f"sequence_id {sequence_id} does not exist"
        return ArgoverseBoxAnnotationSequence(
            sequence_id,
            self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_raw_data[sequence_id],
            with_classes=False,
            **self.load_sequence_kwargs,
        )

    def cache_folder_name(self) -> str:
        return f"av2_box_data_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_No_flow_data_path"
