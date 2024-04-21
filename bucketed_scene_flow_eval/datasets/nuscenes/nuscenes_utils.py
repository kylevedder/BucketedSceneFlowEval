"""This file contains utility functions helpful for using the NuScenes dataset."""

import numpy as np
from av2.geometry.geometry import compute_interior_points_mask
from av2.utils.typing import NDArrayBool, NDArrayFloat
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from bucketed_scene_flow_eval.datastructures.se3 import SE3


class InstanceBox(Box):
    def __init__(
        self,
        center: list[float],
        size: list[float],
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity: tuple = (np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
        instance_token: str = None,
    ):
        super().__init__(center, size, orientation, label, score, velocity, name, token)
        self.instance_token = instance_token

    def compute_interior_points(
        self, points_xyz_m: NDArrayFloat, wlh_factor: float = 1.0
    ) -> tuple[NDArrayFloat, NDArrayBool]:
        """Given a query point cloud, filter to points interior to the cuboid, and provide mask.

        Note: comparison is to cuboid vertices in the destination reference frame.

        Args:
            points_xyz_m: (N,3) Points to filter.
            wlh_factor: Multiply w, l, h by a factor to scale the box.

        Returns:
            The interior points and the boolean array indicating which points are interior.
        """
        vertices_dst_xyz_m = self.corners(wlh_factor=wlh_factor).T
        is_interior = compute_interior_points_mask(points_xyz_m, vertices_dst_xyz_m)
        return points_xyz_m[is_interior], is_interior

    def transform(self, pose: SE3) -> None:
        self.rotate(Quaternion(matrix=pose.rotation_matrix))
        self.translate(pose.translation)
        return self

    @property
    def dst_SE3_object(self) -> SE3:
        return SE3(self.rotation_matrix, self.center)


class NuScenesWithInstanceBoxes(NuScenes):
    """A wrapper class of the NuScenes DB object that provides extra functions.

    Most importantly allows boxes to carry over their instance tokens when they are interpolated using `get_boxes` from a sample data token.
    """

    def __init__(
        self,
        version: str = "v1.0-mini",
        dataroot: str = "/data/sets/nuscenes",
        verbose: bool = True,
        map_resolution: float = 0.1,
    ):
        super().__init__(version, dataroot, verbose, map_resolution)

    def get_instance_box(self, sample_annotation_token: str) -> InstanceBox:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get("sample_annotation", sample_annotation_token)
        return InstanceBox(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
            instance_token=record["instance_token"],
        )

    def get_boxes_with_instance_token(self, sample_data_token: str) -> list[InstanceBox]:
        """
        This function is a copy of get_boxes() except that it adds the instance tokens to the box objects.

        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])

        if curr_sample_record["prev"] == "" or sd_record["is_key_frame"]:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_instance_box, curr_sample_record["anns"]))

        else:
            prev_sample_record = self.get("sample", curr_sample_record["prev"])

            curr_ann_recs = [
                self.get("sample_annotation", token) for token in curr_sample_record["anns"]
            ]
            prev_ann_recs = [
                self.get("sample_annotation", token) for token in prev_sample_record["anns"]
            ]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

            t0 = prev_sample_record["timestamp"]
            t1 = curr_sample_record["timestamp"]
            t = sd_record["timestamp"]

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec["instance_token"] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec["instance_token"]]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=Quaternion(prev_ann_rec["rotation"]),
                        q1=Quaternion(curr_ann_rec["rotation"]),
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = InstanceBox(
                        center,
                        curr_ann_rec["size"],
                        rotation,
                        name=curr_ann_rec["category_name"],
                        token=curr_ann_rec["token"],
                        instance_token=curr_ann_rec["instance_token"],
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_instance_box(curr_ann_rec["token"])

                boxes.append(box)
        return boxes
