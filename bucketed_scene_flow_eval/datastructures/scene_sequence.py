from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy._typing import NDArray

from .camera_projection import CameraProjection
from .o3d_visualizer import O3DVisualizer
from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3

# Type alias for particle IDs
ParticleID = int
ParticleClassId = int
Timestamp = int

# Type alias for world points
WorldParticle = np.ndarray


@dataclass
class PoseInfo:
    sensor_to_ego: SE3
    ego_to_global: SE3

    def __repr__(self) -> str:
        return f"PoseInfo(sensor_to_ego={self.sensor_to_ego}, ego_to_global={self.ego_to_global})"


@dataclass
class PointCloudFrame:
    full_pc: PointCloud
    pose: PoseInfo
    mask: NDArray

    @property
    def pc(self) -> PointCloud:
        return self.full_pc.mask_points(self.mask)

    @property
    def full_global_pc(self) -> PointCloud:
        pose = self.global_pose
        return self.full_pc.transform(pose)

    @property
    def global_pc(self) -> PointCloud:
        pose = self.global_pose
        return self.pc.transform(pose)

    @property
    def global_pose(self) -> SE3:
        return self.pose.ego_to_global @ self.pose.sensor_to_ego


@dataclass
class RGBFrame:
    rgb: RGBImage
    pose: PoseInfo
    camera_projection: CameraProjection

    def __repr__(self) -> str:
        return f"RGBFrame(rgb={self.rgb},\npose={self.pose},\ncamera_projection={self.camera_projection})"


@dataclass
class RGBFrameLookup:
    lookup: dict[str, RGBFrame]
    entries: list[str]

    @staticmethod
    def empty() -> "RGBFrameLookup":
        return RGBFrameLookup({}, [])

    def __contains__(self, key: str) -> bool:
        return key in self.lookup

    def items(self) -> list[tuple[str, RGBFrame]]:
        return [(key, self.lookup[key]) for key in self.entries]

    def values(self) -> list[RGBFrame]:
        return [self.lookup[key] for key in self.entries]

    def __getitem__(self, key: str) -> RGBFrame:
        return self.lookup[key]

    def __len__(self) -> int:
        return len(self.lookup)


@dataclass
class RawSceneItem:
    pc_frame: PointCloudFrame
    rgb_frames: RGBFrameLookup


def _particle_id_to_color(particle_id: ParticleID) -> NDArray:
    particle_id = int(particle_id)
    assert isinstance(
        particle_id, ParticleID
    ), f"particle_id must be a ParticleID ({ParticleID}) , got {type(particle_id)}"
    hash_val = abs(hash(particle_id)) % (256**3)
    return np.array(
        [
            ((hash_val >> 16) & 0xFF) / 255,
            ((hash_val >> 8) & 0xFF) / 255,
            (hash_val & 0xFF) / 255,
        ]
    )


class RawSceneSequence:
    """
    This class contains only the raw percepts from a sequence. Its goal is to
    describe the scene as it is observed by the sensors; it does not contain
    any other information such as point position descriptions.

    These percept modalities are:
        - RGB
        - PointClouds

    Additionally, we store frame conversions for each percept.
    """

    def __init__(self, percept_lookup: dict[Timestamp, RawSceneItem], log_id: str):
        assert isinstance(
            percept_lookup, dict
        ), f"percept_lookup must be a dict, got {type(percept_lookup)}"
        assert all(
            isinstance(key, Timestamp) for key in percept_lookup.keys()
        ), f"percept_lookup keys must be Timestamp, got {[type(key) for key in percept_lookup.keys()]}"
        assert all(
            isinstance(value, RawSceneItem) for value in percept_lookup.values()
        ), f"percept_lookup values must be RawSceneItem, got {[type(value) for value in percept_lookup.values()]}"
        self.percept_lookup = percept_lookup
        self.log_id = log_id

    def get_percept_timesteps(self) -> list[int]:
        return sorted(self.percept_lookup.keys())

    def __len__(self):
        return len(self.get_percept_timesteps())

    def __getitem__(self, timestamp: int) -> RawSceneItem:
        assert isinstance(timestamp, int), f"timestamp must be an int, got {type(timestamp)}"
        return self.percept_lookup[timestamp]

    def visualize(self, vis: O3DVisualizer) -> O3DVisualizer:
        timesteps = self.get_percept_timesteps()
        grayscale_color = np.linspace(0, 1, len(timesteps) + 1)
        for idx, timestamp in enumerate(timesteps):
            item: RawSceneItem = self[timestamp]

            vis.add_pc_frame(item.pc_frame, color=[grayscale_color[idx]] * 3)
            vis.add_pose(item.pc_frame.global_pose)
        return vis

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, RawSceneSequence):
            return False
        return self.percept_lookup == __value.percept_lookup and self.log_id == __value.log_id


class QueryPointLookup:
    """
    This class is an efficient lookup table for query points.
    """

    def __init__(self, num_entries: int, query_init_timestamp: Timestamp):
        self.num_entries = num_entries
        self.query_init_world_points = np.zeros((num_entries, 3), dtype=np.float32)
        self.query_init_timestamp = query_init_timestamp
        self.is_valid = np.zeros((num_entries,), dtype=bool)

    def __len__(self) -> int:
        return self.is_valid.sum()

    def __getitem__(self, particle_id: ParticleID) -> tuple[WorldParticle, Timestamp]:
        assert (
            particle_id < self.num_entries
        ), f"particle_id {particle_id} must be less than {self.num_entries}"
        return self.query_init_world_points[particle_id], self.query_init_timestamp

    def __setitem__(self, particle_id_arr: np.ndarray, value: WorldParticle):
        assert (
            particle_id_arr < self.num_entries
        ).all(), f"particle_id value must be less than {self.num_entries}"
        self.query_init_world_points[particle_id_arr] = value
        self.is_valid[particle_id_arr] = True

    @property
    def particle_ids(self) -> NDArray:
        return np.arange(self.num_entries)[self.is_valid]

    def valid_query_init_world_points(self) -> NDArray:
        return self.query_init_world_points[self.is_valid]


class QuerySceneSequence:
    """
    This class describes a scene sequence with a query for motion descriptions.

    A query is a point + timestamp in the global frame of the scene, along with
    series of timestamps for which a point description is requested; motion is
    implied to be linear between these points at the requested timestamps.
    """

    def __init__(
        self,
        scene_sequence: RawSceneSequence,
        query_points: QueryPointLookup,
        query_flow_timestamps: list[Timestamp],
    ):
        assert isinstance(
            scene_sequence, RawSceneSequence
        ), f"scene_sequence must be a RawSceneSequence, got {type(scene_sequence)}"
        assert isinstance(
            query_points, QueryPointLookup
        ), f"query_particles must be a dict, got {type(query_points)}"
        assert isinstance(
            query_flow_timestamps, list
        ), f"query_timestamps must be a list, got {type(query_flow_timestamps)}"

        self.scene_sequence = scene_sequence

        ###################################################
        # Sanity checks to ensure that the query is valid #
        ###################################################

        # Check that the query timestamps all have corresponding percepts
        assert set(query_flow_timestamps).issubset(
            set(self.scene_sequence.get_percept_timesteps())
        ), f"Query timestamps {query_flow_timestamps} must be a subset of the scene sequence percepts {self.scene_sequence.get_percept_timesteps()}"

        self.query_flow_timestamps = query_flow_timestamps
        self.query_particles = query_points

    def __len__(self) -> int:
        return len(self.query_flow_timestamps)

    def visualize(
        self,
        vis: O3DVisualizer,
        percent_subsample: Union[None, float] = None,
        verbose=False,
    ) -> O3DVisualizer:
        if percent_subsample is not None:
            assert (
                percent_subsample > 0 and percent_subsample <= 1
            ), f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1
        # Visualize the query points ordered by particle ID
        particle_ids = self.query_particles.particle_ids
        world_particles = self.query_particles.valid_query_init_world_points()

        kth_particle_ids = particle_ids[::every_kth_particle]
        kth_world_particles = world_particles[::every_kth_particle]

        assert len(kth_particle_ids) == len(
            kth_world_particles
        ), f"Expected kth_particle_ids and kth_world_particles to have the same length, got {len(kth_particle_ids)} and {len(kth_world_particles)} instead"

        kth_particle_colors = [
            _particle_id_to_color(particle_id) for particle_id in kth_particle_ids
        ]
        assert len(kth_particle_colors) == len(
            kth_particle_ids
        ), f"Expected kth_particle_colors and kth_particle_ids to have the same length, got {len(kth_particle_colors)} and {len(kth_particle_ids)} instead"

        vis.add_spheres(kth_world_particles, 0.1, kth_particle_colors)
        return vis


class EstimatedPointFlow:
    def __init__(
        self,
        num_entries: int,
        trajectory_timestamps: Union[list[Timestamp], np.ndarray],
    ):
        self.num_entries = num_entries

        if isinstance(trajectory_timestamps, list):
            trajectory_timestamps = np.array(trajectory_timestamps)

        assert (
            trajectory_timestamps.ndim == 1
        ), f"trajectory_timestamps must be a 1D array, got {trajectory_timestamps.ndim}"
        self.trajectory_timestamps = trajectory_timestamps
        self.trajectory_length = len(trajectory_timestamps)

        self.world_points = np.zeros((num_entries, self.trajectory_length, 3), dtype=np.float32)

        # By default, all trajectories are invalid
        self.is_valid_flow = np.zeros((num_entries,), dtype=bool)

    def valid_particle_ids(self) -> NDArray:
        return np.arange(self.num_entries)[self.is_valid_flow]

    def __len__(self) -> int:
        return self.is_valid_flow.sum()

    def __setitem__(self, particle_id: ParticleID, points: NDArray):
        self.world_points[particle_id] = points
        self.is_valid_flow[particle_id] = True

    def visualize(
        self,
        vis: O3DVisualizer,
        percent_subsample: Union[None, float] = None,
        verbose: bool = False,
    ) -> O3DVisualizer:
        if percent_subsample is not None:
            assert (
                percent_subsample > 0 and percent_subsample <= 1
            ), f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1

        # Shape: points, 2, 3
        world_points = self.world_points.copy()
        world_points = world_points[self.is_valid_flow]
        world_points = world_points[::every_kth_particle]

        vis.add_trajectories(world_points)
        return vis


class GroundTruthPointFlow(EstimatedPointFlow):
    def __init__(
        self,
        num_entries: int,
        trajectory_timestamps: Union[list[Timestamp], np.ndarray],
        query_timestamp: int,
        class_name_map: Optional[dict[ParticleClassId, str]] = None,
    ):
        super().__init__(num_entries, trajectory_timestamps)
        self.class_name_map = class_name_map
        self.cls_ids = np.zeros((num_entries,), dtype=np.int64)
        self.query_timestamp = query_timestamp
        assert (
            self.query_timestamp in self.trajectory_timestamps
        ), f"query_timestamp {self.query_timestamp} must be in trajectory_timestamps {self.trajectory_timestamps}"

    def _mask_entries(self, mask: np.ndarray):
        assert mask.ndim == 1, f"mask must be a 1D array, got {mask.ndim}"

        assert (
            len(mask) == self.num_entries
        ), f"mask must be the same length as the number of entries, got {len(mask)} and {self.num_entries} instead"

        self.is_valid_flow[~mask] = False

    def __setitem__(
        self,
        particle_id: ParticleID,
        data_tuple: tuple[NDArray, ParticleClassId, NDArray],
    ):
        points, cls_ids, is_valids = data_tuple
        self.world_points[particle_id] = points
        self.cls_ids[particle_id] = cls_ids
        self.is_valid_flow[particle_id] = is_valids

    def pretty_name(self, class_id: ParticleClassId) -> str:
        if self.class_name_map is None:
            return str(class_id)

        if class_id not in self.class_name_map:
            return str(class_id)

        return self.class_name_map[class_id]
