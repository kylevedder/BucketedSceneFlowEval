import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional

from numpy._typing import NDArray

from .camera_projection import CameraProjection
from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3
from .o3d_visualizer import O3DVisualizer

from dataclasses import dataclass
# import named tuple
from collections import namedtuple

# Type alias for particle IDs
ParticleID = int
ParticleClassId = int
Timestamp = int

# Type alias for world points
WorldParticle = np.array


@dataclass
class EstimatedParticle():
    point: WorldParticle
    is_occluded: bool


@dataclass
class PoseInfo():
    sensor_to_ego: SE3
    ego_to_global: SE3


@dataclass
class PointCloudFrame():
    pc: PointCloud
    pose: PoseInfo

    @property
    def global_pc(self) -> PointCloud:
        pose = self.global_pose
        return self.pc.transform(pose)

    @property
    def global_pose(self) -> SE3:
        return self.pose.ego_to_global @ self.pose.sensor_to_ego


@dataclass
class RGBFrame():
    rgb: RGBImage
    pose: PoseInfo
    camera_projection: CameraProjection


@dataclass
class ParticleTrajectory():
    id: ParticleID
    trajectory: Dict[Timestamp, EstimatedParticle]
    cls: Union[ParticleClassId, None] = None

    def __len__(self):
        return len(self.trajectory)

    def get_first_timestamp(self) -> Timestamp:
        return min(self.trajectory.keys())

    def __getitem__(self, timestamp: Timestamp) -> EstimatedParticle:
        return self.trajectory[timestamp]


def _particle_id_to_color(
        particle_id: ParticleID) -> Tuple[float, float, float]:
    particle_id = int(particle_id)
    assert isinstance(particle_id, ParticleID), \
        f"particle_id must be a ParticleID ({ParticleID}) , got {type(particle_id)}"
    hash_val = abs(hash(particle_id)) % (256**3)
    return np.array([((hash_val >> 16) & 0xff) / 255,
                     ((hash_val >> 8) & 0xff) / 255, (hash_val & 0xff) / 255])


@dataclass
class RawSceneItem():
    pc_frame: PointCloudFrame
    rgb_frame: Optional[RGBFrame]


class RawSceneSequence():
    """
    This class contains only the raw percepts from a sequence. Its goal is to 
    describe the scene as it is observed by the sensors; it does not contain
    any other information such as point position descriptions.

    These percept modalities are:
        - RGB
        - PointClouds

    Additionally, we store frame conversions for each percept.
    """
    def __init__(self, percept_lookup: Dict[Timestamp, RawSceneItem], log_id : str):
        assert isinstance(percept_lookup, dict), \
            f"percept_lookup must be a dict, got {type(percept_lookup)}"
        assert all(
            isinstance(key, Timestamp) for key in percept_lookup.keys()), \
            f"percept_lookup keys must be Timestamp, got {[type(key) for key in percept_lookup.keys()]}"
        assert all(
            isinstance(value, RawSceneItem) for value in percept_lookup.values()), \
            f"percept_lookup values must be RawSceneItem, got {[type(value) for value in percept_lookup.values()]}"
        self.percept_lookup = percept_lookup
        self.log_id = log_id

    def get_percept_timesteps(self) -> List[int]:
        return sorted(self.percept_lookup.keys())

    def __len__(self):
        return len(self.get_percept_timesteps())

    def __getitem__(self, timestamp: int) -> RawSceneItem:
        assert isinstance(
            timestamp, int), f"timestamp must be an int, got {type(timestamp)}"
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


class QueryParticleLookup():
    """
    This class is an efficient lookup table for query particles.
    """
    def __init__(self, num_entries: int, query_init_timestamp: Timestamp):
        self.num_entries = num_entries
        self.query_init_world_particles = np.zeros((num_entries, 3),
                                                   dtype=np.float32)
        self.query_init_timestamp = query_init_timestamp
        self.is_valid = np.zeros((num_entries, ), dtype=bool)

    def __len__(self) -> int:
        return self.num_entries

    def __getitem__(
            self, particle_id: ParticleID) -> Tuple[WorldParticle, Timestamp]:
        assert particle_id < self.num_entries, \
            f"particle_id {particle_id} must be less than {self.num_entries}"
        return self.query_init_world_particles[
            particle_id], self.query_init_timestamp

    def __setitem__(self, particle_id: ParticleID, value: WorldParticle):
        assert (particle_id < self.num_entries).all(), \
            f"particle_ids {particle_id[particle_id >= self.num_entries]} must be less than {self.num_entries}"
        self.query_init_world_particles[particle_id] = value
        self.is_valid[particle_id] = True

    @property
    def particle_ids(self) -> NDArray:
        return np.arange(self.num_entries)[self.is_valid]

    def valid_query_init_world_particles(self) -> NDArray:
        return self.query_init_world_particles[self.is_valid]


class QuerySceneSequence:
    """
    This class describes a scene sequence with a query for motion descriptions.

    A query is a point + timestamp in the global frame of the scene, along with 
    series of timestamps for which a point description is requested; motion is
    implied to be linear between these points at the requested timestamps.
    """
    def __init__(self, scene_sequence: RawSceneSequence,
                 query_particles: QueryParticleLookup,
                 query_trajectory_timestamps: List[Timestamp]):
        assert isinstance(scene_sequence, RawSceneSequence), \
            f"scene_sequence must be a RawSceneSequence, got {type(scene_sequence)}"
        assert isinstance(query_particles, QueryParticleLookup), \
            f"query_particles must be a dict, got {type(query_particles)}"
        assert isinstance(query_trajectory_timestamps, list), \
            f"query_timestamps must be a list, got {type(query_trajectory_timestamps)}"

        self.scene_sequence = scene_sequence

        ###################################################
        # Sanity checks to ensure that the query is valid #
        ###################################################

        # Check that the query timestamps all have corresponding percepts
        assert set(query_trajectory_timestamps).issubset(set(self.scene_sequence.get_percept_timesteps())), \
            f"Query timestamps {query_trajectory_timestamps} must be a subset of the scene sequence percepts {self.scene_sequence.get_percept_timesteps()}"

        # Check that the query points all have corresponding timestamps
        # assert len(
        #     set(self.scene_sequence.get_percept_timesteps()).intersection(
        #         set([t for _, t in query_particles.values()]))) > 0

        self.query_trajectory_timestamps = query_trajectory_timestamps
        self.query_particles = query_particles

    def __len__(self) -> int:
        return len(self.query_trajectory_timestamps)

    def visualize(self,
                  vis: O3DVisualizer,
                  percent_subsample: Union[None, float] = None,
                  verbose=False) -> O3DVisualizer:
        if percent_subsample is not None:
            assert percent_subsample > 0 and percent_subsample <= 1, \
                f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1
        # Visualize the query points ordered by particle ID
        particle_ids = self.query_particles.particle_ids
        world_particles = self.query_particles.valid_query_init_world_particles(
        )

        kth_particle_ids = particle_ids[::every_kth_particle]
        kth_world_particles = world_particles[::every_kth_particle]

        assert len(kth_particle_ids) == len(kth_world_particles), \
            f"Expected kth_particle_ids and kth_world_particles to have the same length, got {len(kth_particle_ids)} and {len(kth_world_particles)} instead"

        kth_particle_colors = [
            _particle_id_to_color(particle_id)
            for particle_id in kth_particle_ids
        ]
        assert len(kth_particle_colors) == len(kth_particle_ids), \
            f"Expected kth_particle_colors and kth_particle_ids to have the same length, got {len(kth_particle_colors)} and {len(kth_particle_ids)} instead"

        vis.add_spheres(kth_world_particles, 0.1, kth_particle_colors)
        return vis


class EstimatedParticleTrajectories():
    def __init__(self, num_entries: int,
                 trajectory_timestamps: Union[List[Timestamp], np.ndarray]):
        self.num_entries = num_entries

        if isinstance(trajectory_timestamps, list):
            trajectory_timestamps = np.array(trajectory_timestamps)

        assert trajectory_timestamps.ndim == 1, \
            f"trajectory_timestamps must be a 1D array, got {trajectory_timestamps.ndim}"
        self.trajectory_timestamps = trajectory_timestamps
        self.trajectory_length = len(trajectory_timestamps)

        self.world_points = np.zeros((num_entries, self.trajectory_length, 3),
                                     dtype=np.float32)

        self.is_occluded = np.zeros((num_entries, self.trajectory_length),
                                    dtype=bool)
        # By default, all trajectories are invalid
        self.is_valid = np.zeros((num_entries, self.trajectory_length),
                                 dtype=bool)

    def valid_particle_ids(self) -> NDArray:
        is_valid_sum = self.is_valid.sum(axis=1)
        return np.arange(self.num_entries)[is_valid_sum > 0]

    def __len__(self) -> int:
        return self.num_entries

    def __setitem__(self, particle_id: ParticleID,
                    data_tuple: Tuple[NDArray, NDArray, NDArray]):
        points, timestamps, is_occludeds = data_tuple
        self.world_points[particle_id] = points
        self.is_occluded[particle_id] = is_occludeds
        self.is_valid[particle_id] = True


class GroundTruthParticleTrajectories():
    """
    This class is an efficient lookup table for particle trajectories.

    It is designed to present like Dict[ParticleID, ParticleTrajectory] but backed by a numpy array.
    """
    def __init__(self,
                 num_entries: int,
                 trajectory_timestamps: Union[List[Timestamp], np.ndarray],
                 query_timestamp: int,
                 class_name_map: Dict[ParticleClassId, str] = None):
        self.num_entries = num_entries
        self.class_name_map = class_name_map

        if isinstance(trajectory_timestamps, list):
            trajectory_timestamps = np.array(trajectory_timestamps)

        assert trajectory_timestamps.ndim == 1, \
            f"trajectory_timestamps must be a 1D array, got {trajectory_timestamps.ndim}"
        self.trajectory_timestamps = trajectory_timestamps
        self.trajectory_length = len(trajectory_timestamps)

        self.world_points = np.zeros((num_entries, self.trajectory_length, 3),
                                     dtype=np.float32)
        self.is_occluded = np.zeros((num_entries, self.trajectory_length),
                                    dtype=bool)
        # By default, all trajectories are invalid
        self.is_valid = np.zeros((num_entries, self.trajectory_length),
                                 dtype=bool)
        self.cls_ids = np.zeros((num_entries, ), dtype=np.int64)
        self.query_timestamp = query_timestamp
        assert self.query_timestamp in self.trajectory_timestamps, \
            f"query_timestamp {self.query_timestamp} must be in trajectory_timestamps {self.trajectory_timestamps}"

    def __len__(self) -> int:
        return self.num_entries

    def _mask_entries(self, mask: np.ndarray):
        assert mask.ndim == 1, \
            f"mask must be a 1D array, got {mask.ndim}"

        assert len(mask) == self.num_entries, \
            f"mask must be the same length as the number of entries, got {len(mask)} and {self.num_entries} instead"

        self.is_valid[~mask] = False

    def __setitem__(self, particle_id: ParticleID,
                    data_tuple: Tuple[NDArray, NDArray, ParticleClassId,
                                      NDArray]):
        points, is_occludeds, cls_ids, is_valids = data_tuple
        self.world_points[particle_id] = points
        self.is_occluded[particle_id] = is_occludeds
        self.cls_ids[particle_id] = cls_ids
        self.is_valid[particle_id] = is_valids

    def valid_particle_ids(self) -> NDArray:
        is_valid_sum = self.is_valid.sum(axis=1)
        return np.arange(self.num_entries)[is_valid_sum > 0]

    def pretty_name(self, class_id: ParticleClassId) -> str:
        if self.class_name_map is None:
            return str(class_id)

        if class_id not in self.class_name_map:
            return str(class_id)

        return self.class_name_map[class_id]

    def visualize(self,
                  vis: O3DVisualizer,
                  percent_subsample: Union[None, float] = None,
                  verbose: bool = False) -> O3DVisualizer:
        if percent_subsample is not None:
            assert percent_subsample > 0 and percent_subsample <= 1, \
                f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1

        # Shape: points, 2, 3
        world_points = self.world_points
        is_valid = self.is_valid

        vis.add_trajectories(world_points)
        return vis
