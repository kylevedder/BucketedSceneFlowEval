import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    TimeSyncedSceneFlowBoxFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractSequence
from bucketed_scene_flow_eval.utils.glfw_key_ids import *


def _update_o3d_mesh_pose(mesh: o3d.geometry.TriangleMesh, start_pose: SE3, target_pose: SE3):
    global_translation = target_pose.translation - start_pose.translation
    global_rotation = target_pose.rotation_matrix @ np.linalg.inv(start_pose.rotation_matrix)

    mesh.translate(global_translation)
    mesh.rotate(global_rotation, center=target_pose.translation)


class BoxGeometryWithPose:
    def __init__(self, base_box: BoundingBox, color=[0.1, 0.1, 0.1]):
        self.base_box = base_box
        self.color = color

        # O3D doesn't support rendering boxers as wireframes directly, so we create a box and its associated rendered lineset.
        self.o3d_triangle_mesh = o3d.geometry.TriangleMesh.create_box(
            width=base_box.length, height=base_box.height, depth=base_box.width
        )
        self.o3d_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.o3d_triangle_mesh)
        self.imit_pose_of_o3d_geomerty(base_box.pose)
        self.set_color(self.color)

    def set_color(self, color):
        """
        Sets the color of the geomerty

        Args:
            color: A list or array of three floats representing the RGB color.
        """
        num_lines = len(self.o3d_wireframe.lines)
        self.o3d_wireframe.colors = o3d.utility.Vector3dVector([color] * num_lines)

    def imit_pose_of_o3d_geomerty(self, pose: SE3):
        o3d_geom_centering_translation = -np.array(
            [0.5 * self.base_box.length, 0.5 * self.base_box.height, 0.5 * self.base_box.width]
        )

        center_offset_se3 = SE3.identity().translate(o3d_geom_centering_translation)
        o3d_target_pose = pose.compose(center_offset_se3)

        _update_o3d_mesh_pose(self.o3d_wireframe, SE3.identity(), o3d_target_pose)
        _update_o3d_mesh_pose(self.o3d_triangle_mesh, SE3.identity(), o3d_target_pose)

    def compute_global_pose(
        self,
        forward: float = 0,
        left: float = 0,
        up: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
    ) -> SE3:
        local_frame_offset_se3 = SE3.from_rot_x_y_z_translation_x_y_z(
            roll, pitch, yaw, forward, left, up
        )

        return self.base_box.pose.compose(local_frame_offset_se3)

    def update_from_global(self, global_se3: SE3):
        _update_o3d_mesh_pose(self.o3d_wireframe, self.base_box.pose, global_se3)
        _update_o3d_mesh_pose(self.o3d_triangle_mesh, self.base_box.pose, global_se3)
        self.base_box.pose = global_se3

    def triangle_mesh_o3d(self) -> o3d.geometry.TriangleMesh:
        return self.o3d_triangle_mesh

    def wireframe_o3d(self) -> o3d.geometry.LineSet:
        return self.o3d_wireframe

    @property
    def pose(self) -> SE3:
        return self.base_box.pose


def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2) -> tuple[bool, np.ndarray | None]:
    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False, None  # This ray is parallel to this triangle.
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if not (0.0 <= u <= 1.0):
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if not (0.0 <= v <= 1.0):
        return False, None
    if u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        intersect_point = ray_origin + ray_direction * t
        return True, intersect_point
    else:
        return (
            False,
            None,
        )  # This means that there is a line intersection but not a ray intersection.


class ViewStateManager:
    def __init__(self, frames, cropping_range, rolling_window_size) -> None:
        self.prior_mouse_position: tuple[float, float] | None = None
        self.is_view_rotating = False
        self.is_translating = False
        self.pixel_to_rotate_scale_factor = 1
        self.pixel_to_translate_scale_factor = 1
        self.clickable_geometries: dict[str, BoxGeometryWithPose] = {}
        self.selection_axes: o3d.geometry.TriangleMesh | None = None
        self.selected_mesh_id: str | None = None
        self.current_frame_index = 0
        self.tuning_scale = 0.1
        self.frames = frames
        self.cropping_range = cropping_range
        self.rolling_window_size = rolling_window_size
        self.trajectory_geometries = []  # Track geometries for the selected mesh trajectory

    def add_clickable_geometry(self, id: str, box_geometry: BoxGeometryWithPose):
        self.clickable_geometries[id] = box_geometry

    def _update_selection(
        self,
        vis,
        forward: float = 0,
        left: float = 0,
        up: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
    ):
        assert self.selected_mesh_id is not None
        assert self.selection_axes is not None
        selected_mesh = self.clickable_geometries[self.selected_mesh_id]
        global_target_se3 = selected_mesh.compute_global_pose(
            forward=forward, left=left, up=up, pitch=pitch, yaw=yaw, roll=roll
        )
        # for g in global_target_se3.to_o3d():
        #     vis.add_geometry(g, reset_bounding_box=False)

        _update_o3d_mesh_pose(self.selection_axes, selected_mesh.pose, global_target_se3)
        selected_mesh.update_from_global(global_target_se3)

        vis.update_geometry(selected_mesh.wireframe_o3d())
        vis.update_geometry(self.selection_axes)

    def forward_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, forward=self.tuning_scale)

    def backward_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, forward=-self.tuning_scale)

    def left_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=self.tuning_scale)

    def right_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=-self.tuning_scale)

    def up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=self.tuning_scale)

    def down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=-self.tuning_scale)

    def yaw_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=self.tuning_scale)

    def yaw_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=-self.tuning_scale)

    def pitch_up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=self.tuning_scale)

    def pitch_down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=-self.tuning_scale)

    def roll_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=self.tuning_scale)

    def roll_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=-self.tuning_scale)

    def forward_frame_press(self, vis):
        self.current_frame_index = self.current_frame_index + 1
        self.current_frame_index = min(len(self.frames) - 1, self.current_frame_index)
        self.render_pc_and_boxes(vis)

    def backward_frame_press(self, vis):
        self.current_frame_index = self.current_frame_index - 1
        self.current_frame_index = max(0, self.current_frame_index)
        self.render_pc_and_boxes(vis)

    def shift_actions(self, vis, action, mods):
        actions = ["up", "down"]
        action = actions[action]
        if action == "down":
            self.tuning_scale = 0.02
        elif action == "up":
            self.tuning_scale = 0.1

    def on_mouse_move(self, vis, x, y):
        if self.prior_mouse_position is not None:
            dx = x - self.prior_mouse_position[0]
            dy = y - self.prior_mouse_position[1]
            view_control = vis.get_view_control()
            if self.is_view_rotating:
                view_control.rotate(
                    dx * self.pixel_to_rotate_scale_factor, dy * self.pixel_to_rotate_scale_factor
                )
            elif self.is_translating:
                view_control.translate(
                    dx * self.pixel_to_translate_scale_factor,
                    dy * self.pixel_to_translate_scale_factor,
                )

        self.prior_mouse_position = (x, y)

    def on_mouse_scroll(self, vis, x, y):
        view_control = vis.get_view_control()
        view_control.scale(y)

    def on_mouse_button(self, vis, button, action, mods):
        buttons = ["left", "right", "middle"]
        actions = ["up", "down"]
        mods_name = ["shift", "ctrl", "alt", "cmd"]

        button = buttons[button]
        action = actions[action]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]

        if button == "left" and action == "down":
            self.is_view_rotating = True
        elif button == "left" and action == "up":
            self.is_view_rotating = False
        elif button == "middle" and action == "down":
            self.is_translating = True
        elif button == "middle" and action == "up":
            self.is_translating = False
        elif button == "right" and action == "down":
            self.pick_mesh(vis, self.prior_mouse_position[0], self.prior_mouse_position[1])

        print(f"on_mouse_button: {button}, {action}, {mods}")
        if button == "right" and action == "down":
            self.pick_mesh(vis, self.prior_mouse_position[0], self.prior_mouse_position[1])

    def select_mesh(self, vis, mesh_id: str):
        self.selected_mesh_id = mesh_id
        if self.selection_axes is not None:
            vis.remove_geometry(self.selection_axes, reset_bounding_box=False)
        self.selection_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        # o3d.geometry.TriangleMesh.create_sphere(radius=1)

        selected_box_with_pose = self.clickable_geometries[mesh_id]
        center = selected_box_with_pose.pose.translation
        rotation_matrix = selected_box_with_pose.pose.rotation_matrix
        # Use the oriented bounding box center as the origin of the axes
        self.selection_axes.translate(center, relative=False)
        # Use the oriented bounding box rotation as the rotation of the axes
        self.selection_axes.rotate(rotation_matrix)
        vis.add_geometry(self.selection_axes, reset_bounding_box=False)
        self.render_selected_mesh_trajectory(vis)

    def deselect_mesh(self, vis):
        self.selected_mesh_id = None

        if self.selection_axes is not None:
            vis.remove_geometry(self.selection_axes, reset_bounding_box=False)
            self.selection_axes = None
            self.clear_trajectory_geometries(vis)

    def pick_mesh(self, vis, x, y, visualize_click: bool = False):
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic

        # Create a ray in camera space
        ray_camera = np.array(
            [
                (x - intrinsic[0, 2]) / intrinsic[0, 0],
                (y - intrinsic[1, 2]) / intrinsic[1, 1],
                1.0,
            ]
        )

        # Normalize the ray direction
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # Convert the ray to world space
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        ray_world = np.dot(rotation.T, ray_camera)
        ray_dir = ray_world / np.linalg.norm(ray_world)

        camera_pos = -np.dot(rotation.T, translation)

        if visualize_click:
            # Add sphere at camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(camera_pos)
            vis.add_geometry(sphere, reset_bounding_box=False)

            # Draw the ray in world space
            ray_end = camera_pos + ray_dir * 100  # Extend the ray 100 units
            ray_line = o3d.geometry.LineSet()
            ray_line.points = o3d.utility.Vector3dVector([camera_pos, ray_end])
            ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            ray_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            vis.add_geometry(ray_line, reset_bounding_box=False)

        closest_mesh_lookup: dict[str, float] = {}
        for id, box_with_pose in self.clickable_geometries.items():
            mesh = box_with_pose.triangle_mesh_o3d()
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            for tri in triangles:
                v0, v1, v2 = vertices[tri]
                hit, intersect_point = ray_triangle_intersect(camera_pos, ray_dir, v0, v1, v2)
                if hit:
                    intersection_distance = np.linalg.norm(intersect_point - camera_pos)
                    closest_mesh_lookup[id] = min(
                        intersection_distance, closest_mesh_lookup.get(id, np.inf)
                    )

        if len(closest_mesh_lookup) == 0:
            self.deselect_mesh(vis)
            return

        closest_mesh_id = min(closest_mesh_lookup, key=closest_mesh_lookup.get)
        print(f"Selected mesh: {closest_mesh_id}")
        self.selected_mesh_id = closest_mesh_id
        self.select_mesh(vis, closest_mesh_id)

    def rolling_window_range(self):
        """
        Calculate the range of frames to display
        """
        half_window_size = self.rolling_window_size // 2
        adjustment = 0 if self.rolling_window_size % 2 else 1
        start_index = max(0, self.current_frame_index - half_window_size - adjustment)
        end_index = min(len(self.frames), self.current_frame_index + half_window_size + 1)

        return start_index, end_index

    def clear_trajectory_geometries(self, vis):
        """
        Removes the trajectory geometries from the visualizer
        """
        for geometry in self.trajectory_geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)
        self.trajectory_geometries.clear()

    def render_selected_mesh_trajectory(self, vis):
        """
        Adds the trajectory geometries for the selected mesh
        """
        start_index, end_index = self.rolling_window_range()
        self.clear_trajectory_geometries(vis)

        if self.selected_mesh_id is not None:
            selected_box = self.clickable_geometries[self.selected_mesh_id].base_box
            selected_box_uuid = selected_box.track_uuid

            for i in range(start_index, end_index):
                if i == self.current_frame_index:
                    continue

                # Green color for other frames
                frame = self.frames[i]
                for box in frame.boxes:
                    if box.track_uuid == selected_box_uuid:
                        box_geom = BoxGeometryWithPose(box, color=[0, 0.5, 0])
                        wireframe = box_geom.wireframe_o3d()
                        vis.add_geometry(wireframe, reset_bounding_box=False)
                        self.trajectory_geometries.append(wireframe)

    def render_pc_and_boxes(self, vis, reset_bounding_box: bool = False):
        """
        Renders the point clouds and bounding boxes for the given frames
        """
        current_frame_index = self.current_frame_index
        start_index, end_index = self.rolling_window_range()

        vis.clear_geometries()
        # Loop over the frames and display pointclouds
        for i in range(start_index, end_index):
            frame = self.frames[i]
            pc_color = [1, 0, 0] if i == current_frame_index else [0.75, 0.75, 0.75]
            vis.add_geometry(
                frame.pc.global_pc.to_o3d().paint_uniform_color(pc_color),
                reset_bounding_box=reset_bounding_box,
            )

        # Render bounding boxes for the current frame only
        frame: TimeSyncedSceneFlowBoxFrame = self.frames[current_frame_index]
        # ego_translation = frame.pc.global_pose.translation
        for idx, (box, pose_info) in enumerate(frame.boxes.valid_boxes()):
            global_box = box.transform(pose_info.sensor_to_global)
            self.add_clickable_geometry(f"box{idx:06d}", BoxGeometryWithPose(global_box))
            vis.add_geometry(
                self.clickable_geometries[f"box{idx:06d}"].wireframe_o3d(),
                reset_bounding_box=reset_bounding_box,
            )


def load_box_frames() -> list[TimeSyncedSceneFlowBoxFrame]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Argoverse2NonCausalSceneFlow")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--sequence_id", type=str, required=True)
    args = parser.parse_args()

    sequence_length = args.sequence_length
    log_subset = [args.sequence_id]

    dataset = construct_dataset(
        name=args.dataset_name,
        args=dict(
            root_dir=args.root_dir,
            subsequence_length=sequence_length,
            with_ground=False,
            range_crop_type="ego",
            load_boxes=True,
            log_subset=log_subset,
        ),
    )
    assert len(dataset) == 1, f"Expected 1 sequence, got {len(dataset)}"

    return dataset[0]


def main(cropping_range=10000.0, rolling_window_size=5):
    """
    Args:
        rolling_window_size: The number of frames to display around the current frame
        cropping_range: The maximum distance from the ego vehicle within which to display bounding boxes
    """

    frames = load_box_frames()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    state_manager = ViewStateManager(frames, cropping_range, rolling_window_size)

    vis.register_mouse_move_callback(state_manager.on_mouse_move)
    vis.register_mouse_scroll_callback(state_manager.on_mouse_scroll)
    vis.register_mouse_button_callback(state_manager.on_mouse_button)

    # fmt: off
    # Use WASD keys for translation, Q and E keys for yaw
    vis.register_key_callback(ord("W"), state_manager.forward_press)
    vis.register_key_callback(ord("S"), state_manager.backward_press)
    vis.register_key_callback(ord("A"), state_manager.left_press)
    vis.register_key_callback(ord("D"), state_manager.right_press)
    vis.register_key_callback(ord("Z"), state_manager.down_press)
    vis.register_key_callback(ord("X"), state_manager.up_press)
    vis.register_key_callback(ord("Q"), state_manager.yaw_clockwise_press)
    vis.register_key_callback(ord("E"), state_manager.yaw_counterclockwise_press)
    # Use arrow keys for pitch and roll
    vis.register_key_callback(GLFW_KEY_UP, state_manager.pitch_up_press)
    vis.register_key_callback(GLFW_KEY_DOWN, state_manager.pitch_down_press)
    vis.register_key_callback(GLFW_KEY_RIGHT, state_manager.roll_clockwise_press)
    vis.register_key_callback(GLFW_KEY_LEFT, state_manager.roll_counterclockwise_press)
    # Use , and . keys for going forward and backward through the frames
    vis.register_key_callback(ord(","), lambda vis: state_manager.backward_frame_press(vis))
    vis.register_key_callback(ord("."), lambda vis: state_manager.forward_frame_press(vis))
    # Press Shift to fine tune
    vis.register_key_action_callback(GLFW_KEY_LEFT_SHIFT , state_manager.shift_actions)
    # fmt: on

    vis.create_window()
    state_manager.render_pc_and_boxes(vis, reset_bounding_box=True)

    render_option = vis.get_render_option()
    # render_option.mesh_show_wireframe = True
    # render_option.light_on = False
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    vis.run()


if __name__ == "__main__":

    print("Customized visualization with mouse action.")
    main()
