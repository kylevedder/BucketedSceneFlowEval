import numpy as np
import open3d as o3d
from annotation_saver import AnnotationSaver
from scipy.spatial.transform import Rotation as R

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    PoseInfo,
    TimeSyncedSceneFlowBoxFrame,
)
from bucketed_scene_flow_eval.utils.glfw_key_ids import *


def _update_o3d_mesh_pose(mesh: o3d.geometry.TriangleMesh, start_pose: SE3, target_pose: SE3):
    global_translation = target_pose.translation - start_pose.translation
    global_rotation = target_pose.rotation_matrix @ np.linalg.inv(start_pose.rotation_matrix)

    mesh.translate(global_translation)
    mesh.rotate(global_rotation, center=target_pose.translation)


class RenderableBox:
    def __init__(self, base_box: BoundingBox, pose: PoseInfo, color: list[float] = [0.1, 0.1, 0.1]):
        self.base_box = base_box
        self.pose = pose
        self.color = color

        # O3D doesn't support rendering boxers as wireframes directly, so we create a box and its associated rendered lineset.
        self.o3d_triangle_mesh = o3d.geometry.TriangleMesh.create_box(
            width=base_box.length, height=base_box.height, depth=base_box.width
        )
        self.o3d_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.o3d_triangle_mesh)
        self.imit_pose_of_o3d_geomerty(self.pose.sensor_to_global)
        self.set_color(self.color)

    def imit_pose_of_o3d_geomerty(self, pose: SE3):
        o3d_geom_centering_translation = -np.array(
            [0.5 * self.base_box.length, 0.5 * self.base_box.height, 0.5 * self.base_box.width]
        )

        center_offset_se3 = SE3.identity().translate(o3d_geom_centering_translation)
        o3d_target_pose = pose.compose(center_offset_se3)

        _update_o3d_mesh_pose(self.o3d_wireframe, SE3.identity(), o3d_target_pose)
        _update_o3d_mesh_pose(self.o3d_triangle_mesh, SE3.identity(), o3d_target_pose)

    def compute_new_global_pose_from_inputs(
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

        return self.pose.sensor_to_global.compose(local_frame_offset_se3)

    def update_from_global(self, global_target_se3: SE3):
        _update_o3d_mesh_pose(self.o3d_wireframe, self.pose.sensor_to_global, global_target_se3)
        _update_o3d_mesh_pose(self.o3d_triangle_mesh, self.pose.sensor_to_global, global_target_se3)
        # We only edit sensor to ego; ego to global stays fixed for the box
        # because the ego vehicle position never changes.
        self.pose.sensor_to_ego = self.pose.ego_to_global.inverse().compose(global_target_se3)

    def triangle_mesh_o3d(self) -> o3d.geometry.TriangleMesh:
        return self.o3d_triangle_mesh

    def wireframe_o3d(self) -> o3d.geometry.LineSet:
        return self.o3d_wireframe

    def set_color(self, color: list[float]):
        """
        Sets the color of the geomerty

        Args:
            color: A list or array of three floats representing the RGB color.
        """
        num_lines = len(self.o3d_wireframe.lines)
        self.o3d_wireframe.colors = o3d.utility.Vector3dVector([color] * num_lines)


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
    def __init__(
        self,
        frames: list[TimeSyncedSceneFlowBoxFrame],
        annotation_saver: AnnotationSaver,
        rolling_window_size: int,
    ) -> None:
        self.prior_mouse_position: tuple[float, float] | None = None
        self.is_view_rotating = False
        self.is_translating = False
        self.pixel_to_rotate_scale_factor = 1
        self.pixel_to_translate_scale_factor = 1
        self.clickable_geometries: dict[str, RenderableBox] = {}
        self.selection_axes: o3d.geometry.TriangleMesh | None = None
        self.selected_mesh_id: str | None = None
        self.current_frame_index = 0
        self.tuning_scale = 0.1
        self.annotation_saver = annotation_saver
        self.frames = frames
        self.rolling_window_size = rolling_window_size
        self.trajectory_geometries: list[RenderableBox] = []
        # the two blow is used for zoom
        self.is_zoomed = False
        self.original_view = None
        # Used for velocity
        self.propagate_with_velocity = False
        self.velocities = {}
        # Used for toggle box
        self.current_box_index = -1

    def add_clickable_geometry(self, id: str, box_geometry: RenderableBox):
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
        global_target_se3 = selected_mesh.compute_new_global_pose_from_inputs(
            forward=forward, left=left, up=up, pitch=pitch, yaw=yaw, roll=roll
        )
        # for g in global_target_se3.to_o3d():
        #     vis.add_geometry(g, reset_bounding_box=False)

        _update_o3d_mesh_pose(
            self.selection_axes, selected_mesh.pose.sensor_to_global, global_target_se3
        )
        selected_mesh.update_from_global(global_target_se3)

        vis.update_geometry(selected_mesh.wireframe_o3d())
        vis.update_geometry(self.selection_axes)

    def forward_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, forward=self.tuning_scale)
        print("forward_press")

    def backward_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, forward=-self.tuning_scale)
        print("backward_press")

    def left_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=self.tuning_scale)
        print("left_press")

    def right_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=-self.tuning_scale)
        print("right_press")

    def up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=self.tuning_scale)
        print("up_press")

    def down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=-self.tuning_scale)
        print("down_press")

    def yaw_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=self.tuning_scale)
        print("yaw_clockwise_press")

    def yaw_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=-self.tuning_scale)
        print("yaw_counterclockwise_press")

    def pitch_up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=self.tuning_scale)
        print("pitch_up_press")

    def pitch_down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=-self.tuning_scale)
        print("pitch_down_press")

    def roll_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=self.tuning_scale)
        print("roll_clockwise_press")

    def roll_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=-self.tuning_scale)
        print("roll_counterclockwise_press")

    def forward_frame_press(self, vis):
        # self.annotation_saver.save(self.frames)
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.render_pc_and_boxes(vis)
            if self.propagate_with_velocity:
                self.apply_velocity()
            self.render_pc_and_boxes(vis)

    def backward_frame_press(self, vis):
        # self.annotation_saver.save(self.frames)
        self.current_frame_index = self.current_frame_index - 1
        self.current_frame_index = max(0, self.current_frame_index)
        self.render_pc_and_boxes(vis)

    def shift_actions(self, vis, action, mods):
        actions = ["release", "press", "repeat"]
        action = actions[action]
        if action == "press":
            self.tuning_scale = 0.02
        elif action == "release":
            self.tuning_scale = 0.1

    def key_S_actions(self, vis, action, mods):
        actions = ["release", "press", "repeat"]
        mods_name = ["shift", "ctrl", "alt", "cmd"]
        action = actions[action]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]

        if action == "press" or action == "repeat":
            if mods == []:
                self.backward_press(vis)
            elif mods == ["shift"]:
                self.tuning_scale = 0.02
                self.backward_press(vis)
                self.tuning_scale = 0.1
            elif mods == ["ctrl"]:
                self.annotation_saver.save(self.frames)

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
        actions = ["up", "down", "drag"]
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
        center = selected_box_with_pose.pose.sensor_to_global.translation
        rotation_matrix = selected_box_with_pose.pose.sensor_to_global.rotation_matrix
        # Use the oriented bounding box center as the origin of the axes
        self.selection_axes.translate(center, relative=False)
        # Use the oriented bounding box rotation as the rotation of the axes
        self.selection_axes.rotate(rotation_matrix)
        vis.add_geometry(self.selection_axes, reset_bounding_box=False)
        # self.render_selected_mesh_trajectory(vis)

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
                frame: TimeSyncedSceneFlowBoxFrame = self.frames[i]
                for box, pose_info in frame.boxes.valid_boxes():
                    if box.track_uuid == selected_box_uuid:
                        box_geom = RenderableBox(box, pose_info, color=[0, 0.5, 0])
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
        self.clickable_geometries = {}
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
        for idx, (box, pose_info) in enumerate(frame.boxes.valid_boxes()):
            self.add_clickable_geometry(f"box{idx:06d}", RenderableBox(box, pose_info))
            vis.add_geometry(
                self.clickable_geometries[f"box{idx:06d}"].wireframe_o3d(),
                reset_bounding_box=reset_bounding_box,
            )

        # Add axes at the origin
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        vis.add_geometry(axes, reset_bounding_box=reset_bounding_box)

    def get_annotations(self) -> list[dict]:
        """
        Collect the current annotations.
        Returns:
            A list of dictionaries, each representing an annotation.
        """
        annotations = []
        for box_id, box in self.clickable_geometries.items():
            global_pose = box.pose.sensor_to_global
            translation = global_pose.transform_matrix[:3, 3]
            rotation = R.from_matrix(global_pose.transform_matrix[:3, :3])
            annotations.append(
                {
                    "id": box_id,
                    "translation": translation,
                    "rotation": rotation.as_quat(),  # (x,y,z,w) quaternion, may change as need
                    "dimensions": (box.base_box.length, box.base_box.width, box.base_box.height),
                    "category": box.base_box.category,
                    "track_uuid": box.base_box.track_uuid,
                    # Add other relevant attributes
                }
            )
        return annotations

    def zoom_press(self, vis, action, mods):
        """
        Callback function for zoom to box.
        """
        actions = ["release", "press", "repeat"]
        mods_name = ["shift", "ctrl", "alt", "cmd"]
        action = actions[action]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]

        if action == "press":
            ctr = vis.get_view_control()
            if mods == ["ctrl"] and self.original_view is not None:
                ctr.convert_from_pinhole_camera_parameters(self.original_view)
                self.is_zoomed = False
            elif mods == [] and self.selected_mesh_id:
                if not self.is_zoomed:
                    self.original_view = ctr.convert_to_pinhole_camera_parameters()
                    self.is_zoomed = True
                self.zoom_to_box(vis)

    def zoom_to_box(self, vis):
        assert self.selected_mesh_id is not None, "No box selected. Cannot zoom to box."

        ctr = vis.get_view_control()
        selected_box = self.clickable_geometries[self.selected_mesh_id]
        box_center = selected_box.pose.sensor_to_global.translation

        ctr.set_lookat(box_center)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.1)

    def toggle_box(self, vis, action, mods):
        """
        Toggle the selected box.
        """
        actions = ["release", "press", "repeat"]
        mods_name = ["shift", "ctrl", "alt", "cmd"]
        action = actions[action]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]
        if action == "press":
            box_indices = sorted(
                self.clickable_geometries.keys(),
                key=lambda k: self.clickable_geometries[k].pose.sensor_to_global.translation[0],
            )

            if self.selected_mesh_id:
                try:
                    self.current_box_index = box_indices.index(self.selected_mesh_id)
                except ValueError:
                    print(f"Selected box {self.selected_mesh_id} not found in box indices.")
                    self.current_box_index = -1

            if mods == []:
                self.current_box_index = (self.current_box_index + 1) % len(box_indices)
            elif mods == ["ctrl"]:
                self.current_box_index = (self.current_box_index - 1) % len(box_indices)

            new_mesh_id = box_indices[self.current_box_index]
            self.selected_mesh_id = new_mesh_id
            self.select_mesh(vis, new_mesh_id)
            print(f"Selected mesh: {new_mesh_id}")
            if self.is_zoomed == True:
                self.zoom_to_box(vis)

            # print(f"Toggled to box {new_mesh_id}")

    def toggle_propagate_with_velocity(self, vis):
        """
        Toggle the propagate_with_velocity feature on and off.
        """
        self.propagate_with_velocity = not self.propagate_with_velocity
        print(f"Propagate with velocity: {'On' if self.propagate_with_velocity else 'Off'}")
        if self.propagate_with_velocity:
            self.compute_velocities()

    def compute_velocities(self):
        """
        Compute velocities for all the meshes.
        """
        if self.current_frame_index < 2:
            return  # Need at least two frames to calculate velocity
        self.velocities.clear()

        current_frame = self.frames[self.current_frame_index]
        prev_frame = self.frames[self.current_frame_index - 1]
        prev_prev_frame = self.frames[self.current_frame_index - 2]

        for box, pose in current_frame.boxes.valid_boxes():
            uuid = box.track_uuid
            prev_pose = self.find_pose_in_frame(prev_frame, uuid)
            prev_prev_pose = self.find_pose_in_frame(prev_prev_frame, uuid)

            if prev_pose and prev_prev_pose:
                velocity = self.calculate_velocity(prev_prev_pose, prev_pose)
                self.velocities[uuid] = velocity
            else:
                self.velocities[uuid] = np.zeros(3)  # No movement if no corresponding box found

    def find_pose_in_frame(self, frame, track_uuid: str):
        """
        Find the pose of the box according to track_uuid
        """
        for box, pose in frame.boxes.valid_boxes():
            if box.track_uuid == track_uuid:
                return pose
        return None

    def calculate_velocity(self, pose1: PoseInfo, pose2: PoseInfo) -> np.ndarray:
        """
        Calculate displacement between two poses (suppose time interval=1 and use it as velocity).
        """
        translation1 = pose1.sensor_to_global.translation
        translation2 = pose2.sensor_to_global.translation
        velocity = translation2 - translation1  # Update the pose of the RenderableBox
        return velocity

    def apply_velocity(self):
        """
        Apply stored velocities to the boxes in the current frame.
        """
        self.compute_velocities()

        current_frame = self.frames[self.current_frame_index]
        last_frame = self.frames[self.current_frame_index - 1]

        for key, renderable_box in self.clickable_geometries.items():
            # for box, pose in current_frame.boxes.valid_boxes():
            uuid = renderable_box.base_box.track_uuid
            if uuid in self.velocities:
                last_pose = self.find_pose_in_frame(last_frame, uuid)
                if last_pose:
                    # Update the current pose based on the last pose and velocity
                    new_translation = last_pose.sensor_to_global.translation + self.velocities[uuid]
                    # Create a new SE3 object for the new global pose
                    new_global_pose = SE3(
                        rotation_matrix=last_pose.sensor_to_global.rotation_matrix,
                        translation=new_translation,
                    )

                    renderable_box.update_from_global(new_global_pose)
