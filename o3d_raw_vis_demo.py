from dataclasses import dataclass

import numpy as np
import open3d as o3d

# Import scipy rotation
from scipy.spatial.transform import Rotation as R

from bucketed_scene_flow_eval.datastructures import SE3
from bucketed_scene_flow_eval.utils.glfw_key_ids import *


@dataclass
class BoxSize:
    length: float
    width: float
    height: float


def _update_o3d_mesh_pose(mesh: o3d.geometry.TriangleMesh, start_pose: SE3, target_pose: SE3):
    # Compute the transformation matrix from the start pose to the target pose
    delta_se3 = start_pose.inverse().compose(target_pose)
    # Transform the mesh
    delta_translation = delta_se3.translation
    delta_rotation = delta_se3.rotation_matrix

    global_translation = target_pose.translation - start_pose.translation
    global_rotation = target_pose.rotation_matrix @ np.linalg.inv(start_pose.rotation_matrix)

    mesh.translate(global_translation)
    mesh.rotate(global_rotation, center=target_pose.translation)


class BoxGeometryWithPose:
    def __init__(self, pose: SE3, box_size: BoxSize):
        self.pose = pose
        self.box_size = box_size

        # O3D doesn't support rendering boxers as wireframes directly, so we create a box and its associated rendered lineset.
        self.o3d_triangle_mesh = o3d.geometry.TriangleMesh.create_box(
            width=box_size.length, height=box_size.height, depth=box_size.width
        )
        self.o3d_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.o3d_triangle_mesh)
        self.imit_pose_of_o3d_geomerty(pose)

    def imit_pose_of_o3d_geomerty(self, pose: SE3):
        o3d_geom_centering_translation = -np.array(
            [0.5 * self.box_size.length, 0.5 * self.box_size.height, 0.5 * self.box_size.width]
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

        return self.pose.compose(local_frame_offset_se3)

    def update_from_global(self, global_se3: SE3):
        _update_o3d_mesh_pose(self.o3d_wireframe, self.pose, global_se3)
        _update_o3d_mesh_pose(self.o3d_triangle_mesh, self.pose, global_se3)
        self.pose = global_se3

    def triangle_mesh_o3d(self) -> o3d.geometry.TriangleMesh:
        return self.o3d_triangle_mesh

    def wireframe_o3d(self) -> o3d.geometry.LineSet:
        return self.o3d_wireframe


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
    def __init__(self) -> None:
        self.prior_mouse_position: tuple[float, float] | None = None
        self.is_view_rotating = False
        self.is_translating = False
        self.pixel_to_rotate_scale_factor = 1
        self.pixel_to_translate_scale_factor = 1
        self.clickable_geometries: dict[str, BoxGeometryWithPose] = {}
        self.selection_axes: o3d.geometry.TriangleMesh | None = None
        self.selected_mesh_id: str | None = None

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
        self._update_selection(vis, forward=0.1)

    def backward_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, forward=-0.1)

    def left_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=0.1)

    def right_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, left=-0.1)

    def up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=0.1)

    def down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, up=-0.1)

    def yaw_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=0.1)

    def yaw_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, yaw=-0.1)

    def pitch_up_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=0.1)

    def pitch_down_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, pitch=-0.1)

    def roll_clockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=0.1)

    def roll_counterclockwise_press(self, vis):
        if self.selected_mesh_id is None:
            return
        self._update_selection(vis, roll=-0.1)

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

    def deselect_mesh(self, vis):
        self.selected_mesh_id = None

        if self.selection_axes is not None:
            vis.remove_geometry(self.selection_axes, reset_bounding_box=False)
            self.selection_axes = None

    def pick_mesh(self, vis, x, y):
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


def custom_mouse_action(pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()

    state_manager = ViewStateManager()

    box1 = BoxGeometryWithPose(
        SE3.from_rot_x_y_z_translation_x_y_z(0, 0, 0, 3, 2, 1),
        box_size=BoxSize(1.5, 3.0, 1.0),
    )
    box2 = BoxGeometryWithPose(
        SE3.identity().translate([10.0, 0.0, 2.0]),
        box_size=BoxSize(1.5, 0.2, 1.0),
    )

    state_manager.add_clickable_geometry("box1", box1)
    state_manager.add_clickable_geometry("box2", box2)

    vis.register_mouse_move_callback(state_manager.on_mouse_move)
    vis.register_mouse_scroll_callback(state_manager.on_mouse_scroll)
    vis.register_mouse_button_callback(state_manager.on_mouse_button)

    # fmt: off
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
    # fmt: on

    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(box1.wireframe_o3d())
    vis.add_geometry(box2.wireframe_o3d())
    # Add a coordinate frame at the origin
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5))
    render_option = vis.get_render_option()
    # render_option.mesh_show_wireframe = True
    # render_option.light_on = False
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    vis.run()


if __name__ == "__main__":
    ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_data.path)

    print("Customized visualization with mouse action.")
    custom_mouse_action(pcd)
