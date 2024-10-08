import numpy as np
from pyquaternion import Quaternion


class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation_matrix: np.ndarray, translation: np.ndarray) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.
        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)
        """
        assert rotation_matrix.shape == (3, 3)
        assert translation.shape == (3,)

        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = rotation_matrix
        self.transform_matrix[:3, 3] = translation

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.transform_matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.transform_matrix[:3, 3]

    @staticmethod
    def identity() -> "SE3":
        """Return the identity transformation."""
        return SE3(rotation_matrix=np.eye(3), translation=np.zeros(3))

    @staticmethod
    def from_rot_x_y_z_translation_x_y_z(rx, ry, rz, tx, ty, tz) -> "SE3":
        rotation_matrix = (
            Quaternion(axis=[1, 0, 0], angle=rx).rotation_matrix
            @ Quaternion(axis=[0, 1, 0], angle=ry).rotation_matrix
            @ Quaternion(axis=[0, 0, 1], angle=rz).rotation_matrix
        )
        translation = np.array([tx, ty, tz])
        return SE3(rotation_matrix, translation)

    @staticmethod
    def from_rot_w_x_y_z_translation_x_y_z(rw, rx, ry, rz, tx, ty, tz) -> "SE3":
        rotation_matrix = Quaternion(w=rw, x=rx, y=ry, z=rz).rotation_matrix
        translation = np.array([tx, ty, tz])
        return SE3(rotation_matrix, translation)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, SE3):
            return False
        return np.allclose(self.rotation_matrix, __value.rotation_matrix) and np.allclose(
            self.translation, __value.translation
        )

    def translate(self, translation: np.ndarray) -> "SE3":
        """Return a new SE3 instance with the given translation applied."""
        if isinstance(translation, list):
            translation = np.array(translation)
        assert translation.shape == (
            3,
        ), f"Translation must be a 3D vector, got {translation.shape}"
        return SE3(
            rotation_matrix=self.rotation_matrix,
            translation=self.translation + translation,
        )

    def scale(self, scale: float) -> "SE3":
        """Return a new SE3 instance with the given scale applied."""
        return SE3(
            rotation_matrix=self.rotation_matrix * scale,
            translation=self.translation * scale,
        )

    def transform_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(3) transformation to this point cloud.
        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`
        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return point_cloud @ self.rotation_matrix.T + self.translation

    def transform_flow(self, flow: np.ndarray) -> np.ndarray:
        """Apply the SE(3)'s rotation transformation to this flow field.
        Args:
            flow: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then flow should consist of flow vectors in frame `src`
        Returns:
            Array of shape (N, 3) representing the transformed flow field, i.e. flow vectors in frame `dst`
        """
        return flow @ self.rotation_matrix.T

    def inverse(self) -> "SE3":
        """Return the inverse of the current SE3 transformation.
        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.
        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        return SE3(
            rotation_matrix=self.rotation_matrix.T,
            translation=self.rotation_matrix.T.dot(-self.translation),
        )

    def compose(self, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.
        Algebraic representation: chained_se3 = T * right_se3
        Args:
            right_se3: another instance of SE3 class
        Returns:
            chained_se3: new instance of SE3 class
        """
        return SE3.from_array(self.transform_matrix @ right_se3.transform_matrix)

    def __matmul__(self, right_se3: "SE3") -> "SE3":
        return self.compose(right_se3)

    def to_array(self) -> np.ndarray:
        """Return the SE3 transformation matrix as a numpy array."""
        return self.transform_matrix

    @staticmethod
    def from_array(transform_matrix: np.ndarray) -> "SE3":
        """Initialize an SE3 instance from a numpy array."""
        return SE3(
            rotation_matrix=transform_matrix[:3, :3],
            translation=transform_matrix[:3, 3],
        )

    def __repr__(self) -> str:
        return f"SE3(rotation_matrix={self.rotation_matrix}, translation={self.translation})"

    def to_o3d(self, simple: bool = True):
        import open3d as o3d

        # Draw ball at origin
        origin_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        origin_ball = origin_ball.translate(self.translation)
        origin_ball.paint_uniform_color([0, 0, 0])

        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.1, height=0.5)
        point_forward = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        cone = cone.rotate(self.rotation_matrix @ point_forward, center=(0, 0, 0))
        cone = cone.translate(self.translation)
        cone = cone.compute_vertex_normals()

        if simple:
            return [origin_ball, cone]

        forward_vec = np.array([1, 0, 0])
        forward_rotated_vec = self.rotation_matrix @ forward_vec
        left_vec = np.array([0, 1, 0])
        left_rotated_vec = self.rotation_matrix @ left_vec
        up_vec = np.array([0, 0, 1])
        up_rotated_vec = self.rotation_matrix @ up_vec

        # Draw ball at unit length in x direction
        forward_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        forward_ball = forward_ball.translate(self.translation + forward_rotated_vec)
        forward_ball.paint_uniform_color([1, 0, 0])

        left_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        left_ball = left_ball.translate(self.translation + left_rotated_vec)
        left_ball.paint_uniform_color([0, 1, 0])

        up_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        up_ball = up_ball.translate(self.translation + up_rotated_vec)
        up_ball.paint_uniform_color([0, 0, 1])

        # Draw line between balls
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(
            np.vstack((self.translation, self.translation + forward_rotated_vec))
        )
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))

        return [origin_ball, forward_ball, left_ball, up_ball, line, cone]
