import math

import cv2
import numpy as np


class RGBImage:
    """
    RGBImage is a wrapper around a numpy array of shape (H, W, 3) representing an RGB image.

    Each pixel is assumed to be in the range [0, 1] as a float32.
    """

    def __init__(self, image: np.ndarray):
        assert len(image.shape) == 3, f"image must have shape (H, W, 3), got {image.shape}"
        assert image.shape[2] == 3, f"image must have shape (H, W, 3), got {image.shape}"

        assert (
            image.dtype == np.float32
        ), f"image must have dtype float32 or float64, got {image.dtype} with a min of {np.min(image)} and a max of {np.max(image)}"

        assert np.all(image >= 0) and np.all(
            image <= 1
        ), f"image must have values in range [0, 1], got min {np.min(image)} and max {np.max(image)}"

        self.image = image.astype(np.float32)

    @staticmethod
    def white_image(shape: tuple[int, int]) -> "RGBImage":
        assert len(shape) == 2, f"shape must be a 2-tuple, got {shape}"
        return RGBImage(np.ones(shape + (3,), dtype=np.float32))

    @staticmethod
    def white_image_like(image: "RGBImage") -> "RGBImage":
        return RGBImage.white_image(image.shape[:2])

    @staticmethod
    def black_image(shape: tuple[int, int]) -> "RGBImage":
        assert len(shape) == 2, f"shape must be a 2-tuple, got {shape}"
        return RGBImage(np.zeros(shape + (3,), dtype=np.float32))

    @staticmethod
    def black_image_like(image: "RGBImage") -> "RGBImage":
        return RGBImage.black_image(image.shape[:2])

    def __repr__(self) -> str:
        return f"RGBImage(shape={self.image.shape}, dtype={self.image.dtype})"

    def copy(self) -> "RGBImage":
        return RGBImage(self.image.copy())

    def rescale(self, reduction_factor: float) -> "RGBImage":
        new_shape = (
            int(math.ceil(self.image.shape[1] / reduction_factor)),
            int(math.ceil(self.image.shape[0] / reduction_factor)),
        )
        return RGBImage(cv2.resize(self.image, new_shape))

    @property
    def shape(self):
        return self.image.shape
