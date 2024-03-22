import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class RGBImageCrop:
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    def __post_init__(self):
        assert (
            self.min_x < self.max_x
        ), f"min_x must be less than max_x, got {self.min_x} and {self.max_x}"
        assert (
            self.min_y < self.max_y
        ), f"min_y must be less than max_y, got {self.min_y} and {self.max_y}"
        # Ensure that the crop is non-negative
        assert self.min_x >= 0, f"min_x must be non-negative, got {self.min_x}"
        assert self.min_y >= 0, f"min_y must be non-negative, got {self.min_y}"

    @staticmethod
    def from_full_image(image: "RGBImage") -> "RGBImageCrop":
        return RGBImageCrop(0, 0, image.full_image.shape[1], image.full_image.shape[0])

    def apply_to_image(self, image: "RGBImage") -> "RGBImage":
        return RGBImage(image.full_image[self.min_y : self.max_y, self.min_x : self.max_x])

    def get_is_valid_mask(self, image: "RGBImage") -> np.ndarray:
        mask = np.zeros(image.full_image.shape[:2], dtype=bool)
        mask[self.min_y : self.max_y, self.min_x : self.max_x] = True
        return mask

    def resize(self, reduction_factor: float) -> "RGBImageCrop":
        return RGBImageCrop(
            int(math.floor(self.min_x / reduction_factor)),
            int(math.floor(self.min_y / reduction_factor)),
            int(math.ceil(self.max_x / reduction_factor)),
            int(math.ceil(self.max_y / reduction_factor)),
        )


class RGBImage:
    """
    RGBImage is a wrapper around a numpy array of shape (H, W, 3) representing an RGB image.

    Each pixel is assumed to be in the range [0, 1] as a float32.
    """

    def __init__(self, full_image: np.ndarray, valid_crop: Optional[RGBImageCrop] = None):
        assert (
            len(full_image.shape) == 3
        ), f"image must have shape (H, W, 3), got {full_image.shape}"
        assert full_image.shape[2] == 3, f"image must have shape (H, W, 3), got {full_image.shape}"

        assert (
            full_image.dtype == np.float32
        ), f"image must have dtype float32 or float64, got {full_image.dtype} with a min of {np.min(full_image)} and a max of {np.max(full_image)}"

        assert np.all(full_image >= 0) and np.all(
            full_image <= 1
        ), f"image must have values in range [0, 1], got min {np.min(full_image)} and max {np.max(full_image)}"

        self.full_image = full_image.astype(np.float32)

        if valid_crop is None:
            self.valid_crop = RGBImageCrop.from_full_image(self)
        else:
            assert isinstance(
                valid_crop, RGBImageCrop
            ), f"valid_crop must be an RGBImageCrop, got {type(valid_crop)}"
            self.valid_crop = valid_crop

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
        return f"RGBImage(shape={self.full_image.shape}, dtype={self.full_image.dtype})"

    def copy(self) -> "RGBImage":
        return RGBImage(self.full_image.copy())

    def rescale(self, reduction_factor: float) -> "RGBImage":
        new_shape = (
            int(math.ceil(self.full_image.shape[1] / reduction_factor)),
            int(math.ceil(self.full_image.shape[0] / reduction_factor)),
        )
        new_img = cv2.resize(self.full_image, new_shape)
        valid_crop = None
        if self.valid_crop is not None:
            valid_crop = self.valid_crop.resize(reduction_factor)
        return RGBImage(new_img, valid_crop)

    @property
    def masked_image(self) -> "RGBImage":
        return self.valid_crop.apply_to_image(self)

    def get_is_valid_mask(self) -> np.ndarray:
        return self.valid_crop.get_is_valid_mask(self)

    @property
    def shape(self):
        return self.full_image.shape
