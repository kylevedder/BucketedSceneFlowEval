import numpy as np
import pytest

from bucketed_scene_flow_eval.datastructures import RGBImage, RGBImageCrop


@pytest.fixture
def cropped_square() -> RGBImage:
    # Create a 100x100 image with a red square in the middle
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[25:75, 25:75] = [1, 0, 0]
    mask = RGBImageCrop(25, 25, 75, 75)
    return RGBImage(image, mask)


def test_masked_image_extract(cropped_square: RGBImage):
    center_image = cropped_square.masked_image

    assert center_image.shape == (50, 50, 3), f"expected 50x50x3, got {center_image.shape}"
    assert np.all(center_image.full_image == [1, 0, 0]), f"expected all red, got {center_image}"
