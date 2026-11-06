from io import BytesIO

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from src.utils.image_processor import load_and_preprocess_image


@pytest.mark.parametrize(
    ("extension", "image_format"),
    [("jpg", "JPEG"), ("jpeg", "JPEG"), ("png", "PNG")],
)
def test_image_formats_are_preprocessed_as_rgb(extension, image_format, tmp_path):
    source = Image.new("RGB", (18, 12), color=(0, 128, 255))
    path = tmp_path / f"leaf.{extension}"
    source.save(path, format=image_format)
    seen = []

    def preprocess(values):
        seen.append(values.copy())
        return values / 127.5 - 1.0

    result = load_and_preprocess_image(path, preprocess_input=preprocess)

    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32
    assert seen[0].shape == (1, 224, 224, 3)
    assert seen[0].min() >= 0
    assert seen[0].max() <= 255
    assert result.min() >= -1
    assert result.max() <= 1


def test_grayscale_input_is_converted_to_three_channel_rgb():
    grayscale = Image.new("L", (20, 20), color=128)
    result = load_and_preprocess_image(
        grayscale, preprocess_input=lambda values: values
    )

    assert result.shape == (1, 224, 224, 3)
    np.testing.assert_array_equal(result[0, :, :, 0], result[0, :, :, 1])
    np.testing.assert_array_equal(result[0, :, :, 1], result[0, :, :, 2])


@pytest.mark.parametrize("contents", [b"", b"not an image"])
def test_empty_and_corrupted_images_raise_clear_pillow_errors(contents):
    with pytest.raises(UnidentifiedImageError):
        load_and_preprocess_image(BytesIO(contents))


def test_missing_image_path_is_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image(tmp_path / "missing.jpg")


def test_empty_or_unsupported_image_input_is_rejected():
    with pytest.raises(TypeError, match="chemin"):
        load_and_preprocess_image(None)
    with pytest.raises(ValueError, match="target_size"):
        load_and_preprocess_image(
            Image.new("RGB", (1, 1)),
            target_size=(0, 224),
            preprocess_input=lambda values: values,
        )
