import imageio.v2 as imageio
import numpy as np
import pytest

from projection.video import (
    canvas_shape,
    error_frame,
    normalize_to_uint8,
    pad_to_canvas,
    write_video,
)


def test_normalize_uses_vmax_and_rounds():
    # 68 * 255 / 279 = 62.15, which rounds to 62.
    E = np.array([[0, 68, 279]], dtype=np.uint16)
    out = normalize_to_uint8(E, 279)

    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 62, 255]]


def test_normalize_does_not_overflow_on_uint16():
    # 65535 * 255 does not fit in uint16; the result must still be 255.
    E = np.array([[65535]], dtype=np.uint16)
    assert normalize_to_uint8(E, 65535)[0, 0] == 255


def test_normalize_clips_values_above_vmax():
    E = np.array([[1000.0]])
    assert normalize_to_uint8(E, 500)[0, 0] == 255


def test_normalize_with_zero_vmax_is_black():
    assert not normalize_to_uint8(np.zeros((2, 2)), 0).any()


def test_canvas_shape_is_max_rows_and_cols():
    assert canvas_shape([(10, 40), (30, 20), (8, 8)]) == (30, 40)


def test_canvas_shape_rounds_up_to_even():
    assert canvas_shape([(499, 400)]) == (500, 400)
    assert canvas_shape([(10, 401)]) == (10, 402)


def test_canvas_shape_of_empty_list_is_256_by_256():
    assert canvas_shape([]) == (256, 256)


def test_pad_keeps_content_top_aligned_and_centered():
    img = np.full((2, 4), 200, dtype=np.uint8)
    out = pad_to_canvas(img, (6, 10))

    assert out.shape == (6, 10)
    # Content sits in rows 0-1 and columns 3-6; (10 - 4) // 2 = 3.
    assert np.array_equal(out[0:2, 3:7], img)
    # Everything else is black.
    assert out.sum() == img.sum()


def test_pad_with_odd_free_space_puts_the_extra_column_on_the_right():
    img = np.full((1, 2), 9, dtype=np.uint8)
    out = pad_to_canvas(img, (2, 5))  # 3 free columns: 1 left, 2 right

    assert out[0].tolist() == [0, 9, 9, 0, 0]


def test_pad_does_not_change_the_input():
    img = np.full((2, 2), 7, dtype=np.uint8)
    pad_to_canvas(img, (4, 4))

    assert np.array_equal(img, np.full((2, 2), 7))


def test_pad_rejects_an_image_larger_than_the_canvas():
    with pytest.raises(ValueError, match="does not fit"):
        pad_to_canvas(np.zeros((5, 5), dtype=np.uint8), (4, 4))


def test_error_frame_has_the_right_size_and_some_white_pixels():
    frame = error_frame((64, 128), "row 3: not isosceles")

    assert frame.shape == (64, 128)
    assert frame.dtype == np.uint8
    assert frame.any()


def test_error_frame_wraps_long_text_to_more_lines():
    # A long message wraps, so it reaches lower rows than a short one.
    short = error_frame((200, 100), "error")
    long = error_frame((200, 100), "error " * 20)

    assert np.nonzero(long.any(axis=1))[0].max() > np.nonzero(short.any(axis=1))[0].max()


def test_write_then_read_back_data_frame_and_error_frame(tmp_path):
    # 30 x 50 is even but not a multiple of 16, so a silent resize would show up.
    shape = (30, 50)
    data = pad_to_canvas(np.full((20, 30), 255, dtype=np.uint8), shape)
    error = error_frame(shape, "row 2: bad plane")
    path = tmp_path / "out.mp4"

    assert write_video([data, error], path, fps=2) == 2

    frames = imageio.mimread(str(path), memtest=False)
    assert len(frames) == 2
    assert frames[0].shape == (30, 50, 3)
    assert frames[1].shape == (30, 50, 3)
    # H.264 is lossy, so only check that bright stays bright and black stays dark.
    assert frames[0][5, 25].mean() > 200
    assert frames[0][25, 2].mean() < 50


def test_write_video_rejects_odd_frame_size(tmp_path):
    frame = np.zeros((31, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="even"):
        write_video([frame], tmp_path / "odd.mp4", fps=1)


def test_write_video_rejects_frames_of_different_size(tmp_path):
    frames = [np.zeros((30, 50), dtype=np.uint8), np.zeros((30, 52), dtype=np.uint8)]
    with pytest.raises(ValueError, match="expected"):
        write_video(frames, tmp_path / "mixed.mp4", fps=1)


def test_write_video_rejects_no_frames(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        write_video([], tmp_path / "empty.mp4", fps=1)
