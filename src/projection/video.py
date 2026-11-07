import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def normalize_to_uint8(E, vmax):
    """Scale E from 0..vmax to 0..255 (decision 11, docs/decisions.md) and return a uint8 array."""
    if vmax <= 0:
        return np.zeros(E.shape, dtype=np.uint8)
    # Float on purpose: multiplying a uint16 array by 255 would overflow silently.
    scaled = np.rint(E.astype(float) * 255 / vmax)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def canvas_shape(shapes):
    """Return the (rows, cols) canvas that fits every shape, with even sides (decision 11, docs/decisions.md)."""
    shapes = list(shapes)
    if not shapes:
        return (256, 256)
    rows = max(r for r, _ in shapes)
    cols = max(c for _, c in shapes)
    # H.264 needs even sides.
    return (rows + rows % 2, cols + cols % 2)


def pad_to_canvas(img, shape):
    """Put img on a black canvas: top-aligned, centered horizontally, never resized."""
    rows, cols = img.shape
    if rows > shape[0] or cols > shape[1]:
        raise ValueError(f"image {img.shape} does not fit in canvas {shape}")
    canvas = np.zeros(shape, dtype=img.dtype)
    left = (shape[1] - cols) // 2
    canvas[:rows, left:left + cols] = img
    return canvas


def error_frame(shape, text):
    """Black canvas with white text, wrapped to the canvas width."""
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    margin = 4
    # Greedy word wrap: add words to the line while it still fits.
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width - 2 * margin:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)

    line_height = font.getbbox("Ag")[3] + 2
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_height), line, fill=255, font=font)
    return np.array(image, dtype=np.uint8)


def write_video(frames, path, fps):
    """Write uint8 frames (2D gray or 3D color) to an H.264 mp4; return the frame count."""
    writer = None
    shape = None
    count = 0
    try:
        for frame in frames:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if writer is None:
                shape = frame.shape
                if shape[0] % 2 or shape[1] % 2:
                    raise ValueError(f"frame sides must be even, got {shape[:2]}")
                # macro_block_size=1 stops imageio from silently resizing to a multiple of 16.
                writer = imageio.get_writer(
                    str(path), fps=fps, codec="libx264", macro_block_size=1
                )
            elif frame.shape != shape:
                raise ValueError(f"frame {count} has shape {frame.shape}, expected {shape}")
            writer.append_data(frame)
            count += 1
    finally:
        if writer is not None:
            writer.close()
    if count == 0:
        raise ValueError("no frames to write")
    return count
