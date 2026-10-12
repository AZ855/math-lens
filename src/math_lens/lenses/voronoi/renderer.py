"""Nearest-site partitioning and rendering for the Voronoi Lens."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def _validate_points(points_xy: np.ndarray) -> None:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (n_points, 2)")
    if points_xy.shape[0] == 0:
        raise ValueError("points_xy must contain at least one point")
    if not np.isfinite(points_xy).all():
        raise ValueError("points_xy must contain only finite coordinates")


def _validate_labels(labels: np.ndarray, width: int, height: int) -> None:
    if labels.shape != (height, width):
        raise ValueError("labels must have shape (height, width)")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must have an integer dtype")


def voronoi_label_map(
    points_xy: np.ndarray,
    width: int,
    height: int,
    chunk: int = 200_000,
) -> np.ndarray:
    """Assign every image pixel to its nearest seed.

    The returned integer array has shape ``(height, width)``. A value ``i`` at
    ``labels[y, x]`` means pixel ``(x, y)`` belongs to the Voronoi cell whose
    generator is ``points_xy[i]``.
    """

    _validate_points(points_xy)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    if chunk <= 0:
        raise ValueError("chunk must be greater than zero")

    tree = cKDTree(points_xy.astype(np.float32, copy=False))
    xs = np.arange(width, dtype=np.int32)
    ys = np.arange(height, dtype=np.int32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    coords = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1).astype(np.float32)
    labels = np.empty(coords.shape[0], dtype=np.int32)

    # Chunking preserves the direct nearest-site definition without allocating
    # the much larger matrix of every pixel-to-every-site distance.
    for start in range(0, coords.shape[0], chunk):
        stop = min(start + chunk, coords.shape[0])
        _, indices = tree.query(coords[start:stop], k=1)
        labels[start:stop] = indices.astype(np.int32)

    return labels.reshape(height, width)


def render_voronoi_mean_color(
    rgb_uint8: np.ndarray,
    points_xy: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Fill each Voronoi cell with its source pixels' mean RGB color."""

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must have shape (height, width, 3)")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must have dtype uint8")
    _validate_points(points_xy)

    height, width = rgb_uint8.shape[:2]
    if labels is None:
        labels = voronoi_label_map(points_xy, width, height)
    else:
        _validate_labels(labels, width, height)
    image_flat = rgb_uint8.reshape(-1, 3).astype(np.float32)
    label_flat = labels.reshape(-1)
    n_sites = points_xy.shape[0]
    counts = np.zeros(n_sites, dtype=np.float32)
    sums = np.zeros((n_sites, 3), dtype=np.float32)

    # These indexed sums integrate color over each mathematical Voronoi cell.
    np.add.at(counts, label_flat, 1.0)
    np.add.at(sums, label_flat, image_flat)

    means = sums / (counts[:, None] + 1e-12)
    means_uint8 = np.clip(means, 0, 255).astype(np.uint8)
    return means_uint8[label_flat].reshape(height, width, 3)


def render_voronoi_seed_color(
    rgb_uint8: np.ndarray,
    points_xy: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Fill each Voronoi cell with the source color at its generating site.

    For site ``p_i``, the cell color is ``C_i = I(p_i)``. The partition itself
    is identical to mean-color rendering; only the regional color rule changes.
    """

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must have shape (height, width, 3)")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must have dtype uint8")
    _validate_points(points_xy)

    height, width = rgb_uint8.shape[:2]
    if labels is None:
        labels = voronoi_label_map(points_xy, width, height)
    else:
        _validate_labels(labels, width, height)
    site_x = np.clip(np.floor(points_xy[:, 0]), 0, width - 1).astype(np.intp)
    site_y = np.clip(np.floor(points_xy[:, 1]), 0, height - 1).astype(np.intp)
    site_colors = rgb_uint8[site_y, site_x]
    return site_colors[labels].astype(np.uint8, copy=False)


def render_voronoi(
    rgb_uint8: np.ndarray,
    points_xy: np.ndarray,
    color_mode: str = "mean",
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Render Voronoi cells using the requested regional color rule."""

    if color_mode == "mean":
        return render_voronoi_mean_color(rgb_uint8, points_xy, labels=labels)
    if color_mode == "seed":
        return render_voronoi_seed_color(rgb_uint8, points_xy, labels=labels)
    raise ValueError(f"Unsupported Voronoi color mode: {color_mode}")


def render_voronoi_cells(
    rgb_uint8: np.ndarray,
    points_xy: np.ndarray,
    show_seeds: bool = True,
    line_value: int = 70,
    background_value: int = 245,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Render the nearest-neighbor partition as a clean cell diagram.

    Cell edges are pixels whose horizontal or vertical neighbor has a different
    nearest-site label. This visualizes the existing discrete Voronoi partition
    directly; no polygon construction or additional geometry algorithm is used.
    """

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must have shape (height, width, 3)")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must have dtype uint8")
    _validate_points(points_xy)
    if not 0 <= line_value <= 255 or not 0 <= background_value <= 255:
        raise ValueError("line_value and background_value must be in [0, 255]")

    height, width = rgb_uint8.shape[:2]
    if labels is None:
        labels = voronoi_label_map(points_xy, width, height)
    else:
        _validate_labels(labels, width, height)

    boundary = np.zeros((height, width), dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical

    diagram = np.full((height, width, 3), background_value, dtype=np.uint8)
    diagram[boundary] = line_value
    if show_seeds:
        canvas = Image.fromarray(diagram, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        for x_float, y_float in points_xy:
            x = int(np.clip(np.floor(x_float), 0, width - 1))
            y = int(np.clip(np.floor(y_float), 0, height - 1))
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(20, 20, 20))
        diagram = np.asarray(canvas, dtype=np.uint8).copy()
    return diagram


def overlay_points(
    image: np.ndarray,
    points_xy: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Return a copy of an RGB image with white seed-point markers."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 array shaped (height, width, 3)")
    _validate_points(points_xy)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    height, width = image.shape[:2]
    canvas = Image.fromarray(image.copy(), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for x_float, y_float in points_xy:
        # A continuous seed is visualized at the pixel containing that point.
        x = int(np.clip(np.floor(x_float), 0, width - 1))
        y = int(np.clip(np.floor(y_float), 0, height - 1))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255),
        )
    return np.asarray(canvas, dtype=np.uint8).copy()
