"""Independent local-color and RGB-width encodings for Delaunay edges."""

from dataclasses import dataclass
import math

import numpy as np

from math_lens.core.geometry import validate_points_xy
from math_lens.core.parameters import clamp01


@dataclass(frozen=True)
class EdgeEncoding:
    edges: np.ndarray
    adjacent_triangles: np.ndarray
    colors: np.ndarray
    widths: np.ndarray


def mesh_visibility_style(visibility: float) -> dict[str, float]:
    """Explorer opacity/width bounds retain visible lines at the subtle endpoint."""
    value = clamp01(visibility)
    return {
        "uniform_width": .8 + 1.2 * value,
        "min_width": .75 + .25 * value,
        "max_width": 2.0 + value,
        "line_alpha": .55 + .45 * value,
    }


def triangle_edge_adjacency(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique sorted edges and up to two incident triangle indices (-1 = none)."""

    faces = np.asarray(triangles)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("triangles must have nonempty shape (M, 3)")
    if not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0):
        raise ValueError("triangle indices must be non-negative integers")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("triangle vertices must be distinct")
    all_edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse = np.unique(np.sort(all_edges, axis=1), axis=0, return_inverse=True)
    adjacent = np.full((len(edges), 2), -1, dtype=np.int32)
    for edge_index, triangle_index in zip(inverse, np.repeat(np.arange(len(faces)), 3)):
        slot = 0 if adjacent[edge_index, 0] == -1 else 1
        if adjacent[edge_index, slot] != -1:
            raise ValueError("an edge cannot belong to more than two triangles")
        adjacent[edge_index, slot] = triangle_index
    return edges, adjacent


def triangle_centroid_colors(
    image: np.ndarray, points_xy: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Sample I(floor(centroid.y), floor(centroid.x)), clamped to image bounds."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB")
    if min(image.shape[:2]) < 1:
        raise ValueError("image dimensions must be nonempty")
    points = validate_points_xy(points_xy)
    faces = np.asarray(triangles)
    triangle_edge_adjacency(faces)
    if faces.max() >= len(points):
        raise ValueError("triangle indices must refer to valid points")
    centers = points[faces].mean(axis=1)
    xs = np.clip(np.floor(centers[:, 0]), 0, image.shape[1] - 1).astype(int)
    ys = np.clip(np.floor(centers[:, 1]), 0, image.shape[0] - 1).astype(int)
    return image[ys, xs].astype(np.float64)


def representative_edge_colors(
    centroid_colors: np.ndarray, adjacent_triangles: np.ndarray
) -> np.ndarray:
    """Average two incident centroid colors, or use the boundary triangle's color."""

    colors = np.asarray(centroid_colors, dtype=np.float64)
    adjacent = np.asarray(adjacent_triangles)
    if colors.ndim != 2 or colors.shape[1] != 3 or not np.isfinite(colors).all():
        raise ValueError("centroid_colors must be finite RGB rows")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("centroid colors must lie in [0, 255]")
    if adjacent.ndim != 2 or adjacent.shape[1] != 2 or not np.issubdtype(adjacent.dtype, np.integer):
        raise ValueError("adjacent_triangles must contain two integer indices per edge")
    if np.any(adjacent[:, 0] < 0) or np.any(adjacent >= len(colors)) or np.any(adjacent[:, 1] < -1):
        raise ValueError("invalid triangle adjacency")
    first = colors[adjacent[:, 0]]
    second = colors[np.where(adjacent[:, 1] >= 0, adjacent[:, 1], adjacent[:, 0])]
    return (first + second) / 2.0


def rgb_edge_widths(
    edge_colors: np.ndarray,
    channel: str = "Uniform",
    min_width: float = 0.8,
    max_width: float = 2.8,
    uniform_width: float = 1.2,
) -> np.ndarray:
    """Linear channel/255 width mapping with a positive visible lower bound."""

    colors = np.asarray(edge_colors, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3 or not np.isfinite(colors).all():
        raise ValueError("edge_colors must be finite RGB rows")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("edge colors must lie in [0, 255]")
    if any(not math.isfinite(value) or value < 0.5 for value in (min_width, max_width, uniform_width)):
        raise ValueError("line widths must be finite and at least 0.5 pixels")
    if max_width < min_width:
        raise ValueError("max_width must be >= min_width")
    if channel == "Uniform":
        return np.full(len(colors), uniform_width, dtype=np.float64)
    if channel not in {"R", "G", "B"}:
        raise ValueError("channel must be Uniform, R, G, or B")
    return min_width + colors[:, "RGB".index(channel)] / 255.0 * (max_width - min_width)


def build_edge_encoding(
    image: np.ndarray, points_xy: np.ndarray, triangles: np.ndarray,
    channel: str = "Uniform", min_width: float = 0.8, max_width: float = 2.8,
    uniform_width: float = 1.2,
) -> EdgeEncoding:
    edges, adjacent = triangle_edge_adjacency(triangles)
    centroid_colors = triangle_centroid_colors(image, points_xy, triangles)
    colors = representative_edge_colors(centroid_colors, adjacent)
    widths = rgb_edge_widths(colors, channel, min_width, max_width, uniform_width)
    return EdgeEncoding(edges, adjacent, colors, widths)
