"""SciPy Delaunay connectivity, geometry drawing, and vertex-color rendering.

Delaunay is the geometric dual of the Voronoi diagram for the same sites:
neighboring Voronoi cells induce Delaunay edges, and triangles describe local
neighborhoods. Their open circumcircles contain no other sites. Cocircular
sites can admit several valid triangulations; SciPy chooses the connectivity.
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, QhullError

from math_lens.core.geometry import validate_points_xy
from .encodings import build_edge_encoding


def compute_delaunay_triangles(points_xy: np.ndarray) -> np.ndarray:
    """Return SciPy's simplices, retaining original point indices and sites."""

    points = validate_points_xy(points_xy)
    try:
        triangulation = Delaunay(points)
    except QhullError as error:
        raise ValueError("Delaunay requires at least three non-collinear sites") from error
    if len(np.unique(triangulation.simplices)) != len(points):
        raise ValueError("Delaunay omitted sites due to geometric degeneracy")
    return triangulation.simplices.astype(np.int32, copy=True)


def _validate_triangles(triangles: np.ndarray, n_points: int) -> np.ndarray:
    faces = np.asarray(triangles)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("triangles must have shape (M, 3) with M >= 1")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("triangle indices must be integers")
    if np.any(faces < 0) or np.any(faces >= n_points):
        raise ValueError("triangle indices must refer to valid sites")
    return faces


def _mesh_overlay(
    background: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    line_width: float,
    line_alpha: float,
    show_points: bool,
) -> np.ndarray:
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError("line_width must be finite and greater than zero")
    if not math.isfinite(line_alpha) or not 0 <= line_alpha <= 1:
        raise ValueError("line_alpha must lie in [0, 1]")
    height, width = background.shape[:2]
    scale = 3
    mask = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(mask)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    for first, second in points[edges] * scale:
        draw.line((tuple(first), tuple(second)), fill=255, width=max(1, round(line_width * scale)))
    if show_points:
        radius = 1.1 * scale
        for x, y in points * scale:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    mask = mask.resize((width, height), Image.Resampling.LANCZOS)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0 * line_alpha
    return np.round(background * (1.0 - alpha) + 30.0 * alpha).astype(np.uint8)


def render_triangulation_mesh(
    points_xy: np.ndarray,
    triangles: np.ndarray,
    width: int,
    height: int,
    line_width: float = 1.0,
    line_alpha: float = 1.0,
    show_points: bool = True,
) -> np.ndarray:
    """Draw the skeleton on white using image coordinates: x right, y down."""

    if width < 1 or height < 1:
        raise ValueError("mesh image dimensions must be positive")
    points = validate_points_xy(points_xy)
    faces = _validate_triangles(triangles, len(points))
    background = np.full((height, width, 3), 255, dtype=np.uint8)
    return _mesh_overlay(background, points, faces, line_width, line_alpha, show_points)


def render_encoded_mesh(
    image: np.ndarray,
    points_xy: np.ndarray,
    triangles: np.ndarray,
    edge_color_mode: str = "Monochrome",
    width_channel: str = "Uniform",
    min_width: float = 0.8,
    max_width: float = 2.8,
    uniform_width: float = 1.2,
    line_alpha: float = 1.0,
) -> np.ndarray:
    """Draw independently encoded edges on white; triangle interiors stay unfilled."""

    if edge_color_mode not in {"Monochrome", "Local Image Color"}:
        raise ValueError("edge_color_mode must be Monochrome or Local Image Color")
    if not math.isfinite(line_alpha) or not 0 <= line_alpha <= 1:
        raise ValueError("line_alpha must lie in [0, 1]")
    points = validate_points_xy(points_xy)
    encoding = build_edge_encoding(
        image, points, triangles, width_channel, min_width, max_width, uniform_width
    )
    height, width = image.shape[:2]
    scale = 3
    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)
    for edge, color, edge_width in zip(encoding.edges, encoding.colors, encoding.widths):
        rgb = color if edge_color_mode == "Local Image Color" else np.full(3, 30.)
        fill = tuple(np.round(255.0 * (1.0 - line_alpha) + rgb * line_alpha).astype(int))
        first, second = points[edge] * scale
        draw.line((tuple(first), tuple(second)), fill=fill, width=max(1, round(edge_width * scale)))
    return np.asarray(canvas.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8).copy()


def render_triangulation(
    rgb_uint8: np.ndarray,
    points_xy: np.ndarray,
    triangles: np.ndarray,
    color_mode: str = "mean_vertex",
    render_mode: str = "filled_mesh",
    line_width: float = 1.0,
    line_alpha: float = 0.7,
) -> np.ndarray:
    """Rasterize flat mean-vertex or barycentrically interpolated RGB colors.

    Colors are sampled at image[floor(y), floor(x)] with boundary clipping.
    mean_vertex averages the three samples, not the pixels inside a triangle.
    gouraud interpolates those RGB samples with the triangle's barycentric
    weights. Pixel coordinates are evaluated directly, so y is never flipped.
    Mesh edges can be omitted, overlaid, or rendered without any image color.
    """

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3 or rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must be a uint8 RGB image")
    if color_mode not in {"mean_vertex", "gouraud"}:
        raise ValueError("color_mode must be 'mean_vertex' or 'gouraud'")
    if render_mode not in {"mesh", "filled", "filled_mesh"}:
        raise ValueError("render_mode must be 'mesh', 'filled', or 'filled_mesh'")
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError("line_width must be finite and greater than zero")
    if not math.isfinite(line_alpha) or not 0 <= line_alpha <= 1:
        raise ValueError("line_alpha must lie in [0, 1]")
    points = validate_points_xy(points_xy)
    faces = _validate_triangles(triangles, len(points))
    height, width = rgb_uint8.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("rgb_uint8 must have nonempty dimensions")
    if render_mode == "mesh":
        return render_triangulation_mesh(points, faces, width, height, line_width, 1.0)
    indices = np.floor(points).astype(np.int64)
    indices[:, 0] = np.clip(indices[:, 0], 0, width - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, height - 1)
    vertex_colors = rgb_uint8[indices[:, 1], indices[:, 0]].astype(np.float64)
    output = np.full_like(rgb_uint8, 255)

    for face in faces:
        vertices = points[face]
        x_min, y_min = np.maximum(np.ceil(vertices.min(axis=0)).astype(int), 0)
        x_max, y_max = np.minimum(np.floor(vertices.max(axis=0)).astype(int), [width - 1, height - 1])
        if x_max < x_min or y_max < y_min:
            continue
        first, second, third = vertices
        denominator = (
            (second[1] - third[1]) * (first[0] - third[0])
            + (third[0] - second[0]) * (first[1] - third[1])
        )
        if abs(denominator) <= np.finfo(np.float64).eps:
            raise ValueError("triangles must have nonzero area")
        xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        weight_first = (
            (second[1] - third[1]) * (xs - third[0])
            + (third[0] - second[0]) * (ys - third[1])
        ) / denominator
        weight_second = (
            (third[1] - first[1]) * (xs - third[0])
            + (first[0] - third[0]) * (ys - third[1])
        ) / denominator
        weight_third = 1.0 - weight_first - weight_second
        inside = (weight_first >= -1e-9) & (weight_second >= -1e-9) & (weight_third >= -1e-9)
        if color_mode == "mean_vertex":
            colors = vertex_colors[face].mean(axis=0)
        else:
            weights = np.stack((weight_first[inside], weight_second[inside], weight_third[inside]), axis=1)
            colors = weights @ vertex_colors[face]
        patch = output[y_min:y_max + 1, x_min:x_max + 1]
        patch[inside] = np.round(np.clip(colors, 0, 255)).astype(np.uint8)
    if render_mode == "filled_mesh" and line_alpha > 0:
        output = _mesh_overlay(output, points, faces, line_width, line_alpha, False)
    return output
