"""Shared infrastructure for Math Lens image pipelines."""

from .image_features import (
    build_density_map,
    build_density_from_features,
    compute_edge_map,
    compute_density_features,
    compute_luminance,
    normalize_01,
)
from .image_io import load_rgb_bytes, load_rgb_image, save_rgb_image
from .parameters import (
    TriangulationLensParameters,
    VoronoiLensParameters,
    clamp01,
    lerp,
    map_voronoi_lens_controls,
    map_triangulation_controls,
)

__all__ = [
    "build_density_map",
    "build_density_from_features",
    "compute_density_features",
    "compute_edge_map",
    "compute_luminance",
    "load_rgb_image",
    "load_rgb_bytes",
    "VoronoiLensParameters",
    "TriangulationLensParameters",
    "clamp01",
    "lerp",
    "map_voronoi_lens_controls",
    "map_triangulation_controls",
    "normalize_01",
    "save_rgb_image",
]
