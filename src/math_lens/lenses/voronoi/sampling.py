"""Compatibility imports for the original Voronoi sampling API."""

from math_lens.core.sampling import (
    generate_adaptive_points,
    generate_uniform_points,
    sample_points_from_pdf,
)

__all__ = [
    "generate_adaptive_points",
    "generate_uniform_points",
    "sample_points_from_pdf",
]
