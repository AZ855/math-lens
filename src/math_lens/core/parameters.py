"""Mappings from interaction-level controls to algorithm parameters."""

from dataclasses import dataclass
import math


def clamp01(value: float) -> float:
    """Clamp a finite numeric value to the closed interval ``[0, 1]``."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("value must be finite")
    return max(0.0, min(1.0, numeric))


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate from ``a`` to ``b`` by fraction ``t``."""

    return float(a) + (float(b) - float(a)) * float(t)


@dataclass(frozen=True)
class VoronoiLensParameters:
    """Low-level parameters resolved from the Voronoi Lens controls."""

    total_points: int
    n_coarse: int
    n_fine: int
    blur_sigma: float
    density_power: float
    edge_weight: float
    tone_weight: float


def map_voronoi_lens_controls(
    abstraction: float,
    detail_emphasis: float,
    structure: float,
) -> VoronoiLensParameters:
    """Map three normalized visual controls to adaptive Voronoi parameters.

    The mapping is intentionally linear and inspectable. Increasing abstraction
    reduces the number of cells and broadens the coarse density. Increasing
    detail emphasis moves sites into the fine sampler and sharpens density
    peaks. Structure directly crossfades from luminance to edge density.
    """

    abstraction = clamp01(abstraction)
    detail_emphasis = clamp01(detail_emphasis)
    structure = clamp01(structure)

    total_points = round(lerp(5_000, 500, abstraction))
    blur_sigma = round(lerp(2.0, 14.0, abstraction), 12)

    fine_ratio = lerp(0.40, 0.90, detail_emphasis)
    n_fine = round(total_points * fine_ratio)
    n_coarse = total_points - n_fine
    density_power = round(lerp(1.0, 4.0, detail_emphasis), 12)

    return VoronoiLensParameters(
        total_points=total_points,
        n_coarse=n_coarse,
        n_fine=n_fine,
        blur_sigma=blur_sigma,
        density_power=density_power,
        edge_weight=round(structure, 12),
        tone_weight=round(1.0 - structure, 12),
    )


@dataclass(frozen=True)
class TriangulationLensParameters:
    """Resolved mesh, shared-sampler, and drawing parameters.

    Four corner anchors are included in total_points. n_coarse + n_fine counts
    only the interior sites; the anchors stay fixed during Lloyd relaxation.
    """

    total_points: int
    lloyd_iterations: int
    structure_weight: float
    uniform_weight: float
    n_coarse: int
    n_fine: int
    blur_sigma: float
    density_power: float
    line_width: float
    line_alpha: float


def map_triangulation_controls(
    structural_resolution: float,
    regularity: float,
    structure_emphasis: float,
    mesh_visibility: float,
) -> TriangulationLensParameters:
    """Map normalized controls with explicit linear interpolation.

    Resolution: round(400 + 4600*r) sites, including four fixed corners.
    Regularity: round(4*r) unweighted bounded Lloyd iterations.
    Structure: crossfade uniform density to existing Sobel, fine-site share
    40% -> 90%, and density power 1 -> 4; coarse blur stays at 6 pixels.
    Visibility: width 0.5 -> 2 image pixels and opacity 0 -> 1.
    """

    resolution = clamp01(structural_resolution)
    regularity = clamp01(regularity)
    structure = clamp01(structure_emphasis)
    visibility = clamp01(mesh_visibility)
    total = round(lerp(400, 5_000, resolution))
    interior = total - 4
    fine = round(interior * lerp(0.40, 0.90, structure))
    return TriangulationLensParameters(
        total_points=total,
        lloyd_iterations=round(lerp(0, 4, regularity)),
        structure_weight=structure,
        uniform_weight=round(1.0 - structure, 12),
        n_coarse=interior - fine,
        n_fine=fine,
        blur_sigma=6.0,
        density_power=round(lerp(1.0, 4.0, structure), 12),
        line_width=round(lerp(0.5, 2.0, visibility), 12),
        line_alpha=visibility,
    )
