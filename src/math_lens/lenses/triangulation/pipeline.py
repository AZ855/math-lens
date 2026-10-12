"""Reusable image-to-Delaunay pipeline and saved experiment artifacts."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from math_lens.core.geometry import lloyd_relax_points
from math_lens.core.image_features import build_density_from_features, compute_edge_map
from math_lens.core.image_io import load_rgb_image, save_rgb_image
from math_lens.core.parameters import map_triangulation_controls
from math_lens.core.sampling import generate_adaptive_points, generate_uniform_points
from math_lens.core.visualization import make_image_panel
from math_lens.lenses.voronoi.renderer import overlay_points, render_voronoi_cells
from .renderer import compute_delaunay_triangles, render_triangulation, render_triangulation_mesh


TRIANGULATION_DEFAULTS = {
    "structural_resolution": 0.50,
    "regularity": 0.50,
    "structure_emphasis": 0.80,
    "mesh_visibility": 0.70,
}
TRIANGULATION_PRESETS = {
    "sparse": {
        "structural_resolution": 0.10,
        "regularity": 0.50,
        "structure_emphasis": 0.50,
        "mesh_visibility": 0.55,
    },
    "balanced": dict(TRIANGULATION_DEFAULTS),
    "structural": {
        "structural_resolution": 0.80,
        "regularity": 0.25,
        "structure_emphasis": 0.95,
        "mesh_visibility": 0.90,
    },
}


def _resolve_controls(
    structural_resolution: float | None,
    regularity: float | None,
    structure_emphasis: float | None,
    mesh_visibility: float | None,
    preset: str | None,
) -> dict[str, float]:
    controls = dict(TRIANGULATION_DEFAULTS)
    if preset is not None:
        if preset not in TRIANGULATION_PRESETS:
            raise ValueError(f"unknown triangulation preset: {preset}")
        controls.update(TRIANGULATION_PRESETS[preset])
    for name, value in (
        ("structural_resolution", structural_resolution),
        ("regularity", regularity),
        ("structure_emphasis", structure_emphasis),
        ("mesh_visibility", mesh_visibility),
    ):
        if value is not None:
            numeric = float(value)
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            controls[name] = numeric
    return controls


def _mesh_quality(points: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    vertices = points[triangles]
    angles = []
    for corner in range(3):
        first = vertices[:, (corner + 1) % 3] - vertices[:, corner]
        second = vertices[:, (corner + 2) % 3] - vertices[:, corner]
        cosine = (first * second).sum(axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    minimum_angles = np.min(angles, axis=0)
    return {
        "median_minimum_angle_degrees": float(np.median(minimum_angles)),
        "p10_minimum_angle_degrees": float(np.percentile(minimum_angles, 10)),
    }


def compute_triangulation_geometry(
    original: np.ndarray,
    sampling_mode: str = "adaptive",
    structural_resolution: float | None = None,
    regularity: float | None = None,
    structure_emphasis: float | None = None,
    mesh_visibility: float | None = None,
    seed: int = 10,
    preset: str | None = None,
    edge_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sample, relax, triangulate, and render one image entirely in memory.

    The existing sampler's [0, width) x [0, height) sites are scaled onto
    [0, width-1] x [0, height-1] pixel geometry. Four corner sites complete
    the rectangular hull and are pinned during Lloyd. All non-corner sites
    retain their index and count. The dual comparison receives this exact
    final array; it never invokes another sampler.
    """

    if original.ndim != 3 or original.shape[2] != 3 or original.dtype != np.uint8:
        raise ValueError("original must be a uint8 RGB image")
    height, width = original.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("triangulation requires image width and height >= 2")
    if sampling_mode not in {"uniform", "adaptive"}:
        raise ValueError("sampling_mode must be 'uniform' or 'adaptive'")
    controls = _resolve_controls(
        structural_resolution, regularity, structure_emphasis, mesh_visibility, preset
    )
    parameters = map_triangulation_controls(**controls)
    density = None
    if sampling_mode == "uniform":
        sampled = generate_uniform_points(width, height, parameters.total_points - 4, seed)
    else:
        edges = compute_edge_map(original) if edge_map is None else edge_map
        density = build_density_from_features(
            np.ones_like(edges), edges,
            edge_weight=parameters.structure_weight,
            tone_weight=parameters.uniform_weight,
        )
        sampled = generate_adaptive_points(
            density, parameters.n_coarse, parameters.n_fine, seed,
            blur_sigma=parameters.blur_sigma,
            density_power=parameters.density_power,
            jitter=0.0,
        )
    sampled = sampled.astype(np.float64) * [(width - 1) / width, (height - 1) / height]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    initial = np.concatenate((corners, sampled), axis=0)
    final = lloyd_relax_points(
        initial, width, height,
        iterations=parameters.lloyd_iterations,
        fixed_indices=(0, 1, 2, 3),
    )
    triangles = compute_delaunay_triangles(final)
    return {
        "original": original,
        "points_initial": initial,
        "points_relaxed": final,
        "triangles": triangles,
        "density": density,
        "controls": controls,
        "resolved_parameters": asdict(parameters),
        "mesh_quality": _mesh_quality(final, triangles),
        "mean_site_displacement_pixels": float(np.linalg.norm(final - initial, axis=1).mean()),
    }


def render_triangulation_result(
    original: np.ndarray,
    sampling_mode: str = "adaptive",
    structural_resolution: float | None = None,
    regularity: float | None = None,
    structure_emphasis: float | None = None,
    mesh_visibility: float | None = None,
    color_mode: str = "mean_vertex",
    seed: int = 10,
    render_mode: str = "filled_mesh",
    preset: str | None = None,
    include_voronoi_comparison: bool = True,
) -> dict[str, Any]:
    """Render one reusable geometry computation and its optional dual comparison."""

    if color_mode not in {"mean_vertex", "gouraud"}:
        raise ValueError("color_mode must be 'mean_vertex' or 'gouraud'")
    if render_mode not in {"mesh", "filled", "filled_mesh"}:
        raise ValueError("render_mode must be 'mesh', 'filled', or 'filled_mesh'")
    geometry = compute_triangulation_geometry(
        original, sampling_mode, structural_resolution, regularity,
        structure_emphasis, mesh_visibility, seed, preset,
    )
    initial = geometry["points_initial"]
    final = geometry["points_relaxed"]
    triangles = geometry["triangles"]
    parameters = map_triangulation_controls(**geometry["controls"])
    height, width = original.shape[:2]
    mesh = render_triangulation_mesh(
        final, triangles, width, height, line_width=parameters.line_width,
        line_alpha=1.0, show_points=True,
    )
    color = render_triangulation(
        original, final, triangles,
        color_mode=color_mode,
        render_mode=render_mode,
        line_width=parameters.line_width,
        line_alpha=parameters.line_alpha,
    )
    initial_overlay = overlay_points(original, initial)
    final_overlay = overlay_points(original, final)
    comparison = make_image_panel(
        [original, final_overlay, mesh, color],
        ["Original", "Final sites", "Delaunay mesh", "Colored triangulation" if render_mode != "mesh" else "Geometry only"],
        caption=f"{len(final):,} sites | {len(triangles):,} triangles | Lloyd {parameters.lloyd_iterations} steps | seed {seed}",
    )
    result = {
        **geometry,
        "initial_overlay": initial_overlay,
        "relaxed_overlay": final_overlay,
        "mesh": mesh,
        "color": color,
        "comparison": comparison,
    }
    if include_voronoi_comparison:
        # Both diagrams use final directly: no copy, re-sampling, or extra relaxation.
        voronoi = render_voronoi_cells(original, final, show_seeds=True)
        result["voronoi_vs_triangulation"] = make_image_panel(
            [voronoi, mesh], ["Voronoi: regions", "Delaunay: connections"],
            caption=f"Exactly the same {len(final):,} sites | final point set in points_relaxed.npy",
        )
    return result


def run_triangulation_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    sampling_mode: str = "adaptive",
    structural_resolution: float | None = None,
    regularity: float | None = None,
    structure_emphasis: float | None = None,
    mesh_visibility: float | None = None,
    color_mode: str = "mean_vertex",
    seed: int = 10,
    max_size: int | None = 1_000,
    render_mode: str = "filled_mesh",
    preset: str | None = None,
    include_voronoi_comparison: bool = True,
) -> dict[str, Any]:
    """Run the Triangulation Lens and save images, site arrays, and metadata."""

    original = load_rgb_image(input_path, max_size=max_size)
    result = render_triangulation_result(
        original, sampling_mode=sampling_mode,
        structural_resolution=structural_resolution, regularity=regularity,
        structure_emphasis=structure_emphasis, mesh_visibility=mesh_visibility,
        color_mode=color_mode, seed=seed, render_mode=render_mode,
        preset=preset, include_voronoi_comparison=include_voronoi_comparison,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image_names = {
        "original": "01_original.png",
        "initial_overlay": "02_points_initial.png",
        "relaxed_overlay": "03_points_relaxed.png",
        "mesh": "04_triangulation_mesh.png",
        "color": "05_triangulation_color.png",
        "comparison": "06_comparison.png",
    }
    if include_voronoi_comparison:
        image_names["voronoi_vs_triangulation"] = "07_voronoi_vs_triangulation.png"
    paths = {}
    for name, filename in image_names.items():
        path = destination / filename
        save_rgb_image(result[name], path)
        paths[name] = str(path)
    for name in ("points_initial", "points_relaxed", "triangles"):
        path = destination / f"{name}.npy"
        np.save(path, result[name])
        paths[name] = str(path)
    config_path = destination / "config.json"
    paths["config"] = str(config_path)
    height, width = original.shape[:2]
    config = {
        "lens": "triangulation",
        "input_path": str(input_path),
        "output_dir": str(destination),
        "sampling_mode": sampling_mode,
        "controls": result["controls"],
        "resolved_parameters": result["resolved_parameters"],
        "point_count": len(result["points_relaxed"]),
        "triangle_count": len(result["triangles"]),
        "lloyd_iterations": result["resolved_parameters"]["lloyd_iterations"],
        "lloyd_method": "unweighted exact centroids of rectangle-clipped Voronoi cells",
        "fixed_boundary_indices": [0, 1, 2, 3],
        "coordinate_system": "(x, y), x right, y down; [0, width-1] x [0, height-1]",
        "density_method": "uniform-to-Sobel crossfade" if sampling_mode == "adaptive" else None,
        "seed": seed,
        "width": width,
        "height": height,
        "max_size": max_size,
        "color_mode": color_mode,
        "render_mode": render_mode,
        "preset": preset,
        "same_point_voronoi_comparison": include_voronoi_comparison,
        "comparison_points": "points_relaxed.npy" if include_voronoi_comparison else None,
        "mesh_quality": result["mesh_quality"],
        "mean_site_displacement_pixels": result["mean_site_displacement_pixels"],
        "paths": paths,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config
