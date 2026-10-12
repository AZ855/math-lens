"""End-to-end orchestration for the three Voronoi sampling interfaces."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from math_lens.core.image_features import (
    SVDToneResult,
    build_density_from_features,
    build_density_map,
    compute_edge_map,
    compute_luminance,
    compute_svd_tone,
    normalize_01,
)
from math_lens.core.image_io import load_rgb_image, save_rgb_image
from math_lens.core.parameters import map_voronoi_lens_controls
from math_lens.lenses.voronoi.renderer import (
    overlay_points,
    render_voronoi,
    render_voronoi_cells,
    voronoi_label_map,
)
from math_lens.lenses.voronoi.sampling import (
    generate_adaptive_points,
    generate_uniform_points,
)


def _density_to_rgb(density: np.ndarray) -> np.ndarray:
    """Convert a density map into a visible uint8 grayscale RGB image."""

    grayscale = np.round(normalize_01(density) * 255.0).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _unit_map_to_rgb(feature: np.ndarray) -> np.ndarray:
    """Convert a nominally unit-range feature into grayscale RGB."""

    grayscale = np.round(np.clip(feature, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _validate_normalized_control(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric


def render_lens_result(
    original: np.ndarray,
    abstraction: float = 0.5,
    detail_emphasis: float = 0.5,
    structure: float = 0.8,
    color_mode: str = "mean",
    seed: int = 10,
    luminance_normalized: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    tone_method: str = "svd",
    svd_energy_threshold: float = 0.90,
    svd_tone_result: SVDToneResult | None = None,
    include_cells: bool = False,
) -> dict[str, Any]:
    """Compute one high-level Lens result in memory, without saving files.

    This is shared by the CLI pipeline and the interactive demo so both use
    exactly the same mapping, density, sampling, and rendering operations.
    """

    abstraction = _validate_normalized_control("abstraction", abstraction)
    detail_emphasis = _validate_normalized_control(
        "detail_emphasis", detail_emphasis
    )
    structure = _validate_normalized_control("structure", structure)
    if color_mode not in {"mean", "seed"}:
        raise ValueError("color_mode must be 'mean' or 'seed'")
    if tone_method not in {"svd", "luminance"}:
        raise ValueError("tone_method must be 'svd' or 'luminance'")

    parameters = map_voronoi_lens_controls(
        abstraction, detail_emphasis, structure
    )
    if edge_map is None:
        edge_map = compute_edge_map(original)
    if tone_method == "svd":
        if svd_tone_result is None:
            svd_tone_result = compute_svd_tone(original, svd_energy_threshold)
        tone_map = svd_tone_result.tone_map
    else:
        svd_tone_result = None
        if luminance_normalized is None:
            luminance_normalized = normalize_01(compute_luminance(original))
        tone_map = luminance_normalized

    density = build_density_from_features(
        tone_map,
        edge_map,
        mode="mix",
        edge_weight=parameters.edge_weight,
        tone_weight=parameters.tone_weight,
    )
    points = generate_adaptive_points(
        density,
        n_coarse=parameters.n_coarse,
        n_fine=parameters.n_fine,
        seed=seed,
        blur_sigma=parameters.blur_sigma,
        density_power=parameters.density_power,
        jitter=0.0,
    )
    labels = None
    if include_cells:
        height, width = original.shape[:2]
        labels = voronoi_label_map(points, width, height)
    result = {
        "original": original,
        "density": density,
        "density_rgb": _density_to_rgb(density),
        "tone_map": tone_map,
        "tone_map_rgb": _unit_map_to_rgb(tone_map),
        "structure_map": edge_map,
        "structure_map_rgb": _unit_map_to_rgb(edge_map),
        "points_overlay": overlay_points(original, points),
        "voronoi": render_voronoi(
            original, points, color_mode=color_mode, labels=labels
        ),
        "points": points,
        "controls": {
            "abstraction": abstraction,
            "detail_emphasis": detail_emphasis,
            "structure": structure,
            "color_mode": color_mode,
        },
        "resolved_parameters": asdict(parameters),
        "feature_metadata": {
            "tone_method": tone_method,
            "svd_energy_threshold": float(svd_energy_threshold),
            "svd_rank": (
                svd_tone_result.rank if svd_tone_result is not None else None
            ),
            "svd_retained_energy": (
                svd_tone_result.retained_energy
                if svd_tone_result is not None
                else None
            ),
            "svd_centered": tone_method == "svd",
            "svd_luminance_mean": (
                svd_tone_result.luminance_mean
                if svd_tone_result is not None
                else None
            ),
        },
    }
    if svd_tone_result is not None:
        result["svd_reconstruction"] = svd_tone_result.reconstruction
        result["svd_centered_reconstruction"] = (
            svd_tone_result.centered_reconstruction
        )
        result["svd_reconstruction_rgb"] = _unit_map_to_rgb(
            svd_tone_result.reconstruction
        )
    if include_cells:
        assert labels is not None
        result["voronoi_cells"] = render_voronoi_cells(
            original, points, show_seeds=True, labels=labels
        )
    return result


def run_voronoi_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    sampling_mode: str = "uniform",
    n_points: int = 1_000,
    n_coarse: int = 400,
    n_fine: int = 1_600,
    seed: int = 0,
    max_size: int | None = 1_000,
    density_mode: str = "mix",
    edge_weight: float = 0.85,
    tone_weight: float = 0.15,
    blur_sigma: float = 6.0,
    density_power: float = 2.0,
    point_jitter: float = 0.0,
    abstraction: float = 0.5,
    detail_emphasis: float = 0.5,
    structure: float = 0.8,
    color_mode: str = "mean",
    preset: str | None = None,
    tone_method: str = "svd",
    svd_energy_threshold: float = 0.90,
) -> dict[str, Any]:
    """Run the uniform, low-level adaptive, or high-level Lens pipeline."""

    if sampling_mode not in {"uniform", "adaptive", "lens"}:
        raise ValueError("sampling_mode must be 'uniform', 'adaptive', or 'lens'")
    if color_mode not in {"mean", "seed"}:
        raise ValueError("color_mode must be 'mean' or 'seed'")
    if tone_method not in {"svd", "luminance"}:
        raise ValueError("tone_method must be 'svd' or 'luminance'")
    if preset is not None and sampling_mode != "lens":
        raise ValueError("presets can only be used with sampling_mode='lens'")

    source_path = Path(input_path)
    destination = Path(output_dir)
    original = load_rgb_image(source_path, max_size=max_size)
    height, width = original.shape[:2]
    density: np.ndarray | None = None
    lens_result: dict[str, Any] | None = None
    controls: dict[str, float | str] | None = None
    resolved_parameters: dict[str, int | float] | None = None
    feature_metadata: dict[str, Any] = {
        "tone_method": tone_method,
        "svd_energy_threshold": float(svd_energy_threshold),
        "svd_rank": None,
        "svd_retained_energy": None,
        "svd_centered": tone_method == "svd",
        "svd_luminance_mean": None,
    }

    if sampling_mode == "uniform":
        points = generate_uniform_points(width, height, n_points, seed)
    elif sampling_mode == "adaptive":
        if density_mode == "mix" and tone_method == "svd":
            svd_tone = compute_svd_tone(original, svd_energy_threshold)
            density = build_density_from_features(
                svd_tone.tone_map,
                compute_edge_map(original),
                mode="mix",
                edge_weight=edge_weight,
                tone_weight=tone_weight,
            )
            feature_metadata.update(
                {
                    "svd_rank": svd_tone.rank,
                    "svd_retained_energy": svd_tone.retained_energy,
                    "svd_luminance_mean": svd_tone.luminance_mean,
                }
            )
        else:
            density = build_density_map(
                original,
                mode=density_mode,
                edge_weight=edge_weight,
                tone_weight=tone_weight,
                tone_method=tone_method,
                svd_energy_threshold=svd_energy_threshold,
            )
        points = generate_adaptive_points(
            density,
            n_coarse=n_coarse,
            n_fine=n_fine,
            seed=seed,
            blur_sigma=blur_sigma,
            density_power=density_power,
            jitter=point_jitter,
        )
    else:
        lens_result = render_lens_result(
            original,
            abstraction=abstraction,
            detail_emphasis=detail_emphasis,
            structure=structure,
            color_mode=color_mode,
            seed=seed,
            tone_method=tone_method,
            svd_energy_threshold=svd_energy_threshold,
        )
        density = lens_result["density"]
        points = lens_result["points"]
        controls = lens_result["controls"]
        resolved_parameters = lens_result["resolved_parameters"]
        feature_metadata = lens_result["feature_metadata"]

    point_overlay = (
        lens_result["points_overlay"] if lens_result is not None
        else overlay_points(original, points)
    )
    voronoi = (
        lens_result["voronoi"] if lens_result is not None
        else render_voronoi(original, points, color_mode=color_mode)
    )
    destination.mkdir(parents=True, exist_ok=True)

    if sampling_mode == "uniform":
        paths = {
            "original": destination / "01_original.png",
            "points_overlay": destination / "02_points.png",
            "voronoi": destination / "03_voronoi.png",
            "comparison": destination / "04_comparison.png",
            "points": destination / "points.npy",
            "config": destination / "config.json",
        }
        comparison = np.concatenate([original, point_overlay, voronoi], axis=1)
    else:
        assert density is not None
        density_rgb = (
            lens_result["density_rgb"] if lens_result is not None
            else _density_to_rgb(density)
        )
        paths = {
            "original": destination / "01_original.png",
            "density": destination / "02_density.png",
            "points_overlay": destination / "03_points.png",
            "voronoi": destination / "04_voronoi.png",
            "comparison": destination / "05_comparison.png",
            "points": destination / "points.npy",
            "config": destination / "config.json",
        }
        comparison = np.concatenate(
            [original, density_rgb, point_overlay, voronoi], axis=1
        )
        save_rgb_image(density_rgb, paths["density"])

    save_rgb_image(original, paths["original"])
    save_rgb_image(point_overlay, paths["points_overlay"])
    save_rgb_image(voronoi, paths["voronoi"])
    save_rgb_image(comparison, paths["comparison"])
    np.save(paths["points"], points)

    config: dict[str, Any] = {
        "input_path": str(source_path),
        "output_dir": str(destination),
        "sampling_mode": sampling_mode,
        "n_points": n_points,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
        "point_count": int(points.shape[0]),
        "seed": seed,
        "max_size": max_size,
        "density_mode": density_mode,
        "edge_weight": edge_weight,
        "tone_weight": tone_weight,
        "blur_sigma": blur_sigma,
        "density_power": density_power,
        "point_jitter": point_jitter,
        "color_mode": color_mode,
        "width": width,
        "height": height,
        **feature_metadata,
    }
    if sampling_mode == "lens":
        assert resolved_parameters is not None
        config.update(
            {
                "n_points": resolved_parameters["total_points"],
                "n_coarse": resolved_parameters["n_coarse"],
                "n_fine": resolved_parameters["n_fine"],
                "density_mode": "mix",
                "edge_weight": resolved_parameters["edge_weight"],
                "tone_weight": resolved_parameters["tone_weight"],
                "blur_sigma": resolved_parameters["blur_sigma"],
                "density_power": resolved_parameters["density_power"],
                "point_jitter": 0.0,
                "preset": preset,
                "controls": controls,
                "resolved_parameters": resolved_parameters,
            }
        )

    paths["config"].write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        **config,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def run_voronoi_baseline(
    input_path: str | Path,
    output_dir: str | Path,
    n_points: int = 1_000,
    seed: int = 0,
    max_size: int | None = 1_000,
) -> dict[str, Any]:
    """Run the original uniform baseline (kept for API compatibility)."""

    return run_voronoi_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        sampling_mode="uniform",
        n_points=n_points,
        seed=seed,
        max_size=max_size,
    )
