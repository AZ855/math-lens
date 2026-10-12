"""Command-line interface for Math Lens."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from math_lens.lenses.voronoi.pipeline import run_voronoi_pipeline
from math_lens.lenses.triangulation.pipeline import (
    TRIANGULATION_PRESETS,
    run_triangulation_pipeline,
)


LENS_DEFAULTS: dict[str, float | str] = {
    "abstraction": 0.5,
    "detail_emphasis": 0.5,
    "structure": 0.8,
    "color_mode": "mean",
}

LENS_PRESETS: dict[str, dict[str, float | str]] = {
    "detailed": {
        "abstraction": 0.15,
        "detail_emphasis": 0.75,
        "structure": 0.85,
        "color_mode": "mean",
    },
    "balanced": {
        "abstraction": 0.50,
        "detail_emphasis": 0.55,
        "structure": 0.75,
        "color_mode": "mean",
    },
    "abstract": {
        "abstraction": 0.85,
        "detail_emphasis": 0.35,
        "structure": 0.65,
        "color_mode": "mean",
    },
}


def _normalized_float(value: str) -> float:
    try:
        numeric = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from error
    if not 0.0 <= numeric <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return numeric


def _energy_threshold(value: str) -> float:
    numeric = _normalized_float(value)
    if numeric == 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return numeric


def _resolve_lens_controls(args: argparse.Namespace) -> dict[str, Any]:
    controls = dict(LENS_DEFAULTS)
    if args.preset is not None:
        controls.update(LENS_PRESETS[args.preset])

    # None means the flag was omitted. Any explicit flag therefore wins over
    # both the default and its preset value.
    for name in ("abstraction", "detail_emphasis", "structure", "color_mode"):
        value = getattr(args, name)
        if value is not None:
            controls[name] = value
    return controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m math_lens.cli",
        description="Apply mathematical lenses to images.",
    )
    subparsers = parser.add_subparsers(dest="lens", required=True)
    voronoi = subparsers.add_parser(
        "voronoi", help="Run a uniform, adaptive, or high-level Voronoi Lens."
    )
    voronoi.add_argument("--input", required=True, type=Path, help="Source image path.")
    voronoi.add_argument("--output", required=True, type=Path, help="Output directory.")
    voronoi.add_argument(
        "--sampling",
        choices=("uniform", "adaptive", "lens"),
        default="uniform",
        help="Seed sampling interface (default: uniform).",
    )
    voronoi.add_argument(
        "--points", type=int, default=1_000, help="Uniform-mode seed count."
    )
    voronoi.add_argument(
        "--coarse-points",
        type=int,
        default=400,
        help="Adaptive broad-coverage seed count.",
    )
    voronoi.add_argument(
        "--fine-points",
        type=int,
        default=1_600,
        help="Adaptive detail-focused seed count.",
    )
    voronoi.add_argument("--seed", type=int, default=0, help="Random seed.")
    voronoi.add_argument(
        "--max-size",
        type=int,
        default=1_000,
        help="Maximum processed width or height in pixels.",
    )
    voronoi.add_argument(
        "--density-mode",
        choices=("edge", "bright", "dark", "mix"),
        default="mix",
        help="Image feature used by low-level adaptive sampling.",
    )
    voronoi.add_argument("--edge-weight", type=float, default=0.85)
    voronoi.add_argument("--tone-weight", type=float, default=0.15)
    voronoi.add_argument("--blur-sigma", type=float, default=6.0)
    voronoi.add_argument("--density-power", type=float, default=2.0)
    voronoi.add_argument("--point-jitter", type=float, default=0.0)
    voronoi.add_argument(
        "--tone-method",
        choices=("svd", "luminance"),
        default="svd",
        help="Tone feature for mixed adaptive density (default: svd).",
    )
    voronoi.add_argument(
        "--svd-energy-threshold",
        type=_energy_threshold,
        default=0.90,
        help="Singular-value energy retained by SVD tone (default: 0.90).",
    )

    lens_group = voronoi.add_argument_group("high-level Lens mode")
    lens_group.add_argument(
        "--preset",
        choices=tuple(LENS_PRESETS),
        help="Lens preset; explicit Lens controls override its values.",
    )
    lens_group.add_argument("--abstraction", type=_normalized_float)
    lens_group.add_argument("--detail-emphasis", type=_normalized_float)
    lens_group.add_argument("--structure", type=_normalized_float)
    lens_group.add_argument("--color-mode", choices=("mean", "seed"))

    triangulation = subparsers.add_parser(
        "triangulation", help="Run a Delaunay mesh with optional bounded Lloyd relaxation."
    )
    triangulation.add_argument("--input", required=True, type=Path)
    triangulation.add_argument("--output", required=True, type=Path)
    triangulation.add_argument("--sampling", choices=("uniform", "adaptive"), default="adaptive")
    triangulation.add_argument("--preset", choices=tuple(TRIANGULATION_PRESETS))
    triangulation.add_argument("--structural-resolution", type=_normalized_float)
    triangulation.add_argument("--regularity", type=_normalized_float)
    triangulation.add_argument("--structure-emphasis", type=_normalized_float)
    triangulation.add_argument("--mesh-visibility", type=_normalized_float)
    triangulation.add_argument("--color-mode", choices=("mean_vertex", "gouraud"), default="mean_vertex")
    triangulation.add_argument("--render-mode", choices=("mesh", "filled", "filled_mesh"), default="filled_mesh")
    triangulation.add_argument("--seed", type=int, default=10)
    triangulation.add_argument("--max-size", type=int, default=1_000)
    triangulation.add_argument(
        "--voronoi-comparison", action=argparse.BooleanOptionalAction, default=True,
        help="Save the same-site Voronoi/Delaunay comparison (enabled by default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.lens == "triangulation":
        try:
            result = run_triangulation_pipeline(
                input_path=args.input, output_dir=args.output,
                sampling_mode=args.sampling,
                structural_resolution=args.structural_resolution,
                regularity=args.regularity,
                structure_emphasis=args.structure_emphasis,
                mesh_visibility=args.mesh_visibility,
                color_mode=args.color_mode,
                render_mode=args.render_mode,
                seed=args.seed, max_size=args.max_size, preset=args.preset,
                include_voronoi_comparison=args.voronoi_comparison,
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0

    if args.lens == "voronoi":
        if not args.input.is_file():
            parser.error(f"input image does not exist: {args.input}")

        lens_flags_used = any(
            value is not None
            for value in (
                args.preset,
                args.abstraction,
                args.detail_emphasis,
                args.structure,
                args.color_mode,
            )
        )
        if args.sampling != "lens" and lens_flags_used:
            parser.error(
                "--preset, --abstraction, --detail-emphasis, --structure, and "
                "--color-mode require --sampling lens"
            )

        lens_controls = _resolve_lens_controls(args)
        try:
            result = run_voronoi_pipeline(
                input_path=args.input,
                output_dir=args.output,
                sampling_mode=args.sampling,
                n_points=args.points,
                n_coarse=args.coarse_points,
                n_fine=args.fine_points,
                seed=args.seed,
                max_size=args.max_size,
                density_mode=args.density_mode,
                edge_weight=args.edge_weight,
                tone_weight=args.tone_weight,
                blur_sigma=args.blur_sigma,
                density_power=args.density_power,
                point_jitter=args.point_jitter,
                abstraction=lens_controls["abstraction"],
                detail_emphasis=lens_controls["detail_emphasis"],
                structure=lens_controls["structure"],
                color_mode=lens_controls["color_mode"],
                preset=args.preset,
                tone_method=args.tone_method,
                svd_energy_threshold=args.svd_energy_threshold,
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"unsupported lens: {args.lens}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
