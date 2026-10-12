"""Compare fixed centered-SVD ranks through the Voronoi Lens pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Sequence

_CACHE_ROOT = Path(tempfile.gettempdir()) / "math_lens_matplotlib_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from math_lens.core.image_features import (
    CenteredSVDResult,
    compute_centered_svd,
    compute_edge_map,
    cumulative_singular_value_energy,
    reconstruct_svd_tone,
    resolve_svd_rank,
)
from math_lens.core.image_io import load_rgb_image, save_rgb_image
from math_lens.lenses.voronoi.pipeline import render_lens_result


DEFAULT_RANKS = (1, 2, 5, 10)
ABSTRACTION = 0.50
DETAIL_EMPHASIS = 0.55
STRUCTURE = 0.0
COLOR_MODE = "mean"
AUTO_ENERGY_THRESHOLD = 0.90


@dataclass(frozen=True)
class RankMetric:
    """One row of the fixed-rank experiment's tabular results."""

    rank: int
    singular_value: float
    cumulative_energy: float
    total_singular_values: int
    image_width: int
    image_height: int
    n_points: int


@dataclass(frozen=True)
class RankStudyResult:
    """Paths and measurements produced by a completed rank study."""

    output_dir: Path
    panel_path: Path
    decay_path: Path
    metrics_path: Path
    auto_rank: int
    auto_energy: float
    metrics: tuple[RankMetric, ...]


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parse_rank_values(values: Sequence[str]) -> tuple[int, ...]:
    ranks: list[int] = []
    for value in values:
        parts = value.split(",")
        if any(not part.strip() for part in parts):
            raise ValueError("ranks must be positive integers")
        try:
            ranks.extend(_positive_int(part.strip()) for part in parts)
        except argparse.ArgumentTypeError as error:
            raise ValueError("ranks must be positive integers") from error
    return tuple(ranks)


def _resolved_unique_ranks(
    ranks: Sequence[int], total_singular_values: int
) -> tuple[int, ...]:
    if not ranks:
        raise ValueError("at least one comparison rank is required")
    resolved: list[int] = []
    for rank in ranks:
        selected = resolve_svd_rank(rank, total_singular_values)
        if selected not in resolved:
            resolved.append(selected)
    return tuple(resolved)


def _metric(
    decomposition: CenteredSVDResult,
    cumulative_energy: np.ndarray,
    rank: int,
    width: int,
    height: int,
    n_points: int,
) -> RankMetric:
    return RankMetric(
        rank=rank,
        singular_value=float(decomposition.singular_values[rank - 1]),
        cumulative_energy=float(cumulative_energy[rank - 1]),
        total_singular_values=int(decomposition.singular_values.size),
        image_width=width,
        image_height=height,
        n_points=n_points,
    )


def _save_metrics(path: Path, metrics: Sequence[RankMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(RankMetric.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "rank": item.rank,
                    "singular_value": f"{item.singular_value:.12g}",
                    "cumulative_energy": f"{item.cumulative_energy:.12g}",
                    "total_singular_values": item.total_singular_values,
                    "image_width": item.image_width,
                    "image_height": item.image_height,
                    "n_points": item.n_points,
                }
            )


def _save_rank_panel(
    path: Path,
    rows: Sequence[tuple[RankMetric, dict[str, object]]],
    seed: int,
) -> None:
    figure, axes = plt.subplots(
        len(rows),
        4,
        figsize=(16, 3.25 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = ("SVD reconstruction", "Tone / density", "Voronoi", "Metrics")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold")

    for row_index, (metric, render) in enumerate(rows):
        images = (
            render["svd_reconstruction_rgb"],
            render["density_rgb"],
            render["voronoi"],
        )
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_axis_off()
        axes[row_index, 0].set_ylabel(
            f"rank {metric.rank}", fontsize=11, fontweight="bold"
        )
        metric_axis = axes[row_index, 3]
        metric_axis.set_axis_off()
        metric_axis.text(
            0.04,
            0.56,
            "\n".join(
                (
                    f"rank: {metric.rank}",
                    f"sigma[{metric.rank}]: {metric.singular_value:.5g}",
                    f"cumulative energy: {metric.cumulative_energy:.2%}",
                    f"sampled sites: {metric.n_points:,}",
                )
            ),
            va="center",
            ha="left",
            fontsize=12,
            linespacing=1.6,
        )

    figure.suptitle(
        "Centered-SVD Rank Study\n"
        "Only rank varies; abstraction 0.50, detail emphasis 0.55, "
        f"structure 0.00, mean color, seed {seed}",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _save_singular_value_decay(
    path: Path,
    singular_values: np.ndarray,
    cumulative_energy: np.ndarray,
    comparison_ranks: Sequence[int],
    auto_rank: int,
) -> None:
    indices = np.arange(1, singular_values.size + 1)
    safe_values = np.clip(singular_values, np.finfo(np.float32).tiny, None)
    figure, (singular_axis, energy_axis) = plt.subplots(
        2, 1, figsize=(11, 8), constrained_layout=True, sharex=True
    )

    singular_axis.semilogy(indices, safe_values, color="#315b78", linewidth=1.6)
    for rank in comparison_ranks:
        singular_axis.scatter(
            rank,
            safe_values[rank - 1],
            color="#c34a36",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        singular_axis.annotate(
            f"k={rank}",
            (rank, safe_values[rank - 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    singular_axis.set_ylabel("Singular value (log scale)")
    singular_axis.set_title("Centered-luminance singular-value decay")
    singular_axis.grid(alpha=0.25)

    energy_axis.plot(indices, cumulative_energy, color="#315b78", linewidth=1.8)
    energy_axis.axhline(
        AUTO_ENERGY_THRESHOLD,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="90% energy",
    )
    energy_axis.axvline(
        auto_rank,
        color="#da8b2d",
        linestyle=":",
        linewidth=1.8,
        label=f"automatic rank = {auto_rank}",
    )
    energy_axis.scatter(
        comparison_ranks,
        [cumulative_energy[rank - 1] for rank in comparison_ranks],
        color="#c34a36",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        label="comparison ranks",
    )
    energy_axis.set_xlabel("Rank")
    energy_axis.set_ylabel("Cumulative squared energy")
    energy_axis.set_ylim(0.0, 1.02)
    energy_axis.grid(alpha=0.25)
    energy_axis.legend(loc="lower right")

    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def run_svd_rank_study(
    input_path: str | Path,
    output_dir: str | Path,
    ranks: Sequence[int] = DEFAULT_RANKS,
    max_size: int | None = 1_000,
    seed: int = 10,
) -> RankStudyResult:
    """Run the controlled Week 4 rank comparison and save every artifact."""

    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    original = load_rgb_image(source, max_size=max_size)
    height, width = original.shape[:2]
    decomposition = compute_centered_svd(original)
    cumulative_energy = cumulative_singular_value_energy(
        decomposition.singular_values
    )
    selected_ranks = _resolved_unique_ranks(
        ranks, decomposition.singular_values.size
    )
    auto_rank = int(
        np.searchsorted(
            cumulative_energy, AUTO_ENERGY_THRESHOLD, side="left"
        )
        + 1
    )
    edge_map = compute_edge_map(original)

    rows: list[tuple[RankMetric, dict[str, object]]] = []
    for rank in selected_ranks:
        svd_result = reconstruct_svd_tone(decomposition, rank)
        render = render_lens_result(
            original,
            abstraction=ABSTRACTION,
            detail_emphasis=DETAIL_EMPHASIS,
            structure=STRUCTURE,
            color_mode=COLOR_MODE,
            seed=seed,
            edge_map=edge_map,
            tone_method="svd",
            svd_tone_result=svd_result,
        )
        metric = _metric(
            decomposition,
            cumulative_energy,
            rank,
            width,
            height,
            len(render["points"]),
        )
        rows.append((metric, render))

        rank_dir = destination / f"rank_{rank:02d}"
        save_rgb_image(render["svd_reconstruction_rgb"], rank_dir / "reconstruction.png")
        save_rgb_image(render["tone_map_rgb"], rank_dir / "tone_map.png")
        save_rgb_image(render["density_rgb"], rank_dir / "density.png")
        save_rgb_image(render["points_overlay"], rank_dir / "points.png")
        save_rgb_image(render["voronoi"], rank_dir / "voronoi.png")

    metrics = tuple(metric for metric, _ in rows)
    metrics_path = destination / "rank_metrics.csv"
    panel_path = destination / "svd_rank_panel.png"
    decay_path = destination / "singular_value_decay.png"
    _save_metrics(metrics_path, metrics)
    _save_rank_panel(panel_path, rows, seed)
    _save_singular_value_decay(
        decay_path,
        decomposition.singular_values,
        cumulative_energy,
        selected_ranks,
        auto_rank,
    )
    return RankStudyResult(
        output_dir=destination,
        panel_path=panel_path,
        decay_path=decay_path,
        metrics_path=metrics_path,
        auto_rank=auto_rank,
        auto_energy=float(cumulative_energy[auto_rank - 1]),
        metrics=metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m math_lens.analysis.svd_rank_study",
        description=(
            "Compare fixed centered-SVD ranks while holding every Voronoi "
            "Lens parameter constant."
        ),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=None,
        metavar="RANK[,RANK...]",
        help="Fixed ranks, comma- or space-separated (default: 1 2 5 10).",
    )
    parser.add_argument(
        "--max-size",
        type=_positive_int,
        default=1_000,
        help="Maximum processed dimension (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ranks = (
            DEFAULT_RANKS
            if args.ranks is None
            else _parse_rank_values(args.ranks)
        )
        result = run_svd_rank_study(
            input_path=args.input,
            output_dir=args.output,
            ranks=ranks,
            max_size=args.max_size,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved rank panel: {result.panel_path}")
    print(f"Saved decay plot: {result.decay_path}")
    print(f"Saved metrics: {result.metrics_path}")
    print(
        f"Automatic 90% rank: {result.auto_rank} "
        f"({result.auto_energy:.2%} cumulative energy)"
    )
    for metric in result.metrics:
        print(
            f"rank {metric.rank}: sigma={metric.singular_value:.6g}, "
            f"energy={metric.cumulative_energy:.2%}, "
            f"sites={metric.n_points}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
