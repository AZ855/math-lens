import csv

import numpy as np
import pytest
from PIL import Image

from math_lens.analysis.svd_rank_study import (
    _parse_rank_values,
    run_svd_rank_study,
)
from math_lens.core.image_features import (
    compute_centered_svd,
    compute_fixed_rank_svd_tone,
    cumulative_singular_value_energy,
    resolve_svd_rank,
)


def varied_image(height: int = 12, width: int = 14) -> np.ndarray:
    ys, xs = np.indices((height, width))
    return np.stack(
        (
            (19 * xs + 7 * ys) % 256,
            (11 * xs**2 + 13 * ys) % 256,
            (17 * xs + 5 * ys**2) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_fixed_rank_reconstruction_preserves_shape() -> None:
    image = varied_image()
    result = compute_fixed_rank_svd_tone(image, rank=5)
    assert result.reconstruction.shape == image.shape[:2]
    assert result.centered_reconstruction.shape == image.shape[:2]
    assert result.tone_map.shape == image.shape[:2]
    assert result.rank == 5


def test_rank_is_validated_and_clamped() -> None:
    assert resolve_svd_rank(4, 3) == 3
    assert resolve_svd_rank(np.int64(2), 3) == 2
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            resolve_svd_rank(invalid, 3)


def test_rank_arguments_accept_commas_or_spaces() -> None:
    assert _parse_rank_values(("1,2", "5", "10")) == (1, 2, 5, 10)


def test_cumulative_energy_is_monotonic_and_increases_with_rank() -> None:
    decomposition = compute_centered_svd(varied_image())
    cumulative = cumulative_singular_value_energy(decomposition.singular_values)
    assert np.all(np.diff(cumulative) >= -1e-12)
    assert cumulative[-1] == pytest.approx(1.0)
    assert cumulative[9] >= cumulative[4]


def test_study_generates_metrics_for_small_image(tmp_path) -> None:
    input_path = tmp_path / "small.png"
    output_dir = tmp_path / "study"
    Image.fromarray(varied_image()).save(input_path)

    result = run_svd_rank_study(
        input_path=input_path,
        output_dir=output_dir,
        ranks=(1, 2),
        max_size=32,
        seed=10,
    )

    assert [metric.rank for metric in result.metrics] == [1, 2]
    assert result.panel_path.is_file()
    assert result.decay_path.is_file()
    with result.metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["rank"]) for row in rows] == [1, 2]
    assert all(int(row["n_points"]) == 2750 for row in rows)
    assert (output_dir / "rank_01" / "voronoi.png").is_file()
