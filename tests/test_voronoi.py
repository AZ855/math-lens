import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from math_lens.cli import _resolve_lens_controls, build_parser
from math_lens.core.image_features import (
    build_density_from_features,
    build_density_map,
    compute_density_features,
    compute_edge_map,
    compute_luminance,
    compute_svd_tone,
    normalize_01,
)
from math_lens.core.image_io import load_rgb_bytes, save_rgb_image
from math_lens.core.parameters import clamp01, lerp, map_voronoi_lens_controls
from math_lens.lenses.voronoi.pipeline import (
    render_lens_result,
    run_voronoi_baseline,
    run_voronoi_pipeline,
)
from math_lens.lenses.voronoi.renderer import (
    render_voronoi,
    render_voronoi_cells,
    render_voronoi_mean_color,
    render_voronoi_seed_color,
    voronoi_label_map,
)
from math_lens.lenses.voronoi.sampling import (
    generate_adaptive_points,
    generate_uniform_points,
)


def synthetic_image() -> np.ndarray:
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, 4:] = 220
    image[2:4, 2:6, 1] = 80
    return image


def test_uniform_points_have_expected_shape_and_dtype() -> None:
    points = generate_uniform_points(width=20, height=10, n_points=8, seed=3)
    assert points.shape == (8, 2)
    assert points.dtype == np.float32


def test_uniform_points_stay_within_image_bounds() -> None:
    points = generate_uniform_points(width=20, height=10, n_points=100, seed=3)
    assert np.all((0 <= points[:, 0]) & (points[:, 0] < 20))
    assert np.all((0 <= points[:, 1]) & (points[:, 1] < 10))


def test_uniform_points_are_reproducible() -> None:
    first = generate_uniform_points(width=20, height=10, n_points=8, seed=7)
    second = generate_uniform_points(width=20, height=10, n_points=8, seed=7)
    np.testing.assert_array_equal(first, second)


def test_voronoi_label_map_shape_and_label_range() -> None:
    points = np.array([[0.0, 0.0], [5.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = voronoi_label_map(points, width=6, height=5, chunk=7)
    assert labels.shape == (5, 6)
    assert labels.dtype == np.int32
    assert labels.min() >= 0
    assert labels.max() < len(points)


def test_mean_color_render_preserves_shape_and_dtype() -> None:
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    points = np.array([[0.0, 0.0], [5.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rendered = render_voronoi_mean_color(image, points)
    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8


def test_compute_luminance_shape_and_dtype() -> None:
    luminance = compute_luminance(synthetic_image())
    assert luminance.shape == (6, 8)
    assert luminance.dtype == np.float32


def test_compute_luminance_uses_perceptual_rgb_weights() -> None:
    primaries = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8
    )

    luminance = compute_luminance(primaries)

    np.testing.assert_allclose(luminance[0], [0.2126, 0.7152, 0.0722])


def test_svd_tone_selects_smallest_rank_reaching_energy_threshold() -> None:
    grayscale = np.diag([255, 128, 64]).astype(np.uint8)
    image = np.repeat(grayscale[..., None], 3, axis=2)

    result = compute_svd_tone(image, energy_threshold=0.80)

    squared = np.square(result.singular_values.astype(np.float64))
    cumulative = np.cumsum(squared) / squared.sum()
    assert result.rank >= 1
    assert result.retained_energy >= 0.80
    if result.rank > 1:
        assert cumulative[result.rank - 2] < 0.80
    assert result.reconstruction.shape == grayscale.shape
    assert result.reconstruction.dtype == np.float32
    assert result.centered_reconstruction.shape == grayscale.shape
    assert result.centered_reconstruction.dtype == np.float32
    assert result.tone_map.shape == grayscale.shape
    assert result.tone_map.dtype == np.float32
    centered_luminance = compute_luminance(image) - result.luminance_mean
    assert centered_luminance.mean() == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(
        result.reconstruction,
        result.centered_reconstruction + result.luminance_mean,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.tone_map,
        normalize_01(np.abs(result.centered_reconstruction)),
        atol=1e-7,
    )
    assert np.isfinite(result.tone_map).all()
    assert 0.0 <= result.tone_map.min() <= result.tone_map.max() <= 1.0


def test_svd_tone_handles_constant_image() -> None:
    image = np.full((4, 5, 3), 128, dtype=np.uint8)

    result = compute_svd_tone(image)

    assert result.rank == 1
    assert result.retained_energy == 1.0
    assert np.isfinite(result.reconstruction).all()
    assert np.isfinite(result.tone_map).all()
    np.testing.assert_allclose(result.reconstruction, 128 / 255)
    np.testing.assert_array_equal(result.centered_reconstruction, np.zeros((4, 5)))
    np.testing.assert_array_equal(result.tone_map, np.zeros((4, 5)))


def test_centered_svd_differs_from_raw_svd_with_large_dc_offset() -> None:
    grayscale = np.full((8, 10), 180, dtype=np.uint8)
    grayscale[:, :5] = 130
    grayscale[2:7, 3:9] = np.clip(grayscale[2:7, 3:9] + 35, 0, 255)
    image = np.repeat(grayscale[..., None], 3, axis=2)
    centered_result = compute_svd_tone(image, energy_threshold=0.90)

    luminance = compute_luminance(image)
    left, singular_values, right_transpose = np.linalg.svd(
        luminance, full_matrices=False
    )
    squared = np.square(singular_values.astype(np.float64))
    cumulative = np.cumsum(squared) / squared.sum()
    raw_rank = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    raw_reconstruction = (
        left[:, :raw_rank] * singular_values[:raw_rank]
    ) @ right_transpose[:raw_rank, :]
    raw_tone = normalize_01(
        np.abs(raw_reconstruction - raw_reconstruction.mean())
    )

    assert centered_result.rank >= 1
    assert centered_result.retained_energy >= 0.90
    assert not np.allclose(centered_result.tone_map, raw_tone)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, np.inf, np.nan])
def test_svd_tone_rejects_invalid_energy_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="energy_threshold"):
        compute_svd_tone(synthetic_image(), energy_threshold=threshold)


def test_compute_edge_map_preserves_spatial_shape() -> None:
    edge_map = compute_edge_map(synthetic_image())
    assert edge_map.shape == (6, 8)
    assert edge_map.dtype == np.float32


def test_density_map_is_nonnegative_and_has_image_shape() -> None:
    density = build_density_map(synthetic_image(), mode="mix")
    assert density.shape == (6, 8)
    assert density.dtype == np.float32
    assert np.all(density >= 0)


def test_default_density_uses_svd_tone_and_luminance_remains_available() -> None:
    image = synthetic_image()

    svd_density = build_density_map(image, mode="mix", tone_method="svd")
    luminance_density = build_density_map(
        image, mode="mix", tone_method="luminance"
    )

    assert not np.array_equal(svd_density, luminance_density)


def test_adaptive_points_have_expected_shape() -> None:
    density = build_density_map(synthetic_image())
    points = generate_adaptive_points(density, n_coarse=3, n_fine=7, seed=4)
    assert points.shape == (10, 2)
    assert points.dtype == np.float32


def test_adaptive_points_stay_within_image_bounds() -> None:
    density = build_density_map(synthetic_image())
    points = generate_adaptive_points(
        density,
        n_coarse=10,
        n_fine=20,
        seed=4,
        jitter=3.0,
    )
    assert np.all((0 <= points[:, 0]) & (points[:, 0] < density.shape[1]))
    assert np.all((0 <= points[:, 1]) & (points[:, 1] < density.shape[0]))


def test_adaptive_points_are_reproducible() -> None:
    density = build_density_map(synthetic_image())
    first = generate_adaptive_points(density, n_coarse=3, n_fine=7, seed=9)
    second = generate_adaptive_points(density, n_coarse=3, n_fine=7, seed=9)
    np.testing.assert_array_equal(first, second)


def test_pipeline_still_works_in_uniform_mode(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "result"
    save_rgb_image(synthetic_image(), input_path)

    result = run_voronoi_baseline(input_path, output_dir, n_points=4, seed=11)

    assert result["sampling_mode"] == "uniform"
    assert result["width"] == 8
    assert result["height"] == 6
    assert result["n_points"] == 4
    assert result["point_count"] == 4
    assert set(path.name for path in output_dir.iterdir()) == {
        "01_original.png",
        "02_points.png",
        "03_voronoi.png",
        "04_comparison.png",
        "points.npy",
        "config.json",
    }
    assert np.load(output_dir / "points.npy").shape == (4, 2)


def test_adaptive_pipeline_outputs_paths_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "adaptive"
    save_rgb_image(synthetic_image(), input_path)

    result = run_voronoi_pipeline(
        input_path,
        output_dir,
        sampling_mode="adaptive",
        n_coarse=3,
        n_fine=7,
        seed=12,
        blur_sigma=1.0,
    )

    assert result["sampling_mode"] == "adaptive"
    assert result["point_count"] == 10
    assert result["width"] == 8
    assert result["height"] == 6
    assert set(result["paths"]) == {
        "original",
        "density",
        "points_overlay",
        "voronoi",
        "comparison",
        "points",
        "config",
    }
    assert all((output_dir / path.name).is_file() for path in output_dir.iterdir())
    assert set(path.name for path in output_dir.iterdir()) == {
        "01_original.png",
        "02_density.png",
        "03_points.png",
        "04_voronoi.png",
        "05_comparison.png",
        "points.npy",
        "config.json",
    }


def test_clamp01() -> None:
    assert clamp01(-2.0) == 0.0
    assert clamp01(0.4) == 0.4
    assert clamp01(3.0) == 1.0


def test_lerp() -> None:
    assert lerp(10.0, 20.0, 0.0) == 10.0
    assert lerp(10.0, 20.0, 0.25) == 12.5
    assert lerp(10.0, 20.0, 1.0) == 20.0


def test_lens_mapping_stays_in_valid_ranges() -> None:
    parameters = map_voronoi_lens_controls(0.5, 0.5, 0.5)
    assert 500 <= parameters.total_points <= 5_000
    assert parameters.n_coarse + parameters.n_fine == parameters.total_points
    assert parameters.n_coarse >= 0
    assert parameters.n_fine >= 0
    assert 2.0 <= parameters.blur_sigma <= 14.0
    assert 1.0 <= parameters.density_power <= 4.0
    assert 0.0 <= parameters.edge_weight <= 1.0
    assert 0.0 <= parameters.tone_weight <= 1.0


def test_abstraction_reduces_total_points() -> None:
    detailed = map_voronoi_lens_controls(0.0, 0.5, 0.5)
    abstract = map_voronoi_lens_controls(1.0, 0.5, 0.5)
    assert detailed.total_points > abstract.total_points


def test_detail_emphasis_increases_fine_point_ratio() -> None:
    broad = map_voronoi_lens_controls(0.5, 0.0, 0.5)
    detailed = map_voronoi_lens_controls(0.5, 1.0, 0.5)
    broad_ratio = broad.n_fine / broad.total_points
    detailed_ratio = detailed.n_fine / detailed.total_points
    assert detailed_ratio > broad_ratio


def test_structure_maps_directly_to_edge_and_tone_weights() -> None:
    parameters = map_voronoi_lens_controls(0.5, 0.5, 0.8)
    assert parameters.edge_weight == pytest.approx(0.8)
    assert parameters.tone_weight == pytest.approx(0.2)


def test_seed_color_renderer_uses_each_sites_source_color() -> None:
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )
    points = np.array([[0.2, 0.2], [2.8, 1.8]], dtype=np.float32)
    labels = voronoi_label_map(points, width=3, height=2)
    rendered = render_voronoi_seed_color(image, points)

    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert np.all(rendered[labels == 0] == image[0, 0])
    assert np.all(rendered[labels == 1] == image[1, 2])


def test_render_voronoi_dispatcher() -> None:
    image = synthetic_image()
    points = np.array([[0.0, 0.0], [7.0, 5.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        render_voronoi(image, points, color_mode="mean"),
        render_voronoi_mean_color(image, points),
    )
    np.testing.assert_array_equal(
        render_voronoi(image, points, color_mode="seed"),
        render_voronoi_seed_color(image, points),
    )
    with pytest.raises(ValueError, match="Unsupported"):
        render_voronoi(image, points, color_mode="unknown")


def test_lens_pipeline_writes_files_and_resolved_config(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "lens"
    save_rgb_image(synthetic_image(), input_path)

    result = run_voronoi_pipeline(
        input_path,
        output_dir,
        sampling_mode="lens",
        abstraction=1.0,
        detail_emphasis=0.6,
        structure=0.85,
        color_mode="seed",
        seed=5,
    )

    expected_names = {
        "01_original.png",
        "02_density.png",
        "03_points.png",
        "04_voronoi.png",
        "05_comparison.png",
        "points.npy",
        "config.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names
    assert result["sampling_mode"] == "lens"
    assert result["point_count"] == 500
    assert result["controls"] == {
        "abstraction": 1.0,
        "detail_emphasis": 0.6,
        "structure": 0.85,
        "color_mode": "seed",
    }

    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config["tone_method"] == "svd"
    assert saved_config["svd_energy_threshold"] == pytest.approx(0.90)
    assert saved_config["svd_rank"] >= 1
    assert saved_config["svd_retained_energy"] >= 0.90
    assert saved_config["svd_centered"] is True
    assert saved_config["svd_luminance_mean"] == pytest.approx(
        compute_luminance(synthetic_image()).mean()
    )
    assert saved_config["resolved_parameters"] == result["resolved_parameters"]
    assert saved_config["resolved_parameters"]["total_points"] == 500
    assert (
        saved_config["resolved_parameters"]["n_coarse"]
        + saved_config["resolved_parameters"]["n_fine"]
        == 500
    )


def test_explicit_lens_controls_override_preset() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "voronoi",
            "--input",
            "input.png",
            "--output",
            "output",
            "--sampling",
            "lens",
            "--preset",
            "detailed",
            "--abstraction",
            "0.9",
        ]
    )
    controls = _resolve_lens_controls(args)
    assert controls["abstraction"] == 0.9
    assert controls["detail_emphasis"] == 0.75


def test_cli_accepts_legacy_luminance_tone_method() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "voronoi",
            "--input",
            "input.png",
            "--output",
            "output",
            "--sampling",
            "adaptive",
            "--tone-method",
            "luminance",
            "--svd-energy-threshold",
            "0.95",
        ]
    )

    assert args.tone_method == "luminance"
    assert args.svd_energy_threshold == pytest.approx(0.95)


def test_load_rgb_bytes_uses_shared_resize_rules() -> None:
    buffer = BytesIO()
    Image.fromarray(synthetic_image(), mode="RGB").save(buffer, format="PNG")
    loaded = load_rgb_bytes(buffer.getvalue(), max_size=4)
    assert loaded.shape == (3, 4, 3)
    assert loaded.dtype == np.uint8


def test_in_memory_lens_result_has_four_panels_and_no_files(tmp_path) -> None:
    result = render_lens_result(
        synthetic_image(),
        abstraction=1.0,
        detail_emphasis=0.55,
        structure=0.75,
        seed=10,
    )
    assert result["original"].shape == (6, 8, 3)
    assert result["density_rgb"].shape == (6, 8, 3)
    assert result["points_overlay"].shape == (6, 8, 3)
    assert result["voronoi"].shape == (6, 8, 3)
    assert result["points"].shape == (500, 2)
    assert result["resolved_parameters"]["total_points"] == 500
    assert result["feature_metadata"]["tone_method"] == "svd"
    assert result["feature_metadata"]["svd_rank"] >= 1
    assert result["feature_metadata"]["svd_retained_energy"] >= 0.90
    assert result["feature_metadata"]["svd_centered"] is True
    assert result["feature_metadata"]["svd_luminance_mean"] is not None
    assert result["svd_reconstruction"].shape == (6, 8)
    assert result["svd_centered_reconstruction"].shape == (6, 8)
    assert result["tone_map"].shape == (6, 8)
    assert result["structure_map"].shape == (6, 8)
    assert list(tmp_path.iterdir()) == []


def test_structure_control_crossfades_svd_tone_and_sobel_structure() -> None:
    image = synthetic_image()
    svd_tone = compute_svd_tone(image)
    edges = compute_edge_map(image)

    tone_result = render_lens_result(
        image,
        abstraction=1.0,
        structure=0.0,
        edge_map=edges,
        svd_tone_result=svd_tone,
    )
    structure_result = render_lens_result(
        image,
        abstraction=1.0,
        structure=1.0,
        edge_map=edges,
        svd_tone_result=svd_tone,
    )

    np.testing.assert_allclose(
        tone_result["density"], np.clip(svd_tone.tone_map, 1e-12, None)
    )
    np.testing.assert_allclose(
        structure_result["density"], np.clip(edges, 1e-12, None)
    )


def test_luminance_tone_method_preserves_legacy_feature() -> None:
    image = synthetic_image()
    result = render_lens_result(
        image,
        abstraction=1.0,
        structure=0.0,
        tone_method="luminance",
    )

    expected = normalize_01(compute_luminance(image))
    np.testing.assert_allclose(result["tone_map"], expected)
    assert result["feature_metadata"]["tone_method"] == "luminance"
    assert result["feature_metadata"]["svd_rank"] is None
    assert result["feature_metadata"]["svd_centered"] is False
    assert result["feature_metadata"]["svd_luminance_mean"] is None


def test_lens_pipeline_records_luminance_comparison_method(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "luminance-lens"
    save_rgb_image(synthetic_image(), input_path)

    result = run_voronoi_pipeline(
        input_path,
        output_dir,
        sampling_mode="lens",
        abstraction=1.0,
        tone_method="luminance",
    )
    saved_config = json.loads((output_dir / "config.json").read_text())

    assert result["tone_method"] == "luminance"
    assert result["svd_rank"] is None
    assert result["svd_retained_energy"] is None
    assert result["svd_centered"] is False
    assert result["svd_luminance_mean"] is None
    assert saved_config["tone_method"] == "luminance"


def test_cached_features_reproduce_density_map() -> None:
    image = synthetic_image()
    tone_map, edges = compute_density_features(image)
    from_features = build_density_from_features(
        tone_map, edges, mode="mix", edge_weight=0.7, tone_weight=0.3
    )
    direct = build_density_map(
        image, mode="mix", edge_weight=0.7, tone_weight=0.3
    )
    np.testing.assert_array_equal(from_features, direct)


def test_voronoi_cells_show_boundaries_and_seeds() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    points = np.array([[1.0, 2.0], [5.0, 2.0]], dtype=np.float32)
    cells = render_voronoi_cells(
        image,
        points,
        show_seeds=True,
        line_value=60,
        background_value=240,
    )
    assert cells.shape == image.shape
    assert cells.dtype == np.uint8
    assert np.any(np.all(cells == 60, axis=2))
    assert np.any(np.all(cells == 20, axis=2))
    assert np.any(np.all(cells == 240, axis=2))


def test_in_memory_lens_result_can_include_cell_geometry() -> None:
    image = synthetic_image()
    luminance, edges = compute_density_features(image)
    result = render_lens_result(
        image,
        abstraction=1.0,
        detail_emphasis=0.55,
        structure=0.75,
        luminance_normalized=luminance,
        edge_map=edges,
        include_cells=True,
    )
    assert result["voronoi_cells"].shape == image.shape
