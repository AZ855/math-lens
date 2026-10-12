import json

import numpy as np
import pytest

from math_lens.cli import build_parser
from math_lens.core.geometry import lloyd_relax_points
from math_lens.core.image_io import save_rgb_image
from math_lens.core.parameters import map_triangulation_controls
from math_lens.lenses.triangulation import pipeline
from math_lens.lenses.triangulation.renderer import (
    compute_delaunay_triangles,
    render_triangulation,
    render_triangulation_mesh,
)


def sample_sites() -> np.ndarray:
    return np.array([[0., 0.], [19., 0.], [19., 15.], [0., 15.], [3., 6.], [8., 5.], [13., 12.]])


def sample_image() -> np.ndarray:
    ys, xs = np.indices((16, 20))
    return np.stack((xs * 10, ys * 12, xs * 3 + ys * 5), axis=-1).astype(np.uint8)


def test_delaunay_connectivity_uses_valid_indices_and_rejects_collinearity() -> None:
    points = sample_sites()
    triangles = compute_delaunay_triangles(points)
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert triangles.min() >= 0 and triangles.max() < len(points)
    assert len(np.unique(triangles)) == len(points)
    with pytest.raises(ValueError, match="non-collinear"):
        compute_delaunay_triangles(np.array([[0, 0], [1, 1], [2, 2]]))


def test_lloyd_stays_bounded_pins_corners_and_zero_preserves_sites() -> None:
    points = sample_sites()
    zero = lloyd_relax_points(points, width=20, height=16, iterations=0)
    np.testing.assert_array_equal(zero, points)
    relaxed = lloyd_relax_points(points, 20, 16, iterations=2, fixed_indices=(0, 1, 2, 3))
    assert np.all(relaxed >= 0)
    assert np.all(relaxed <= [19, 15])
    np.testing.assert_array_equal(relaxed[:4], points[:4])
    assert not np.array_equal(relaxed[4:], points[4:])
    np.testing.assert_array_equal(points, sample_sites())


def test_lloyd_uses_continuous_bounded_cell_centroids() -> None:
    # Four quadrant cells have exactly known centroids after clipping to 10x10.
    points = np.array([[2., 2.], [8., 2.], [8., 8.], [2., 8.]])
    relaxed = lloyd_relax_points(points, 11, 11, iterations=1)
    np.testing.assert_allclose(relaxed, [[2.5, 2.5], [7.5, 2.5], [7.5, 7.5], [2.5, 7.5]])


def test_resolution_and_regularity_mappings_are_monotonic() -> None:
    parameters = [map_triangulation_controls(value, value, .8, .7) for value in (0, .25, .5, .75, 1)]
    assert [item.total_points for item in parameters] == [400, 1550, 2700, 3850, 5000]
    assert [item.lloyd_iterations for item in parameters] == [0, 1, 2, 3, 4]
    assert all(item.n_coarse + item.n_fine + 4 == item.total_points for item in parameters)


@pytest.mark.parametrize("color_mode", ["mean_vertex", "gouraud"])
def test_renderer_dimensions_color_and_image_orientation(color_mode) -> None:
    image = sample_image()
    points = sample_sites()
    triangles = compute_delaunay_triangles(points)
    rendered = render_triangulation(image, points, triangles, color_mode=color_mode, render_mode="filled")
    assert rendered.shape == image.shape and rendered.dtype == np.uint8
    mesh = render_triangulation_mesh(points, triangles, 20, 16)
    assert mesh.shape == image.shape and mesh.dtype == np.uint8
    if color_mode == "gouraud":
        # This source is linear in x and y; interpolation must reproduce it,
        # including top-to-bottom orientation and complete corner coverage.
        np.testing.assert_array_equal(rendered, image)
    else:
        indices = triangles[-1]
        centroid = points[indices].mean(axis=0)
        x, y = np.round(centroid).astype(int)
        vertex_indices = points[indices].astype(int)
        expected = np.round(image[vertex_indices[:, 1], vertex_indices[:, 0]].mean(axis=0)).astype(np.uint8)
        np.testing.assert_array_equal(rendered[y, x], expected)
        constant_image = np.full_like(image, 91)
        constant_render = render_triangulation(
            constant_image, points, triangles, render_mode="filled"
        )
        # Anchored convex hull must fill the entire rectangle without seams.
        np.testing.assert_array_equal(constant_render, constant_image)


def test_dual_comparison_passes_exact_same_final_array(monkeypatch) -> None:
    captured = {}
    original_delaunay = pipeline.compute_delaunay_triangles
    original_voronoi = pipeline.render_voronoi_cells

    def record_delaunay(points):
        captured["delaunay"] = points
        return original_delaunay(points)

    def record_voronoi(image, points, **kwargs):
        captured["voronoi"] = points
        return original_voronoi(image, points, **kwargs)

    monkeypatch.setattr(pipeline, "compute_delaunay_triangles", record_delaunay)
    monkeypatch.setattr(pipeline, "render_voronoi_cells", record_voronoi)
    result = pipeline.render_triangulation_result(sample_image(), sampling_mode="uniform", structural_resolution=0., regularity=0.)
    assert captured["delaunay"] is captured["voronoi"] is result["points_relaxed"]
    assert "voronoi_vs_triangulation" in result


def test_saved_pipeline_and_cli_preset_overrides(tmp_path) -> None:
    source = save_rgb_image(sample_image(), tmp_path / "input.png")
    destination = tmp_path / "triangulation"
    result = pipeline.run_triangulation_pipeline(
        source, destination, preset="structural", structural_resolution=0.,
        regularity=0., mesh_visibility=0., seed=10,
    )
    assert result["controls"]["structure_emphasis"] == .95
    assert result["controls"]["structural_resolution"] == 0.
    assert result["lloyd_iterations"] == 0
    assert result["point_count"] == 400
    assert (destination / "07_voronoi_vs_triangulation.png").is_file()
    assert (destination / "06_comparison.png").is_file()
    np.testing.assert_array_equal(np.load(destination / "points_initial.npy"), np.load(destination / "points_relaxed.npy"))
    assert json.loads((destination / "config.json").read_text())["comparison_points"] == "points_relaxed.npy"
    args = build_parser().parse_args([
        "triangulation", "--input", str(source), "--output", str(destination),
        "--preset", "structural", "--regularity", "0", "--no-voronoi-comparison",
    ])
    assert args.preset == "structural" and args.regularity == 0
    assert args.voronoi_comparison is False
