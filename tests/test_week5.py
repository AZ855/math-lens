import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from math_lens.lenses.triangulation.encodings import (
    build_edge_encoding, representative_edge_colors, rgb_edge_widths,
    triangle_centroid_colors, triangle_edge_adjacency,
)
from math_lens.lenses.triangulation.process import render_lloyd_process
from math_lens.lenses.triangulation.renderer import render_encoded_mesh
from math_lens.ui.layer_reveal import LAYER_NAMES, build_layer_reveal_html
from math_lens.ui.triangulation_page import PROCESS_VIEWS, ordered_process_views, _stages


def sites_and_faces():
    points = np.array([[0., 0.], [63., 0.], [63., 63.], [0., 63.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return points, faces


def rgb_image():
    ys, xs = np.indices((64, 64))
    return np.stack((xs * 4, ys * 3, xs + ys), axis=2).astype(np.uint8)


def test_adjacency_averages_interior_and_preserves_boundary_colors():
    _, faces = sites_and_faces()
    edges, adjacent = triangle_edge_adjacency(faces)
    colors = representative_edge_colors(np.array([[200, 40, 20], [20, 100, 80]]), adjacent)
    interior = np.flatnonzero((edges == [0, 2]).all(axis=1))[0]
    np.testing.assert_array_equal(colors[interior], [110, 70, 50])
    boundary = np.flatnonzero((edges == [0, 1]).all(axis=1))[0]
    np.testing.assert_array_equal(colors[boundary], [200, 40, 20])


def test_centroid_sampling_uses_y_x_and_clamps_safely():
    points, faces = sites_and_faces()
    image = rgb_image()
    colors = triangle_centroid_colors(image, points, faces)
    np.testing.assert_array_equal(colors[0], image[21, 42])
    np.testing.assert_array_equal(colors[1], image[42, 21])
    outside = triangle_centroid_colors(image, points + [100, -100], faces)
    np.testing.assert_array_equal(outside, np.tile(image[0, 63], (2, 1)))


@pytest.mark.parametrize("channel,index", [("R", 0), ("G", 1), ("B", 2)])
def test_widths_are_linear_and_keep_visible_minimum(channel, index):
    colors = np.array([[0, 255, 64], [255, 0, 192]], dtype=np.float64)
    widths = rgb_edge_widths(colors, channel, min_width=.8, max_width=2.8)
    np.testing.assert_allclose(widths, .8 + colors[:, index] / 255 * 2.)
    assert np.all(widths >= .8) and np.all(widths <= 2.8)


def test_combined_mesh_keeps_interiors_white_and_encodings_independent():
    points, faces = sites_and_faces()
    image = rgb_image()
    local = render_encoded_mesh(image, points, faces, "Local Image Color", "Uniform")
    width = render_encoded_mesh(image, points, faces, "Monochrome", "R")
    combined = render_encoded_mesh(image, points, faces, "Local Image Color", "R")
    assert np.any(local[..., 0] != local[..., 1])
    assert np.all(width[..., 0] == width[..., 1])
    assert not np.array_equal(local, combined)
    for mesh in (local, width, combined):
        np.testing.assert_array_equal(mesh[15, 45], [255, 255, 255])
    encoding = build_edge_encoding(image, points, faces, "R")
    assert np.ptp(encoding.widths) > 0


def test_lloyd_layer_generates_movement_and_zero_step_layer():
    points, _ = sites_and_faces()
    initial = np.vstack((points, [[8, 18], [20, 12], [45, 50]]))
    layer = render_lloyd_process(initial, 64, 64, 2)
    paused = render_lloyd_process(initial, 64, 64, 0)
    assert layer.shape == (64, 64, 3) and layer.dtype == np.uint8
    assert not np.array_equal(layer, paused)


def test_layer_payload_order_and_process_order_are_canonical():
    layers = [np.full((8, 12, 3), index, dtype=np.uint8) for index in range(6)]
    html = build_layer_reveal_html(layers)
    data = html.split('<script type="application/json" id="reveal-payload">')[1].split("</script>")[0]
    payload = json.loads(data)
    assert [layer["name"] for layer in payload["layers"]] == list(LAYER_NAMES)
    assert payload["width"] == 12 and payload["height"] == 8
    assert ordered_process_views(list(reversed(PROCESS_VIEWS))) == list(PROCESS_VIEWS)
    with pytest.raises(ValueError, match="six"):
        build_layer_reveal_html(layers[:5])


def test_triangulation_page_loads_and_voronoi_controls_survive_navigation(demo_app: AppTest):
    app = demo_app
    app.select_slider(key="abstraction").set_value(.2).run()
    app.button_group(key="active_lens").set_value("Triangulation").run()
    assert not app.exception
    assert app.title[0].value == "Math Lens — Triangulation Lens"
    assert len(app.select_slider) == 3
    assert not any(exp.label == "Mesh Encoding" for exp in app.expander)
    app.radio(key="tri_display").set_value("Mesh Explorer").run()
    app.radio(key="tri_edge_color").set_value("Local Image Color").run()
    app.radio(key="tri_edge_width").set_value("R").run()
    assert len(app.select_slider) == 4
    assert not app.toggle
    assert not any(exp.label == "Color Rendering" for exp in app.expander)
    app.radio(key="tri_display").set_value("Colored Triangulation").run()
    app.toggle(key="tri_reveal").set_value(True).run()
    assert not app.exception
    assert any(header.value == "Layer Reveal" for header in app.subheader)
    app.multiselect(key="tri_process").set_value(["Triangulation Mesh", "Original", "Relaxed Points"]).run()
    assert app.multiselect(key="tri_process").value == ["Original", "Relaxed Points", "Triangulation Mesh"]
    app.button_group(key="active_lens").set_value("Voronoi").run()
    assert not app.exception
    assert app.select_slider(key="abstraction").value == .2
    app.button_group(key="active_lens").set_value("Triangulation").run()
    assert not app.exception
    app.radio(key="tri_display").set_value("Mesh Explorer").run()
    assert app.radio(key="tri_edge_color").value == "Local Image Color"
    assert app.radio(key="tri_edge_width").value == "R"


def test_process_layers_are_neutral_without_photographic_background():
    image = rgb_image()
    points, faces = sites_and_faces()
    initial = np.vstack((points, [[8, 18], [20, 12], [45, 50]]))
    stages = _stages(image, initial, points, faces, 2)
    assert len(stages) == 4
    for layer in stages:
        assert layer.shape == image.shape and layer.dtype == np.uint8
        assert np.array_equal(layer[..., 0], layer[..., 1])
        assert np.array_equal(layer[..., 1], layer[..., 2])
        assert np.any(np.all(layer == 255, axis=2))
    np.testing.assert_array_equal(stages[0][40, 20], [255, 255, 255])


def test_conditional_sidebar_remembers_hidden_controls_and_reveal_commands(demo_app: AppTest):
    app = demo_app
    app.button_group(key="active_lens").set_value("Triangulation").run()
    app.radio(key="tri_color").set_value("Gouraud").run()
    app.toggle(key="tri_reveal").set_value(True).run()
    app.select_slider(key="tri_brush").set_value(8.0).run()
    app.button(key="reveal_start_button").click().run()
    assert app.session_state["tri_reveal_status"]["active"]
    first_id = app.session_state["tri_reveal_command"]["id"]
    app.button(key="reveal_preview_button").click().run()
    assert app.session_state["tri_reveal_status"]["previewFinal"]
    assert app.session_state["tri_reveal_command"]["id"] != first_id
    app.radio(key="tri_display").set_value("Mesh Explorer").run()
    app.select_slider(key="tri_mesh_visibility").set_value(.3).run()
    app.radio(key="tri_edge_color").set_value("Local Image Color").run()
    app.radio(key="tri_edge_width").set_value("G").run()
    assert not app.session_state["tri_reveal_status"]["active"]
    assert not app.toggle
    app.radio(key="tri_display").set_value("Colored Triangulation").run()
    assert app.radio(key="tri_color").value == "Gouraud"
    assert app.toggle(key="tri_reveal").value
    assert app.select_slider(key="tri_brush").value == 8.0
    app.button(key="reveal_reset_button").click().run()
    assert not app.session_state["tri_reveal_status"]["active"]
    assert not app.session_state["tri_reveal_status"]["previewFinal"]
    app.radio(key="tri_display").set_value("Mesh Explorer").run()
    assert app.select_slider(key="tri_mesh_visibility").value == .3
    assert app.radio(key="tri_edge_color").value == "Local Image Color"
    assert app.radio(key="tri_edge_width").value == "G"
    assert not app.exception


def test_geometry_controls_precede_display_and_keyboard_instructions_are_visible(demo_app: AppTest):
    app = demo_app
    app.button_group(key="active_lens").set_value("Triangulation").run()

    def control_order():
        return [node.label for node in app.sidebar
                if node.type in {"select_slider", "radio", "expander", "toggle"}]

    common = ["Structural Resolution", "Regularity", "Structure Emphasis", "Display Mode"]
    assert control_order() == common + ["Color Rendering", "Color Interpretation", "Layer Reveal"]
    instruction = next(item.value for item in app.sidebar.caption if "Keyboard:" in item.value)
    assert "S — Start / Resume reveal" in instruction
    assert "Q — Pause reveal" in instruction
    assert "Click the canvas" in instruction
    app.toggle(key="tri_reveal").set_value(True).run()
    assert control_order()[-2:] == ["Layer Reveal", "Reveal Brush"]
    assert [button.label for button in app.sidebar.button if button.key.startswith("reveal_")] == [
        "Start / Resume", "Pause", "Preview Final", "Reset",
    ]
    app.radio(key="tri_display").set_value("Mesh Explorer").run()
    assert control_order() == common + ["Mesh Visibility", "Mesh Encoding", "Edge Color", "Edge Width"]
    assert not any("Keyboard:" in item.value for item in app.sidebar.caption)
    assert not app.exception
