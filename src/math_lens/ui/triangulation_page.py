"""Result-first Triangulation page with independent cached computation stages."""

from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image
import streamlit as st

from math_lens.core.image_features import compute_edge_map
from math_lens.core.image_io import load_rgb_bytes
from math_lens.core.parameters import map_triangulation_controls
from math_lens.lenses.triangulation.encodings import mesh_visibility_style
from math_lens.lenses.triangulation.pipeline import (
    TRIANGULATION_DEFAULTS, TRIANGULATION_PRESETS, compute_triangulation_geometry,
)
from math_lens.lenses.triangulation.process import render_lloyd_process, render_point_process
from math_lens.lenses.triangulation.renderer import render_encoded_mesh, render_triangulation
from .layer_reveal import LAYER_NAMES, build_layer_reveal_payload, render_layer_reveal
from .result_layout import result_frame


PREVIEW_SIZE = 600
PROCESS_VIEWS = (
    "Original", "Initial Points", "Lloyd Relaxation", "Relaxed Points", "Triangulation Mesh",
)
CONTROL_OPTIONS = tuple(index / 100 for index in range(101))
COLOR_OPTIONS = {"Mean Vertex": "mean_vertex", "Gouraud": "gouraud"}
MODE_WIDGETS = (
    "tri_mesh_visibility", "tri_color", "tri_show_edges", "tri_edge_color",
    "tri_edge_width", "tri_reveal", "tri_brush",
)


def ordered_process_views(selected) -> list[str]:
    return [view for view in PROCESS_VIEWS if view in selected]


def _png_bytes(image) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _image_features(data):
    image = load_rgb_bytes(data, max_size=PREVIEW_SIZE)
    return image, compute_edge_map(image)


@st.cache_data(show_spinner=False)
def _geometry(data, resolution, regularity, structure, seed):
    image, edges = _image_features(data)
    return compute_triangulation_geometry(
        image, structural_resolution=resolution, regularity=regularity,
        structure_emphasis=structure, seed=seed, edge_map=edges,
    )


@st.cache_data(show_spinner=False)
def _color(image, points, triangles, color_mode, visibility, show_edges):
    parameters = map_triangulation_controls(.5, .5, .8, visibility)
    return render_triangulation(
        image, points, triangles, color_mode=color_mode,
        render_mode="filled_mesh" if show_edges else "filled",
        line_width=parameters.line_width, line_alpha=parameters.line_alpha,
    )


@st.cache_data(show_spinner=False)
def _mesh(image, points, triangles, color_mode, channel, visibility):
    return render_encoded_mesh(
        image, points, triangles, edge_color_mode=color_mode, width_channel=channel,
        **mesh_visibility_style(visibility),
    )


@st.cache_data(show_spinner=False)
def _stages(image, initial, relaxed, triangles, iterations):
    height, width = image.shape[:2]
    return (
        render_point_process(initial, width, height),
        render_lloyd_process(initial, width, height, iterations),
        render_point_process(relaxed, width, height),
        _mesh(image, relaxed, triangles, "Monochrome", "Uniform", .5),
    )


@st.cache_data(show_spinner=False)
def _reveal_payload(layers):
    return build_layer_reveal_payload(layers)


def _initialize():
    # A hidden conditional widget is deleted by Streamlit. Keep its model value
    # separately, then restore it before the widget is rendered again.
    remembered = st.session_state.setdefault("_tri_mode_values", {})
    for name in MODE_WIDGETS:
        if name in remembered:
            st.session_state[name] = remembered[name]
    for name, value in TRIANGULATION_DEFAULTS.items():
        st.session_state.setdefault("tri_" + name, value)
    defaults = {
        "tri_preset": "Balanced", "tri_seed": 10,
        "tri_display": "Colored Triangulation", "tri_color": "Mean Vertex",
        "tri_show_edges": True, "tri_edge_color": "Monochrome", "tri_edge_width": "Uniform",
        "tri_process": list(PROCESS_VIEWS), "tri_reveal": False,
        "tri_brush": 5.0, "tri_reveal_command": None,
        "tri_reveal_status": {"active": False, "previewFinal": False},
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)
    # Preserve selections from the previous UI's color labels.
    legacy = {"Mean Vertex Color": "Mean Vertex", "Interpolated Color": "Gouraud"}
    st.session_state["tri_color"] = legacy.get(st.session_state["tri_color"], st.session_state["tri_color"])
    for name in MODE_WIDGETS:
        remembered.setdefault(name, st.session_state[name])
    remembered["tri_color"] = st.session_state["tri_color"]


def _remember(name):
    st.session_state["_tri_mode_values"][name] = st.session_state[name]


def _reveal_command(action):
    st.session_state["tri_reveal_command"] = {
        "id": uuid4().hex, "action": action,
        "storageKey": st.session_state.get("tri_reveal_layer_key"),
    }
    status = dict(st.session_state["tri_reveal_status"])
    if action in {"pause", "reset"}:
        status["active"] = False
    elif action == "resume":
        status["active"] = True
    if action == "reset":
        status["previewFinal"] = False
    elif action == "preview":
        status["previewFinal"] = not status.get("previewFinal", False)
    st.session_state["tri_reveal_status"] = status


def _display_changed():
    if st.session_state["tri_display"] == "Mesh Explorer":
        _reveal_command("pause")


def _reveal_toggled():
    _remember("tri_reveal")
    if not st.session_state["tri_reveal"]:
        _reveal_command("pause")


def _reveal_event():
    event = st.session_state.get("layer_reveal_canvas")
    if isinstance(event, dict) and event.get("storageKey") == st.session_state.get("tri_reveal_layer_key"):
        st.session_state["tri_reveal_status"] = event


def _reveal_sidebar():
    enabled = st.toggle("Layer Reveal", key="tri_reveal", on_change=_reveal_toggled,
                        help="A local brush descends through six stages. Focus the canvas for S / Q.")
    st.caption("Keyboard:  \nS — Start / Resume reveal  \nQ — Pause reveal  \nClick the canvas first to focus it.")
    if not enabled:
        return
    left, right = st.columns(2)
    with left:
        st.button("Start / Resume", key="reveal_start_button", on_click=_reveal_command,
                  args=("resume",), width="stretch")
    with right:
        st.button("Pause", key="reveal_pause_button", on_click=_reveal_command,
                  args=("pause",), width="stretch")
    left, right = st.columns(2)
    with left:
        preview = st.session_state["tri_reveal_status"].get("previewFinal", False)
        st.button("Return to Reveal" if preview else "Preview Final", key="reveal_preview_button",
                  on_click=_reveal_command, args=("preview",), width="stretch")
    with right:
        st.button("Reset", key="reveal_reset_button", on_click=_reveal_command,
                  args=("reset",), width="stretch")
    st.select_slider("Reveal Brush", options=tuple(2 + index / 2 for index in range(27)),
                     key="tri_brush", format_func=lambda value: "Small" if value == 2 else "Large" if value == 15 else f"{value:g}%",
                     on_change=_remember, args=("tri_brush",),
                     help="Radius is 2–15% of displayed image width; default 5%.")
    status = st.session_state["tri_reveal_status"]
    st.caption("Reveal Active" if status.get("active") else "Reveal Paused")
    if not status.get("storageAvailable", True):
        st.caption("Session save unavailable; current canvas still works.")


def _preset():
    name = st.session_state["tri_preset"].lower()
    if name != "custom":
        for control, value in TRIANGULATION_PRESETS[name].items():
            st.session_state["tri_" + control] = value
            if "tri_" + control in MODE_WIDGETS:
                _remember("tri_" + control)


def _custom(name=None):
    if name in MODE_WIDGETS:
        _remember(name)
    st.session_state["tri_preset"] = "Custom"


def _regenerate():
    st.session_state["tri_seed"] += 1


def _normalize_process():
    st.session_state["tri_process"] = ordered_process_views(st.session_state["tri_process"])


def _semantic(value, left, right):
    if value == 0:
        return left
    if value == 1:
        return right
    name = left if value < .34 else right if value > .66 else "Balanced"
    return f"{name} · {value:.0%}"


def _sidebar():
    with st.sidebar:
        # Streamlit otherwise fades its native endpoint labels when not hovered.
        # Keep the semantic range visible without adding another explanatory row.
        st.html("<style>[data-testid='stSidebar'] [data-testid='stSliderTickBar'] {opacity: 1 !important;}</style>")
        st.header("Triangulation Lens Controls")
        upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="tri_upload")
        st.selectbox("Preset", ("Custom", "Sparse", "Balanced", "Structural"), key="tri_preset", on_change=_preset)
        definitions = (
            ("structural_resolution", "Structural Resolution", "Sparse", "Dense", "Controls the scale of the mesh: sparse uses larger triangles; dense follows finer detail."),
            ("regularity", "Regularity", "Organic", "Regular", "A few Lloyd steps organize spacing. Stronger regularity can soften concentration around image edges."),
            ("structure_emphasis", "Structure Emphasis", "Broad", "Edge-focused", "Moves sampling from broad coverage toward the existing Sobel image structure."),
        )
        for name, label, left, right, help_text in definitions:
            st.select_slider(
                label, options=CONTROL_OPTIONS, key="tri_" + name,
                format_func=lambda value, l=left, r=right: _semantic(value, l, r),
                help=help_text, on_change=_custom, args=("tri_" + name,),
            )
        st.radio("Display Mode", ("Colored Triangulation", "Mesh Explorer"), key="tri_display", on_change=_display_changed)
        explorer = st.session_state["tri_display"] == "Mesh Explorer"
        if explorer:
            st.select_slider(
                "Mesh Visibility", options=CONTROL_OPTIONS, key="tri_mesh_visibility",
                format_func=lambda value: _semantic(value, "Subtle", "Explicit"),
                help="Controls the strength of visible mesh lines. Explorer lines retain a visible minimum width.",
                on_change=_custom, args=("tri_mesh_visibility",),
            )
            with st.expander("Mesh Encoding", expanded=True):
                st.radio("Edge Color", ("Monochrome", "Local Image Color"), key="tri_edge_color", horizontal=True,
                         on_change=_remember, args=("tri_edge_color",),
                         help="Local color samples triangle centroids and averages the two neighbors along interior edges.")
                st.radio("Edge Width", ("Uniform", "R", "G", "B"), key="tri_edge_width", horizontal=True,
                         on_change=_remember, args=("tri_edge_width",),
                         help="Channel intensity linearly controls width, independently of edge color.")
        else:
            with st.expander("Color Rendering", expanded=True):
                st.radio("Color Interpretation", tuple(COLOR_OPTIONS), key="tri_color", horizontal=True,
                         on_change=_remember, args=("tri_color",))
                st.checkbox("Show mesh edges", key="tri_show_edges", on_change=_remember, args=("tri_show_edges",))
            _reveal_sidebar()
        st.button("Regenerate Pattern", key="tri_regenerate", on_click=_regenerate, width="stretch")
        st.caption(f"Pattern seed: {st.session_state['tri_seed']}")
    return upload


def render_page(example_image: Path) -> None:
    _initialize()
    upload = _sidebar()
    st.title("Math Lens — Triangulation Lens")
    st.write("Explore the image as connections, a mesh, and a structural skeleton.")
    if upload is not None:
        st.session_state["tri_image_data"] = upload.getvalue()
        st.session_state["tri_image_name"] = upload.name
    if "tri_image_data" in st.session_state:
        data, name = st.session_state["tri_image_data"], st.session_state["tri_image_name"]
    elif example_image.is_file():
        data, name = example_image.read_bytes(), example_image.name
    else:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return
    state = st.session_state
    try:
        with st.spinner("Updating the Lens…"):
            geometry = _geometry(data, state["tri_structural_resolution"], state["tri_regularity"], state["tri_structure_emphasis"], state["tri_seed"])
            image, points, triangles = geometry["original"], geometry["points_relaxed"], geometry["triangles"]
            colored = _color(image, points, triangles, COLOR_OPTIONS[state["tri_color"]], state["tri_mesh_visibility"], state["tri_show_edges"])
            mesh = _mesh(image, points, triangles, state["tri_edge_color"], state["tri_edge_width"], state["tri_mesh_visibility"])
            initial, lloyd, relaxed, process_mesh = _stages(image, geometry["points_initial"], points, triangles, geometry["resolved_parameters"]["lloyd_iterations"])
    except (OSError, ValueError) as error:
        st.error(f"Could not process this image: {error}")
        return

    reveal = state["tri_display"] == "Colored Triangulation" and state["tri_reveal"]
    st.subheader("Layer Reveal" if reveal else state["tri_display"])
    with result_frame():
        if reveal:
            payload = _reveal_payload([image, initial, lloyd, relaxed, process_mesh, colored])
            state["tri_reveal_layer_key"] = payload["storageKey"]
            render_layer_reveal(payload, state["tri_reveal_command"], state["tri_brush"], _reveal_event)
        else:
            st.image(mesh if state["tri_display"] == "Mesh Explorer" else colored, width="stretch")
    height, width = image.shape[:2]
    st.caption(f"Live {width}×{height} preview · pattern seed {state['tri_seed']}")
    controls = {name: state["tri_" + name] for name in TRIANGULATION_DEFAULTS}
    config = {
        "lens": "triangulation", "input_name": name, "preview_max_size": PREVIEW_SIZE,
        "preview_width": width, "preview_height": height, "seed": state["tri_seed"],
        "controls": controls, "resolved_parameters": vars(map_triangulation_controls(**controls)),
        "display_mode": state["tri_display"], "color_mode": COLOR_OPTIONS[state["tri_color"]],
        "show_mesh_edges": state["tri_show_edges"], "edge_color": state["tri_edge_color"],
        "edge_width_channel": state["tri_edge_width"],
        "mesh_encoding_style": mesh_visibility_style(state["tri_mesh_visibility"]),
        "reveal_layers_bottom_to_top": LAYER_NAMES,
        "reveal_brush_relative_radius": {"min": .02, "default": .05, "max": .15},
        "reveal_state": "browser session; keyed by prepared image layers",
    }
    for column, (label, content, filename, mime) in zip(st.columns(3), (
        ("Download Triangulation PNG", _png_bytes(colored), "math_lens_triangulation.png", "image/png"),
        ("Download Mesh PNG", _png_bytes(mesh), "math_lens_mesh.png", "image/png"),
        ("Download config JSON", json.dumps(config, indent=2) + "\n", "math_lens_triangulation.json", "application/json"),
    )):
        with column:
            st.download_button(label, content, file_name=filename, mime=mime, width="stretch")
    with st.expander("Show Process", expanded=False):
        selected = st.multiselect("Visible process views", PROCESS_VIEWS, key="tri_process", on_change=_normalize_process)
        images = dict(zip(PROCESS_VIEWS, [image, initial, lloyd, relaxed, process_mesh]))
        ordered = ordered_process_views(selected)
        for start in range(0, len(ordered), 2):
            row = ordered[start:start + 2]
            for column, label in zip(st.columns(len(row)), row):
                with column:
                    st.image(images[label], caption=label, width="stretch")
        if "Lloyd Relaxation" in ordered:
            st.caption("Hollow old sites → filled first-step cell centroids; light-gray boundaries show the old Voronoi cells. Final sites appear in Relaxed Points.")
    with st.expander("How the Lens interpreted this image", expanded=False):
        st.write(f"{len(points):,} sites · {len(triangles):,} Delaunay triangles · {geometry['resolved_parameters']['lloyd_iterations']} Lloyd steps")
        st.caption("Four corner anchors keep the image covered. Delaunay connects neighboring sites; a few unweighted Lloyd steps regularize their spacing.")
        st.write("Mesh Explorer carries local centroid color along connections. Edge Width uses the same representative color, even with monochrome lines.")
