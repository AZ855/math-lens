"""Live Math Lens navigation; the original Voronoi page remains result-first."""

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import streamlit as st

from math_lens.cli import LENS_PRESETS
from math_lens.core.image_features import (
    compute_edge_map,
    compute_luminance,
    compute_svd_tone,
    normalize_01,
)
from math_lens.core.image_io import load_rgb_bytes
from math_lens.lenses.voronoi.pipeline import render_lens_result
from math_lens.lenses.voronoi.renderer import render_voronoi
from math_lens.lenses.voronoi.sampling import generate_uniform_points
from math_lens.ui.result_layout import result_frame


EXAMPLE_IMAGE = Path(__file__).parent / "assets" / "inputs" / "example.JPG"
PREVIEW_MAX_SIZE = 600
SVD_ENERGY_THRESHOLD = 0.90
PROCESS_VIEWS = ("Original", "Density Map", "Seed Distribution", "Voronoi Cells")
CONTROL_OPTIONS = tuple(index / 100 for index in range(101))
COLOR_MODES = {
    "Mean Region Color": "mean",
    "Seed Point Color": "seed",
}


def abstraction_state(value: float) -> str:
    if value <= 0.33:
        return "Detailed"
    if value <= 0.66:
        return "Balanced"
    return "Abstract"


def detail_state(value: float) -> str:
    if value <= 0.33:
        return "Global"
    if value <= 0.66:
        return "Balanced"
    return "Detail-focused"


def structure_state(value: float) -> str:
    if value <= 0.33:
        return "Tone-focused"
    if value <= 0.66:
        return "Balanced"
    return "Structure-focused"


def semantic_option(
    value: float,
    state,
    left_endpoint: str,
    right_endpoint: str,
) -> str:
    """Format slider endpoints semantically and interior values precisely."""

    if value == 0.0:
        return left_endpoint
    if value == 1.0:
        return right_endpoint
    return f"{state(value)} · {value:.0%}"


def ordered_process_views(selected) -> list[str]:
    """Restore mathematical pipeline order regardless of selection order."""

    return [view for view in PROCESS_VIEWS if view in selected]


@st.cache_data(show_spinner=False)
def _preview_features(data: bytes, svd_energy_threshold: float):
    """Cache the resized image and parameter-independent image analysis."""

    image = load_rgb_bytes(data, max_size=PREVIEW_MAX_SIZE)
    luminance_normalized = normalize_01(compute_luminance(image))
    edge_map = compute_edge_map(image)
    svd_tone = compute_svd_tone(image, svd_energy_threshold)
    return image, luminance_normalized, edge_map, svd_tone


@st.cache_data(show_spinner=False)
def _generate_preview(
    data: bytes,
    abstraction: float,
    detail_emphasis: float,
    structure: float,
    color_mode: str,
    seed: int,
):
    image, luminance_normalized, edge_map, svd_tone = _preview_features(
        data, SVD_ENERGY_THRESHOLD
    )
    return render_lens_result(
        image,
        abstraction=abstraction,
        detail_emphasis=detail_emphasis,
        structure=structure,
        color_mode=color_mode,
        seed=seed,
        luminance_normalized=luminance_normalized,
        edge_map=edge_map,
        tone_method="svd",
        svd_energy_threshold=SVD_ENERGY_THRESHOLD,
        svd_tone_result=svd_tone,
        include_cells=True,
    )


@st.cache_data(show_spinner=False)
def _generate_uniform_comparison(
    data: bytes,
    total_points: int,
    color_mode: str,
    seed: int,
):
    image, _, _, _ = _preview_features(data, SVD_ENERGY_THRESHOLD)
    height, width = image.shape[:2]
    points = generate_uniform_points(width, height, total_points, seed)
    return render_voronoi(image, points, color_mode=color_mode)


def _png_bytes(image) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _initialize_controls() -> None:
    balanced = LENS_PRESETS["balanced"]
    st.session_state.setdefault("preset", "Balanced")
    for name in ("abstraction", "detail_emphasis", "structure"):
        st.session_state.setdefault(name, balanced[name])
    st.session_state.setdefault("color_interpretation", "Mean Region Color")
    st.session_state.setdefault("pattern_seed", 10)
    st.session_state.setdefault("compare_uniform", False)
    st.session_state.setdefault("process_views", list(PROCESS_VIEWS))


def _apply_preset() -> None:
    selected = st.session_state["preset"]
    if selected == "Custom":
        return
    values = LENS_PRESETS[selected.lower()]
    for name in ("abstraction", "detail_emphasis", "structure"):
        st.session_state[name] = values[name]
    st.session_state["color_interpretation"] = (
        "Mean Region Color" if values["color_mode"] == "mean" else "Seed Point Color"
    )


def _mark_custom() -> None:
    st.session_state["preset"] = "Custom"


def _set_structure_quick(value: float) -> None:
    """Apply a demonstration-friendly Tone/Structure position."""

    st.session_state["structure"] = value
    _mark_custom()


def _regenerate_pattern() -> None:
    st.session_state["pattern_seed"] += 1


def _normalize_process_selection() -> None:
    st.session_state["process_views"] = ordered_process_views(
        st.session_state["process_views"]
    )


def _show_process(result) -> None:
    with st.expander("Show Process", expanded=False):
        selected = st.multiselect(
            "Visible process views",
            PROCESS_VIEWS,
            key="process_views",
            on_change=_normalize_process_selection,
            help="Choose which intermediate steps to inspect.",
        )
        st.caption(
            "The Density Map shows where the Lens allocates sampling "
            "resolution. Brighter areas have higher sampling probability and "
            "tend to receive more Voronoi sites. It blends SVD Tone with Sobel "
            "Structure."
        )
        ordered_views = ordered_process_views(selected)
        images = {
            "Original": result["original"],
            "Density Map": result["density_rgb"],
            "Seed Distribution": result["points_overlay"],
            "Voronoi Cells": result["voronoi_cells"],
        }
        if not ordered_views:
            st.caption("Select a process view above.")
            return
        column_count = {1: 1, 2: 2, 3: 3, 4: 2}[len(ordered_views)]
        for row_start in range(0, len(ordered_views), column_count):
            row_views = ordered_views[row_start : row_start + column_count]
            columns = st.columns(len(row_views))
            for column, name in zip(columns, row_views, strict=True):
                with column:
                    st.image(images[name], caption=name, width="stretch")


def _show_feature_analysis(result) -> None:
    """Keep research-oriented feature maps separate from the main pipeline."""

    if "svd_reconstruction_rgb" not in result:
        return
    metadata = result["feature_metadata"]
    with st.expander("Feature Analysis", expanded=False):
        st.caption(
            "The luminance image is centered before SVD so the decomposition "
            "focuses on dominant tonal variation rather than overall brightness."
        )
        reconstruction, tone, structure = st.columns(3)
        with reconstruction:
            st.image(
                result["svd_reconstruction_rgb"],
                caption="SVD Tonal Reconstruction",
                width="stretch",
            )
        with tone:
            st.image(
                result["tone_map_rgb"],
                caption="SVD Tone Map",
                width="stretch",
            )
        with structure:
            st.image(
                result["structure_map_rgb"],
                caption="Sobel Structure Map",
                width="stretch",
            )
        st.caption(
            f"Tone method: **SVD** · Selected rank: "
            f"**{metadata['svd_rank']}** · Retained energy: "
            f"**{metadata['svd_retained_energy']:.1%}** "
            f"(target {metadata['svd_energy_threshold']:.0%})"
        )


def _show_interpretation(result) -> None:
    controls = result["controls"]
    parameters = result["resolved_parameters"]
    with st.expander("How the Lens interpreted this image", expanded=False):
        first, second = st.columns(2)
        with first:
            st.markdown(f"**Abstraction: {abstraction_state(controls['abstraction'])}**")
            st.caption(f"Internal value: {controls['abstraction']:.2f}")
            st.write(
                f"→ {parameters['total_points']:,} sites  "
                f"\n→ density smoothing {parameters['blur_sigma']:.2f}"
            )
            st.markdown(
                f"**Detail Emphasis: {detail_state(controls['detail_emphasis'])}**"
            )
            st.caption(f"Internal value: {controls['detail_emphasis']:.2f}")
            st.write(
                f"→ {parameters['n_coarse']:,} coarse / "
                f"{parameters['n_fine']:,} fine sites  "
                f"\n→ density power {parameters['density_power']:.2f}"
            )
        with second:
            st.markdown(
                f"**Structure vs Tone: {structure_state(controls['structure'])}**"
            )
            st.caption(f"Internal value: {controls['structure']:.2f}")
            st.write(
                f"→ {parameters['edge_weight']:.0%} Sobel structure  "
                f"\n→ {parameters['tone_weight']:.0%} SVD tone"
            )
            metadata = result["feature_metadata"]
            st.caption(
                f"SVD rank {metadata['svd_rank']} · "
                f"{metadata['svd_retained_energy']:.1%} energy retained"
            )
            st.markdown("**Color Interpretation**")
            st.write(
                "→ Mean Region Color"
                if controls["color_mode"] == "mean"
                else "→ Seed Point Color"
            )

        st.caption(
            "Voronoi Cells shows the nearest-neighbor partition before color "
            "is assigned to each region."
        )
        st.markdown(
            "A low-rank SVD of centered luminance captures broad tonal variation, "
            "while Sobel magnitude captures local transitions. Centering removes "
            "overall brightness from the SVD energy. Their weighted blend forms "
            "the sampling density. Each pixel then joins its nearest seed, and "
            "the resulting region receives mean or seed color."
        )
        st.latex(r"V_i = \{x : d(x,p_i) \leq d(x,p_j) \text{ for all } j\}")


def _sidebar_controls():
    with st.sidebar:
        st.markdown(
            """
            <style>
            .st-key-structure_quick_row button {
                padding-left: 0.3rem;
                padding-right: 0.3rem;
            }
            .st-key-structure_quick_row button p {
                font-size: 0.8rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.header("Voronoi Lens Controls")
        upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
        st.selectbox(
            "Preset",
            ("Custom", "Detailed", "Balanced", "Abstract"),
            key="preset",
            on_change=_apply_preset,
        )
        abstraction = st.select_slider(
            "Abstraction",
            options=CONTROL_OPTIONS,
            format_func=lambda value: semantic_option(
                value, abstraction_state, "Detailed", "Abstract"
            ),
            help=(
                "Controls cell scale: lower values preserve more detail; "
                "higher values create a more abstract result."
            ),
            key="abstraction",
            on_change=_mark_custom,
        )
        detail_emphasis = st.select_slider(
            "Detail Emphasis",
            options=CONTROL_OPTIONS,
            format_func=lambda value: semantic_option(
                value, detail_state, "Global", "Detail-focused"
            ),
            help=(
                "Controls where sites concentrate: lower values spread them "
                "globally; higher values focus on local image features."
            ),
            key="detail_emphasis",
            on_change=_mark_custom,
        )
        structure = st.select_slider(
            "Structure vs Tone",
            options=CONTROL_OPTIONS,
            format_func=lambda value: semantic_option(
                value, structure_state, "Tone", "Structure"
            ),
            help=(
                "Tone uses an SVD-based low-rank reconstruction of centered "
                "luminance to capture broad tonal variation without overall "
                "brightness dominating. Structure uses Sobel edge magnitude "
                "to capture local intensity transitions."
            ),
            key="structure",
            on_change=_mark_custom,
        )
        with st.container(key="structure_quick_row"):
            tone_button, balanced_button, structure_button = st.columns(3)
            quick_positions = (
                (tone_button, "Tone", 0.0),
                (balanced_button, "Balanced", 0.5),
                (structure_button, "Structure", 1.0),
            )
            for column, label, value in quick_positions:
                with column:
                    st.button(
                        label,
                        key=f"structure_quick_{label.lower()}",
                        on_click=_set_structure_quick,
                        args=(value,),
                        type="primary" if structure == value else "secondary",
                        width="stretch",
                    )
        color_interpretation = st.radio(
            "Color Interpretation",
            tuple(COLOR_MODES),
            key="color_interpretation",
            on_change=_mark_custom,
            help=(
                "Mean Region Color averages source pixels in each cell. "
                "Seed Point Color samples the generator point."
            ),
        )
        st.button(
            "Regenerate Pattern",
            on_click=_regenerate_pattern,
            help="Change only the random realization; controls keep this seed fixed.",
            width="stretch",
        )
        st.caption(f"Pattern seed: {st.session_state['pattern_seed']}")
        with st.expander("Comparison", expanded=False):
            compare_uniform = st.checkbox(
                "Compare with Uniform Baseline",
                key="compare_uniform",
                help="Uses the same total number of sites as the Lens.",
            )
    return (
        upload,
        abstraction,
        detail_emphasis,
        structure,
        color_interpretation,
        compare_uniform,
    )


def _voronoi_page() -> None:
    _initialize_controls()
    (
        upload,
        abstraction,
        detail_emphasis,
        structure,
        color_interpretation,
        compare_uniform,
    ) = _sidebar_controls()

    st.title("Math Lens — Voronoi Lens")
    st.write(
        "Explore how the same image can be interpreted as a collection of "
        "nearest-neighbor regions."
    )

    if upload is not None:
        image_data = upload.getvalue()
        image_name = upload.name
        st.session_state["voronoi_image_data"] = image_data
        st.session_state["voronoi_image_name"] = image_name
    elif "voronoi_image_data" in st.session_state:
        image_data = st.session_state["voronoi_image_data"]
        image_name = st.session_state["voronoi_image_name"]
    elif EXAMPLE_IMAGE.is_file():
        image_data = EXAMPLE_IMAGE.read_bytes()
        image_name = EXAMPLE_IMAGE.name
    else:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return

    try:
        with st.spinner("Updating the Lens…"):
            result = _generate_preview(
                image_data,
                abstraction,
                detail_emphasis,
                structure,
                COLOR_MODES[color_interpretation],
                int(st.session_state["pattern_seed"]),
            )
    except (OSError, ValueError) as error:
        st.error(f"Could not process this image: {error}")
        return

    height, width = result["original"].shape[:2]
    config = {
        "input_name": image_name,
        "sampling_mode": "lens",
        "pattern_seed": int(st.session_state["pattern_seed"]),
        "preview_max_size": PREVIEW_MAX_SIZE,
        "preview_width": width,
        "preview_height": height,
        "controls": result["controls"],
        "resolved_parameters": result["resolved_parameters"],
        **result["feature_metadata"],
        "compare_uniform": compare_uniform,
    }

    st.subheader("Voronoi Result")
    with result_frame():
        st.image(result["voronoi"], width="stretch")
    st.caption(
        f"Live {width}×{height} preview · pattern seed "
        f"{st.session_state['pattern_seed']}"
    )

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Download Voronoi PNG",
            data=_png_bytes(result["voronoi"]),
            file_name="math_lens_voronoi.png",
            mime="image/png",
            width="stretch",
        )
    with download_right:
        st.download_button(
            "Download config JSON",
            data=json.dumps(config, indent=2) + "\n",
            file_name="math_lens_config.json",
            mime="application/json",
            width="stretch",
        )

    _show_process(result)
    _show_feature_analysis(result)

    if compare_uniform:
        uniform = _generate_uniform_comparison(
            image_data,
            result["resolved_parameters"]["total_points"],
            result["controls"]["color_mode"],
            int(st.session_state["pattern_seed"]),
        )
        with st.expander("Uniform Baseline Comparison", expanded=False):
            uniform_column, lens_column = st.columns(2)
            with uniform_column:
                st.image(uniform, caption="Uniform Baseline", width="stretch")
            with lens_column:
                st.image(result["voronoi"], caption="Voronoi Lens", width="stretch")
            st.caption("Both use the same total number of sites and color rule.")

    _show_interpretation(result)


def _switch_lens() -> None:
    """Keep each page's widget values when Streamlit removes hidden widgets."""
    previous = st.session_state.get("previous_lens", "Voronoi")
    voronoi_keys = {
        "preset", "abstraction", "detail_emphasis", "structure", "color_interpretation",
        "pattern_seed", "compare_uniform", "process_views",
    }
    values = {
        key: st.session_state[key] for key in st.session_state
        if (key in voronoi_keys if previous == "Voronoi" else key.startswith("tri_") and key not in {"tri_upload", "tri_regenerate"})
    }
    st.session_state["saved_" + previous] = values
    selected = st.session_state["active_lens"] or previous
    if st.session_state["active_lens"] is None:
        st.session_state["active_lens"] = selected
    for key, value in st.session_state.get("saved_" + selected, {}).items():
        st.session_state[key] = value
    st.session_state["previous_lens"] = selected


def main() -> None:
    st.set_page_config(page_title="Math Lens", layout="wide")
    st.session_state.setdefault("active_lens", "Voronoi")
    st.sidebar.segmented_control(
        "Lens", ("Voronoi", "Triangulation"),
        selection_mode="single", key="active_lens", on_change=_switch_lens,
    )
    if st.session_state["active_lens"] == "Triangulation":
        from math_lens.ui.triangulation_page import render_page
        render_page(EXAMPLE_IMAGE)
    else:
        _voronoi_page()


if __name__ == "__main__":
    main()
