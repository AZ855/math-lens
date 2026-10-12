"""Encode six prepared image layers for a browser-owned reveal component."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


LAYER_NAMES = (
    "Original", "Initial Points", "Lloyd Relaxation", "Relaxed Points",
    "Triangulation Mesh", "Final Triangulation Result",
)
ASSETS = Path(__file__).with_name("assets")


def build_layer_reveal_payload(layers: Sequence[np.ndarray]) -> dict:
    """Bottom-to-top layer order; new image content starts a fresh reveal history."""

    if len(layers) != 6:
        raise ValueError("Layer Reveal requires exactly six image layers")
    shape = layers[0].shape
    if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 1:
        raise ValueError("layers must be nonempty RGB images")
    images = []
    digest = hashlib.sha256(b"math-lens-reveal-v2-neutral")
    for name, layer in zip(LAYER_NAMES, layers):
        if layer.shape != shape or layer.dtype != np.uint8:
            raise ValueError("layers must share the same uint8 RGB dimensions")
        buffer = BytesIO()
        Image.fromarray(layer).save(buffer, format="PNG")
        png = buffer.getvalue()
        digest.update(png)
        images.append({"name": name, "src": "data:image/png;base64," + base64.b64encode(png).decode("ascii")})
    return {
        "width": shape[1], "height": shape[0], "layers": images,
        "storageKey": "math-lens-reveal-" + digest.hexdigest(),
    }


def build_layer_reveal_html(layers: Sequence[np.ndarray]) -> str:
    """Self-contained standalone example; the app uses sidebar controls instead."""
    payload = build_layer_reveal_payload(layers)
    template = (ASSETS / "layer_reveal.html").read_text(encoding="utf-8")
    return (template.replace("__PAYLOAD__", json.dumps(payload))
            .replace("__STATE_JS__", (ASSETS / "reveal_state.js").read_text(encoding="utf-8"))
            .replace("__CANVAS_JS__", (ASSETS / "reveal_canvas.js").read_text(encoding="utf-8")))


def render_layer_reveal(payload, command, brush, on_change):
    """A stable bidirectional canvas; only discrete status changes reach Python."""
    import streamlit.components.v1 as components

    component = components.declare_component("math_lens_reveal", path=str(ASSETS))
    return component(payload=payload, command=command, brush=brush,
                     key="layer_reveal_canvas", on_change=on_change, tab_index=0,
                     default=None)
