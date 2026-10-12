"""Generate the Week 5 mesh encodings and a self-contained reveal example."""

import argparse
import json
from pathlib import Path

from math_lens.core.image_io import load_rgb_image, save_rgb_image
from math_lens.core.visualization import make_image_panel
from math_lens.lenses.triangulation.encodings import mesh_visibility_style
from math_lens.lenses.triangulation.pipeline import render_triangulation_result
from math_lens.lenses.triangulation.process import render_lloyd_process, render_point_process
from math_lens.lenses.triangulation.renderer import render_encoded_mesh
from math_lens.ui.layer_reveal import LAYER_NAMES, build_layer_reveal_html


def generate_examples(input_path: Path, output_dir: Path, max_size: int = 600, seed: int = 10):
    image = load_rgb_image(input_path, max_size=max_size)
    result = render_triangulation_result(image, seed=seed, include_voronoi_comparison=False)
    points, faces = result["points_relaxed"], result["triangles"]
    style = mesh_visibility_style(.7)
    local = render_encoded_mesh(image, points, faces, "Local Image Color", "Uniform", **style)
    red = render_encoded_mesh(image, points, faces, "Monochrome", "R", **style)
    combined = render_encoded_mesh(image, points, faces, "Local Image Color", "R", **style)
    height, width = image.shape[:2]
    lloyd = render_lloyd_process(result["points_initial"], width, height, result["resolved_parameters"]["lloyd_iterations"])
    process_mesh = render_encoded_mesh(image, points, faces, "Monochrome", "Uniform", **style)
    layers = [image, render_point_process(result["points_initial"], width, height), lloyd,
              render_point_process(points, width, height), process_mesh, result["color"]]
    outputs = {
        "local_color_mesh.png": local, "r_width_mesh.png": red,
        "local_color_r_width_mesh.png": combined, "lloyd_process.png": lloyd,
        "mesh_encoding_comparison.png": make_image_panel(
            [local, red, combined], ["Local Image Color", "Monochrome + R Width", "Local Color + R Width"],
            "Same final sites and connectivity | white triangle interiors | independent color and width encodings",
        ),
    }
    for index, layer in enumerate(layers):
        outputs[f"layer_{index:02d}.png"] = layer
    for filename, array in outputs.items():
        save_rgb_image(array, output_dir / filename)
    (output_dir / "layer_reveal.html").write_text(build_layer_reveal_html(layers), encoding="utf-8")
    config = {
        "input": str(input_path), "max_size": max_size, "seed": seed,
        "width": width, "height": height, "controls": result["controls"],
        "point_count": len(points), "triangle_count": len(faces),
        "mesh_encoding_style": style, "layers_bottom_to_top": LAYER_NAMES,
        "files": list(outputs) + ["layer_reveal.html", "config.json"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("assets/inputs/example.JPG"))
    parser.add_argument("--output", type=Path, default=Path("outputs/week5"))
    parser.add_argument("--max-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(generate_examples(args.input, args.output, args.max_size, args.seed), indent=2))


if __name__ == "__main__":
    main()
