"""Readable Lloyd movement layers for process inspection and reveal."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from math_lens.core.geometry import lloyd_relax_points
from math_lens.core.geometry import validate_points_xy
from math_lens.lenses.voronoi.renderer import render_voronoi_cells


def render_point_process(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Neutral site diagram, independent of the photographic input."""
    points = validate_points_xy(points)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for x, y in points:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(45, 45, 45))
    return np.asarray(canvas, dtype=np.uint8).copy()


def render_lloyd_process(
    initial: np.ndarray, width: int, height: int, iterations: int,
    fixed_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Show the first actual Lloyd step with a spatially distributed arrow sample.

    All old Voronoi cell boundaries are subtle gray; at most one moving site
    per 50-pixel grid bin is emphasized. Hollow = old site, filled = its
    first-step cell centroid, line/arrow = movement. With zero iterations no
    invented movement is shown. Final sites appear in a separate layer.
    """

    blank = np.full((height, width, 3), 255, dtype=np.uint8)
    cells = render_voronoi_cells(blank, initial, show_seeds=False, line_value=230, background_value=255)
    targets = lloyd_relax_points(initial, width, height, iterations=min(iterations, 1), fixed_indices=fixed_indices)
    displacement = np.linalg.norm(targets - initial, axis=1)
    selected = {}
    for index, (x, y) in enumerate(initial):
        if index in fixed_indices:
            continue
        bin_key = (int(x // 50), int(y // 50))
        if bin_key not in selected or displacement[index] > displacement[selected[bin_key]]:
            selected[bin_key] = index
    canvas = Image.fromarray(cells)
    draw = ImageDraw.Draw(canvas)
    for index in selected.values():
        old, target = initial[index], targets[index]
        x, y = old
        if iterations > 0 and displacement[index] > 0.5:
            draw.line((tuple(old), tuple(target)), fill=(110, 110, 110), width=1)
            direction = (target - old) / displacement[index]
            normal = np.array([-direction[1], direction[0]])
            for sign in (-1, 1):
                tip = target - direction * 3 + sign * normal * 2
                draw.line((tuple(target), tuple(tip)), fill=(110, 110, 110), width=1)
            tx, ty = target
            draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=(45, 45, 45))
        draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), outline=(65, 65, 65), width=1)
    font = ImageFont.load_default(size=12)
    label = "First Lloyd step: hollow site -> filled centroid" if iterations else "Lloyd paused: 0 steps (sites unchanged)"
    box_width = min(width - 1, int(draw.textlength(label, font=font)) + 18)
    draw.rectangle((4, 4, box_width, 25), fill="white")
    draw.text((9, 8), label, font=font, fill=(45, 45, 45))
    return np.asarray(canvas, dtype=np.uint8).copy()
