"""Shared planar geometry and continuous, rectangle-bounded Lloyd relaxation."""

import operator
from typing import Sequence

import numpy as np
from scipy.spatial import QhullError, Voronoi


def validate_points_xy(points_xy: np.ndarray, minimum: int = 3) -> np.ndarray:
    """Return finite, unique 2D sites as float64, retaining their ordering."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"points_xy must have shape (N, 2) with N >= {minimum}")
    if not np.isfinite(points).all():
        raise ValueError("points_xy must contain only finite coordinates")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("points_xy must contain unique sites")
    return points


def _clip_half_plane(
    polygon: np.ndarray, midpoint: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Clip a convex polygon to the side of a site's neighbor bisector."""

    if len(polygon) == 0:
        return polygon
    distances = (polygon - midpoint) @ normal
    output = []
    previous = polygon[-1]
    previous_distance = distances[-1]
    for current, current_distance in zip(polygon, distances):
        previous_inside = previous_distance <= 0.0
        current_inside = current_distance <= 0.0
        if previous_inside != current_inside:
            fraction = previous_distance / (previous_distance - current_distance)
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous, previous_distance = current, current_distance
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _polygon_centroid(polygon: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Area centroid from the shoelace formula, shifted for numeric stability."""

    if len(polygon) < 3:
        return fallback.copy()
    origin = polygon[0]
    vertices = polygon - origin
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    twice_area = cross.sum()
    if abs(twice_area) <= np.finfo(np.float64).eps:
        return fallback.copy()
    return origin + ((vertices + following) * cross[:, None]).sum(axis=0) / (
        3.0 * twice_area
    )


def lloyd_relax_points(
    points_xy: np.ndarray,
    width: int,
    height: int,
    iterations: int = 2,
    fixed_indices: Sequence[int] = (),
) -> np.ndarray:
    """Move sites to exact centroids of clipped Voronoi cells.

    The rectangle is [0, width-1] x [0, height-1], in image (x, y)
    coordinates. Each iteration constructs scipy.spatial.Voronoi, clips the
    rectangle by every site's neighboring bisectors (including unbounded
    hull cells), and simultaneously replaces sites with polygon centroids.
    fixed_indices can pin boundary anchors; all other cells use ordinary
    unweighted area centroids. This is standard Lloyd relaxation, not a
    density-weighted variant: many iterations weaken adaptive concentration.
    No pixel-grid centroid approximation or random jitter is used.
    """

    points = validate_points_xy(points_xy).copy()
    if width < 2 or height < 2:
        raise ValueError("Lloyd bounds require width and height >= 2")
    try:
        count = operator.index(iterations)
    except TypeError as error:
        raise ValueError("iterations must be a non-negative integer") from error
    if isinstance(iterations, (bool, np.bool_)) or count < 0:
        raise ValueError("iterations must be a non-negative integer")
    upper = np.array([width - 1, height - 1], dtype=np.float64)
    if np.any(points < 0) or np.any(points > upper):
        raise ValueError("points must lie inside the image rectangle")
    fixed = np.zeros(len(points), dtype=bool)
    for index in fixed_indices:
        try:
            integer_index = operator.index(index)
        except TypeError as error:
            raise ValueError("fixed_indices must contain valid site indices") from error
        if integer_index < 0 or integer_index >= len(points):
            raise ValueError("fixed_indices must contain valid site indices")
        fixed[integer_index] = True
    if count == 0:
        return points

    rectangle = np.array(
        [[0.0, 0.0], [upper[0], 0.0], upper, [0.0, upper[1]]]
    )
    for _ in range(count):
        try:
            diagram = Voronoi(points)
        except QhullError as error:
            raise ValueError("Lloyd relaxation requires non-collinear sites") from error
        neighbors: list[list[int]] = [[] for _ in points]
        for first, second in diagram.ridge_points:
            neighbors[first].append(second)
            neighbors[second].append(first)
        updated = points.copy()
        for index, site in enumerate(points):
            if fixed[index]:
                continue
            cell = rectangle.copy()
            for neighbor_index in sorted(neighbors[index]):
                neighbor = points[neighbor_index]
                cell = _clip_half_plane(cell, (site + neighbor) / 2.0, neighbor - site)
            updated[index] = _polygon_centroid(cell, site)
        points = np.clip(updated, 0.0, upper)
    return points
