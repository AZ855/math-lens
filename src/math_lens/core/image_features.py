"""Simple image measurements used to guide adaptive sampling."""

from dataclasses import dataclass
import math
import operator

import numpy as np
from scipy.ndimage import sobel


@dataclass(frozen=True)
class SVDToneResult:
    """Low-rank analysis of luminance variation around its global mean.

    ``reconstruction`` is display-friendly luminance (``X_k + mean``), while
    ``centered_reconstruction`` is the actual low-rank variation ``X_k`` used
    to construct ``tone_map``.
    """

    reconstruction: np.ndarray
    centered_reconstruction: np.ndarray
    tone_map: np.ndarray
    singular_values: np.ndarray
    rank: int
    retained_energy: float
    luminance_mean: float


@dataclass(frozen=True)
class CenteredSVDResult:
    """Reusable centered-luminance decomposition for rank experiments."""

    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_transpose: np.ndarray
    luminance_mean: float


def _validate_rgb(rgb_uint8: np.ndarray) -> None:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must have shape (height, width, 3)")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must have dtype uint8")
    if rgb_uint8.shape[0] == 0 or rgb_uint8.shape[1] == 0:
        raise ValueError("rgb_uint8 must have nonzero height and width")


def compute_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert RGB values to perceptual luminance in the range ``[0, 1]``."""

    _validate_rgb(rgb_uint8)
    image = rgb_uint8.astype(np.float32) / 255.0
    luminance = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )
    return luminance.astype(np.float32, copy=False)


def normalize_01(x: np.ndarray) -> np.ndarray:
    """Linearly scale a nonempty numeric array into ``[0, 1]`` safely."""

    values = np.asarray(x, dtype=np.float32)
    if values.size == 0:
        raise ValueError("cannot normalize an empty array")
    if not np.isfinite(values).all():
        raise ValueError("array must contain only finite values")

    minimum = float(values.min())
    maximum = float(values.max())
    return ((values - minimum) / (maximum - minimum + 1e-12)).astype(
        np.float32, copy=False
    )


def compute_centered_svd(rgb_uint8: np.ndarray) -> CenteredSVDResult:
    """Compute one centered-luminance SVD that can be reused across ranks."""

    luminance = compute_luminance(rgb_uint8)
    luminance_mean = float(luminance.mean(dtype=np.float64))
    centered_luminance = (luminance - np.float32(luminance_mean)).astype(
        np.float32, copy=False
    )
    try:
        left, singular_values, right_transpose = np.linalg.svd(
            centered_luminance,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("SVD tonal analysis failed to converge") from error
    return CenteredSVDResult(
        left_vectors=left.astype(np.float32, copy=False),
        singular_values=singular_values.astype(np.float32, copy=False),
        right_transpose=right_transpose.astype(np.float32, copy=False),
        luminance_mean=luminance_mean,
    )


def cumulative_singular_value_energy(singular_values: np.ndarray) -> np.ndarray:
    """Return cumulative squared singular-value energy in ``[0, 1]``."""

    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a nonempty 1D array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("singular_values must be finite and non-negative")
    squared = np.square(values)
    total_energy = float(squared.sum())
    if total_energy <= np.finfo(np.float64).eps:
        return np.ones(values.shape, dtype=np.float64)
    return np.cumsum(squared) / total_energy


def resolve_svd_rank(rank: int, total_singular_values: int) -> int:
    """Validate a requested rank and clamp it to the available components."""

    if isinstance(rank, (bool, np.bool_)):
        raise ValueError("rank must be a positive integer")
    try:
        requested = operator.index(rank)
    except TypeError as error:
        raise ValueError("rank must be a positive integer") from error
    if isinstance(total_singular_values, (bool, np.bool_)):
        raise ValueError("total_singular_values must be at least 1")
    try:
        available = operator.index(total_singular_values)
    except TypeError as error:
        raise ValueError("total_singular_values must be at least 1") from error
    if requested < 1:
        raise ValueError("rank must be a positive integer")
    if available < 1:
        raise ValueError("total_singular_values must be at least 1")
    return min(requested, available)


def reconstruct_svd_tone(
    decomposition: CenteredSVDResult,
    rank: int,
) -> SVDToneResult:
    """Construct display luminance and Tone from a fixed centered-SVD rank."""

    selected_rank = resolve_svd_rank(rank, decomposition.singular_values.size)
    retained_energy = float(
        cumulative_singular_value_energy(decomposition.singular_values)[
            selected_rank - 1
        ]
    )
    centered_reconstruction = (
        (
            decomposition.left_vectors[:, :selected_rank]
            * decomposition.singular_values[:selected_rank]
        )
        @ decomposition.right_transpose[:selected_rank, :]
    ).astype(np.float32, copy=False)
    reconstruction = (
        centered_reconstruction + np.float32(decomposition.luminance_mean)
    ).astype(np.float32, copy=False)
    tone_map = normalize_01(np.abs(centered_reconstruction))
    return SVDToneResult(
        reconstruction=reconstruction,
        centered_reconstruction=centered_reconstruction,
        tone_map=tone_map,
        singular_values=decomposition.singular_values,
        rank=selected_rank,
        retained_energy=retained_energy,
        luminance_mean=decomposition.luminance_mean,
    )


def compute_fixed_rank_svd_tone(
    rgb_uint8: np.ndarray,
    rank: int,
) -> SVDToneResult:
    """Compute centered-SVD Tone at a requested rank for analysis workflows."""

    return reconstruct_svd_tone(compute_centered_svd(rgb_uint8), rank)


def compute_svd_tone(
    rgb_uint8: np.ndarray,
    energy_threshold: float = 0.90,
) -> SVDToneResult:
    """Extract broad tonal organization with an energy-selected SVD rank.

    Luminance is centered before decomposition so overall brightness does not
    dominate the singular-value energy. The smallest rank whose squared
    singular values retain ``energy_threshold`` of the centered energy is used.
    The tone map is ``normalize_01(abs(X_k))``, so broadly dark and bright
    variations are both important.
    """

    threshold = float(energy_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("energy_threshold must be greater than 0 and at most 1")

    decomposition = compute_centered_svd(rgb_uint8)
    cumulative = cumulative_singular_value_energy(decomposition.singular_values)
    rank = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return reconstruct_svd_tone(
        decomposition,
        min(rank, decomposition.singular_values.size),
    )


def compute_edge_map(rgb_uint8: np.ndarray) -> np.ndarray:
    """Measure local luminance change using horizontal and vertical Sobel filters."""

    luminance = compute_luminance(rgb_uint8)
    gradient_x = sobel(luminance, axis=1, mode="reflect")
    gradient_y = sobel(luminance, axis=0, mode="reflect")
    magnitude = np.hypot(gradient_x, gradient_y)
    return normalize_01(magnitude)


def compute_density_features(
    rgb_uint8: np.ndarray,
    tone_method: str = "svd",
    svd_energy_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected tone feature and Sobel structure feature."""

    if tone_method == "svd":
        tone_map = compute_svd_tone(rgb_uint8, svd_energy_threshold).tone_map
    elif tone_method == "luminance":
        tone_map = normalize_01(compute_luminance(rgb_uint8))
    else:
        raise ValueError("tone_method must be 'svd' or 'luminance'")
    edges = compute_edge_map(rgb_uint8)
    return tone_map, edges


def build_density_from_features(
    luminance_normalized: np.ndarray,
    edges: np.ndarray,
    mode: str = "mix",
    edge_weight: float = 0.85,
    tone_weight: float = 0.15,
) -> np.ndarray:
    """Combine a precomputed tone map and Sobel map into sampling density."""

    luminance_normalized = np.asarray(luminance_normalized, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if luminance_normalized.shape != edges.shape or luminance_normalized.ndim != 2:
        raise ValueError("luminance and edge maps must share one 2D shape")
    if not np.isfinite(luminance_normalized).all() or not np.isfinite(edges).all():
        raise ValueError("luminance and edge maps must contain finite values")
    if not np.isfinite(edge_weight) or not np.isfinite(tone_weight):
        raise ValueError("density weights must be finite")
    if edge_weight < 0 or tone_weight < 0:
        raise ValueError("density weights must be non-negative")

    if mode == "edge":
        density = edges
    elif mode == "bright":
        density = luminance_normalized
    elif mode == "dark":
        density = 1.0 - luminance_normalized
    elif mode == "mix":
        density = edge_weight * edges + tone_weight * luminance_normalized
    else:
        raise ValueError(f"Unknown density mode: {mode}")

    return np.clip(density, 1e-12, None).astype(np.float32)


def build_density_map(
    rgb_uint8: np.ndarray,
    mode: str = "mix",
    edge_weight: float = 0.85,
    tone_weight: float = 0.15,
    tone_method: str = "svd",
    svd_energy_threshold: float = 0.90,
) -> np.ndarray:
    """Build a nonnegative spatial density for adaptive seed sampling.

    Edges place more sites around image structure, while bright/dark modes keep
    their original luminance behavior. ``mix`` combines Sobel structure with
    either SVD-derived broad tone (default) or the legacy luminance feature.
    """

    if mode in {"bright", "dark", "edge"}:
        tone_map = normalize_01(compute_luminance(rgb_uint8))
        edges = compute_edge_map(rgb_uint8)
    else:
        tone_map, edges = compute_density_features(
            rgb_uint8,
            tone_method=tone_method,
            svd_energy_threshold=svd_energy_threshold,
        )
    return build_density_from_features(
        tone_map,
        edges,
        mode=mode,
        edge_weight=edge_weight,
        tone_weight=tone_weight,
    )
