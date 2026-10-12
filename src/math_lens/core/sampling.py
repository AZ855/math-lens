"""Shared uniform and image-adaptive site sampling for mathematical lenses."""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_uniform_points(
    width: int,
    height: int,
    n_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate reproducible, uniformly distributed ``(x, y)`` sites."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, width, size=n_points)
    ys = rng.uniform(0.0, height, size=n_points)
    return np.column_stack([xs, ys]).astype(np.float32)


def sample_points_from_pdf(
    pdf2d: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw continuous image coordinates from a 2D probability density."""

    density = np.asarray(pdf2d)
    if density.ndim != 2 or density.size == 0:
        raise ValueError("pdf2d must be a nonempty 2D array")
    if not np.isfinite(density).all():
        raise ValueError("pdf2d must contain only finite values")
    if n < 0:
        raise ValueError("n must be non-negative")

    height, width = density.shape
    # float64 normalization keeps the probabilities numerically well behaved
    # for NumPy's discrete sampler, even for large images.
    probabilities = np.clip(density.astype(np.float64), 1e-12, None)
    probabilities /= probabilities.sum(dtype=np.float64)
    indices = rng.choice(
        probabilities.size,
        size=n,
        replace=True,
        p=probabilities.ravel(),
    )
    ys, xs = np.divmod(indices, width)

    # A density value belongs to a pixel-sized region, so uniform subpixel
    # offsets turn the discrete choice back into continuous image coordinates.
    xs = xs.astype(np.float32) + rng.random(n).astype(np.float32)
    ys = ys.astype(np.float32) + rng.random(n).astype(np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


def generate_adaptive_points(
    density: np.ndarray,
    n_coarse: int,
    n_fine: int,
    seed: int = 0,
    blur_sigma: float = 6.0,
    density_power: float = 2.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """Generate broad-coverage and detail-focused sites from a density map."""

    density_array = np.asarray(density)
    if density_array.ndim != 2 or density_array.size == 0:
        raise ValueError("density must be a nonempty 2D array")
    if not np.isfinite(density_array).all():
        raise ValueError("density must contain only finite values")
    if n_coarse < 0 or n_fine < 0:
        raise ValueError("n_coarse and n_fine must be non-negative")
    if n_coarse + n_fine < 2:
        raise ValueError("Total number of points must be at least 2")
    if not np.isfinite(blur_sigma) or not np.isfinite(density_power):
        raise ValueError("blur_sigma and density_power must be finite")
    if not np.isfinite(jitter):
        raise ValueError("jitter must be finite")

    rng = np.random.default_rng(seed)
    normalized = density_array.astype(np.float32)
    normalized = normalized / (normalized.max() + 1e-12)

    if blur_sigma > 0:
        coarse_density = gaussian_filter(
            normalized,
            sigma=max(0.0, float(blur_sigma)),
        )
    else:
        coarse_density = normalized
    coarse_density = np.clip(coarse_density, 1e-12, None)

    # Raising the density sharpens its peaks, concentrating fine sites around
    # the strongest image features while coarse sites retain broad coverage.
    fine_density = np.clip(normalized, 1e-12, None) ** max(
        0.1, float(density_power)
    )
    coarse_points = sample_points_from_pdf(coarse_density, n_coarse, rng)
    fine_points = sample_points_from_pdf(fine_density, n_fine, rng)
    points = np.concatenate([coarse_points, fine_points], axis=0)

    if jitter > 0:
        points += rng.normal(0.0, jitter, size=points.shape).astype(np.float32)

    height, width = normalized.shape
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1e-3))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1e-3))
    return points.astype(np.float32)
