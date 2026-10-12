# Attribution and publication review

## Learning sources

The project acknowledges these prior learning sources:

- [Yue Sun and collaborators' generative-art course](https://github.com/yue-sun/generative-art),
  especially [the Voronoi/Delaunay notebook](https://github.com/yue-sun/generative-art/blob/main/02_tuesday/01_voronoi_delaunay_art.ipynb).
- [Math + Code + Art tutorials](https://github.com/math-code-art/tutorials),
  especially [Voronoi Photo Mosaic](https://github.com/math-code-art/tutorials/tree/main/voronoi-photo-mosaic).

### Source-informed implementation and permission review

The original development briefs explicitly requested adaptation of supplied
Voronoi source and supplied source-inspired sampling snippets. These portions
must not be described as wholly original implementations. The included helpers
have structural correspondence with
[cellular_impressions.py](https://github.com/math-code-art/tutorials/blob/main/style-transfer-voronoi-mosaic/cellular_impressions.py)
in Math + Code + Art, blob `244c0b48516ae5d88c621dae52178b71d17a6114`:

| Included code | Corresponding upstream implementation |
| --- | --- |
| `core/sampling.py::sample_points_from_pdf` | `sample_points_from_pdf`, lines 239–253: weighted pixel selection, row/column recovery, subpixel offsets, float32 coordinate assembly |
| `core/sampling.py::generate_adaptive_points` | `sample_points_adaptive`, lines 256–278, plus `jitter_and_clip`, lines 281–287: blurred coarse density, powered fine density, concatenation, optional jitter and bounds clipping |
| `lenses/voronoi/renderer.py::voronoi_label_map` | `voronoi_label_map`, lines 290–303: flattened pixel grid and chunked KD-tree nearest-site queries |
| `lenses/voronoi/renderer.py::render_voronoi_mean_color` | `render_voronoi_mean_color`, lines 306–342: indexed counts/RGB sums, regional means and label-based output assignment |

Probability normalization, weighted `Generator.choice`, index division/remainder,
uniform offsets, nearest-neighbor queries and arithmetic means are standard
mathematical/NumPy/SciPy operations. Their combination with the same helper roles,
operation order, dtype choices and small stabilizing constants is evidence of
source-informed adaptation, especially given the development briefs. The local
code also adds validation, float64 probability normalization and reusable label
maps, removes mask handling, and integrates the helpers into this application.
Those changes do not by themselves establish independent provenance.

The exact upstream origin/revision of the originally supplied snippets and the
rights holder's authorization still require author confirmation. The comparison
establishes correspondence, not the direction of copying between repositories
or legal copyrightability of an individual short sequence. This review does
not clear the included source-derived portions for redistribution.

These credits identify learning context, not an assertion that all project
code is original, nor a grant of permission to redistribute upstream code.
Local source has no recorded per-file adaptation notices. The development
briefs establish source-informed implementation, but neither they nor the
unborn local Git history establish the complete chain of authorship or reuse
rights. Before publication, confirm who supplied/authored the snippets, their
upstream revision and any applicable license or written permission. Preserve
any required notices and confirm that authorization covers public distribution
of the adapted helpers in both lenses. Attribution alone is not authorization.

The upstream repository trees reviewed on 2026-10-11 did not contain files
named LICENSE, COPYING, or NOTICE; the inspected tutorial file also contained
no copyright/license/permission notice. Absence of a notice does not mean the
code is public domain. Public visibility and an acknowledgment
are not substitutes for an explicit reuse license. No blanket license has been
chosen for Math Lens; no MIT license is implied. Code licensing and any required
upstream permissions remain publication-review items.

## Demonstration images

The project owner confirms that `architecture_geometric_facade_4x3.png` and
`colored_glass_still_life_4x3.png` in `assets/inputs/portfolio/` were **AI-generated
as demonstration inputs for Math Lens**. Both are 1448×1086 RGB images; the
original files are retained without recompression. The included overview figure
derives from the AI-generated architecture input and genuine app captures;
the mathematical result panels were computed by the application.

This records the owner's provenance statement, not a blanket copyright or asset
license. The generation service/model and its applicable distribution terms are
not recorded here; the owner should confirm those terms and any third-party
reference-image rights before publication. No ownership guarantee is inferred
solely from AI generation.

The optional local `assets/inputs/example.JPG` is **not** in the proposed public
file set. Permission is unconfirmed, and the image contains camera metadata.
It remains unchanged locally. README examples and UI tests use the public PNGs
instead; the application still supports user uploads.

## Dependencies

Math Lens imports NumPy, SciPy, Pillow, Matplotlib, and Streamlit, with their
transitive dependencies. It does not vendor these libraries in the public file
set. Their own licenses govern their distribution. Local virtual environments,
media encoders, browser runtimes, and generated build artifacts are excluded.

See [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
for the distinction between publishing a repository and granting reuse rights.

### Acknowledgments

Math Lens is an independently developed creative computing project. Some initial ideas, mathematical concepts, and foundational implementation logic were inspired by or adapted from the [Math+Code+Art initiative](https://math-code-art.github.io/).

The project extends these foundations through interactive mathematical lenses, adaptive visual controls, computational process visualizations, and custom interaction design.