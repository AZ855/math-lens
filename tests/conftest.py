"""Public-checkout UI fixtures; no optional private photograph required."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def demo_app() -> AppTest:
    root = Path(__file__).parents[1]
    image = root / "assets/inputs/portfolio/architecture_geometric_facade_4x3.png"
    app = AppTest.from_file(root / "app.py", default_timeout=90)
    # Seed the existing upload/session path for each lens, not the app default.
    for prefix in ("voronoi", "tri"):
        app.session_state[f"{prefix}_image_data"] = image.read_bytes()
        app.session_state[f"{prefix}_image_name"] = image.name
    return app.run()
