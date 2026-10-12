from app import (
    PROCESS_VIEWS,
    abstraction_state,
    detail_state,
    ordered_process_views,
    semantic_option,
    structure_state,
)


def test_semantic_control_states_cover_normalized_ranges() -> None:
    assert abstraction_state(0.15) == "Detailed"
    assert abstraction_state(0.50) == "Balanced"
    assert abstraction_state(0.85) == "Abstract"

    assert detail_state(0.2) == "Global"
    assert detail_state(0.5) == "Balanced"
    assert detail_state(0.8) == "Detail-focused"

    assert structure_state(0.2) == "Tone-focused"
    assert structure_state(0.5) == "Balanced"
    assert structure_state(0.8) == "Structure-focused"


def test_process_views_always_return_in_canonical_order() -> None:
    selected = ["Seed Distribution", "Original", "Voronoi Cells"]
    assert ordered_process_views(selected) == [
        "Original",
        "Seed Distribution",
        "Voronoi Cells",
    ]

    selected.append("Density Map")
    assert ordered_process_views(selected) == list(PROCESS_VIEWS)


def test_semantic_options_are_meaningful_and_unique() -> None:
    labels = [
        semantic_option(index / 100, abstraction_state, "Detailed", "Abstract")
        for index in range(101)
    ]

    assert labels[0] == "Detailed"
    assert labels[15] == "Detailed · 15%"
    assert labels[50] == "Balanced · 50%"
    assert labels[85] == "Abstract · 85%"
    assert labels[100] == "Abstract"
    assert len(labels) == len(set(labels))
