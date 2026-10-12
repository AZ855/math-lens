from streamlit.testing.v1 import AppTest


def test_manual_control_change_marks_preset_custom_and_updates_state(demo_app: AppTest) -> None:
    app = demo_app

    app.select_slider[0].set_value(0.20).run()

    assert app.selectbox[0].value == "Custom"
    assert [slider.value for slider in app.select_slider] == [0.20, 0.55, 0.75]
    assert not any("Current:" in caption.value for caption in app.caption)


def test_preset_updates_semantic_controls(demo_app: AppTest) -> None:
    app = demo_app

    app.selectbox[0].set_value("Abstract").run()

    assert [slider.value for slider in app.select_slider] == [0.85, 0.35, 0.65]
    assert app.radio[0].value == "Mean Region Color"
    assert not any("Current:" in caption.value for caption in app.caption)


def test_process_views_normalize_selection_and_readded_items(demo_app: AppTest) -> None:
    app = demo_app
    selector = app.multiselect[0]

    selector.set_value(["Voronoi Cells", "Original", "Seed Distribution"]).run()
    assert app.multiselect[0].value == [
        "Original",
        "Seed Distribution",
        "Voronoi Cells",
    ]

    app.multiselect[0].set_value(
        ["Original", "Seed Distribution", "Voronoi Cells", "Density Map"]
    ).run()
    assert app.multiselect[0].value == [
        "Original",
        "Density Map",
        "Seed Distribution",
        "Voronoi Cells",
    ]


def test_svd_feature_analysis_is_secondary_and_reports_energy(demo_app: AppTest) -> None:
    app = demo_app

    assert not app.exception
    assert "Feature Analysis" in [expander.label for expander in app.expander]
    assert any(
        "Tone method: **SVD**" in caption.value
        and "Selected rank:" in caption.value
        and "Retained energy:" in caption.value
        for caption in app.caption
    )
    assert any(
        "centered before SVD" in caption.value
        and "rather than overall brightness" in caption.value
        for caption in app.caption
    )


def test_structure_quick_buttons_update_slider_and_mark_custom(demo_app: AppTest) -> None:
    app = demo_app

    next(button for button in app.button if button.label == "Tone").click().run()
    assert app.select_slider[2].value == 0.0
    assert app.selectbox[0].value == "Custom"

    next(button for button in app.button if button.label == "Balanced").click().run()
    assert app.select_slider[2].value == 0.5

    next(button for button in app.button if button.label == "Structure").click().run()
    assert app.select_slider[2].value == 1.0


def test_density_map_explanation_describes_sampling_probability(demo_app: AppTest) -> None:
    app = demo_app

    assert any(
        "Brighter areas have higher sampling probability" in caption.value
        and "SVD Tone" in caption.value
        and "Sobel Structure" in caption.value
        for caption in app.caption
    )
