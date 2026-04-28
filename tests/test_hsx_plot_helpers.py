import pytest

from gliquid.hsx import HSX


def _make_hsx() -> HSX:
    data_dict = {
        "phases": ["Mn", "Zr", "L", "alpha Mn (bcc)", "beta Mn (bcc)", "ZrMn2"],
        "comps": ["Mn", "Zr"],
        "data": [
            [0.0, 0.0, 0.0, "Mn"],
            [1.0, 0.0, 0.0, "Zr"],
            [0.0, 1.0, 2.0, "L"],
            [1.0, 1.0, 2.1, "L"],
            [0.0, 0.5, 1.0, "alpha Mn (bcc)"],
            [0.0, 0.7, 1.2, "beta Mn (bcc)"],
            [0.4, 0.6, 1.1, "ZrMn2"],
        ],
    }
    return HSX(data_dict=data_dict, conds=[0.0, 1500.0])


def test_abbreviate_phase_name_rules():
    hsx = _make_hsx()
    names = ["alpha Mn (bcc)", "beta Mn (bcc)", "delta Fe (bcc)", "ZrMn2", "Liquid"]

    assert hsx._abbreviate_phase_name("alpha Mn (bcc)", names) == "(αMn)"
    assert hsx._abbreviate_phase_name("delta Fe (bcc)", names) == "(Fe)"
    assert hsx._abbreviate_phase_name("ZrMn2", names) == "ZrMn<sub>2</sub>"
    assert hsx._abbreviate_phase_name("Liquid", names) == "L"


def test_merge_close_values_groups_with_mean():
    hsx = _make_hsx()
    merged = hsx._merge_close_values([10.0, 10.2, 12.0, 12.1, 20.0], tol=0.3)

    assert len(merged) == 3
    assert merged[0] == pytest.approx(10.1, abs=1e-6)
    assert merged[1] == pytest.approx(12.05, abs=1e-6)
    assert merged[2] == pytest.approx(20.0, abs=1e-6)


def test_detect_tie_lines_uses_crossings_not_full_xlim():
    hsx = _make_hsx()
    boundary_curves = [
        [(0.0, 1000.0), (50.0, 800.0), (100.0, 900.0)],
        [(20.0, 600.0), (20.0, 900.0)],
        [(80.0, 700.0), (80.0, 950.0)],
    ]

    tie_lines = hsx._detect_tie_lines(
        invariant_temps=[850.0],
        boundary_curves=boundary_curves,
        plot_xlim=(0.0, 100.0),
        temp_tol=0.1,
        x_tol=0.25,
    )

    assert len(tie_lines) == 1
    tl = tie_lines[0]
    assert tl["x_start"] == pytest.approx(20.0, abs=1e-4)
    assert tl["x_end"] == pytest.approx(80.0, abs=1e-4)


def test_resolve_label_collisions_separates_overlaps():
    hsx = _make_hsx()
    labels = [
        {
            "x": 10.0,
            "y": 400.0,
            "text": "(Mn)",
            "xanchor": "center",
            "yanchor": "middle",
            "textangle": -90,
            "font_size": 12,
            "font_color": "black",
        },
        {
            "x": 10.0,
            "y": 400.0,
            "text": "ZrMn<sub>2</sub>",
            "xanchor": "center",
            "yanchor": "middle",
            "textangle": -90,
            "font_size": 12,
            "font_color": "black",
        },
    ]

    resolved = hsx._resolve_label_collisions(labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0), max_iterations=20)

    assert resolved[0]["y"] != pytest.approx(resolved[1]["y"], abs=1e-6)
