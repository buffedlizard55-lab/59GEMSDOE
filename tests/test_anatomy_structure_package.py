import pytest

from gems59.anatomy import package_holdout_structure


def _relative_histogram():
    return {
        "edges": [0.0, 45.0, 90.0],
        "n_withheld": [7.0, 3.0],
        "n_visible_reference": [4.0, 6.0],
        "n_withheld_total": 10,
        "n_visible_total": 10,
        "median_withheld": 22.5,
        "median_visible": 67.5,
        "p25_withheld": 10.0,
        "p75_withheld": 35.0,
        "null_caveat": "descriptive only",
    }


def test_structure_packaging_preserves_histogram_and_distinct_positive_count():
    relative = _relative_histogram()
    result = package_holdout_structure(
        relative,
        withheld_positive_pixels=13,
        distance_positive_quantiles_px=[2.0, 4.0, 8.0, 12.0, 18.0],
        distance_domain_quantiles_px=[1.0, 10.0, 20.0, 30.0, 40.0],
    )
    assert result["withheld_positive_pixels"] == 13
    assert result["relative_strike"]["n_withheld"] == [7.0, 3.0]
    assert sum(result["relative_strike"]["n_withheld"]) == 10
    assert result["relative_strike"]["null_caveat"] == "descriptive only"
    assert result["relative_strike"]["n_withheld_total"] == 10


def test_structure_packaging_rejects_overwritten_histogram():
    relative = _relative_histogram()
    relative["n_withheld"] = 13  # the old scalar-overwrites-array bug
    with pytest.raises(ValueError, match="histogram does not match"):
        package_holdout_structure(
            relative,
            withheld_positive_pixels=13,
            distance_positive_quantiles_px=[2.0, 4.0, 8.0, 12.0, 18.0],
            distance_domain_quantiles_px=[1.0, 10.0, 20.0, 30.0, 40.0],
        )
