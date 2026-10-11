from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.analyze_target_error_decomposition import evaluation_label, split_prediction_error


def _pair(prediction: list[float], label: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(prediction, dtype=np.float64), np.asarray(label, dtype=np.float64)


def test_split_separates_the_constant_level_from_the_scatter():
    # recording 1: errors [2, 4] (median 3); recording 2: a single error of -2
    report = split_prediction_error([_pair([3.0, 7.0], [1.0, 3.0]), _pair([1.0], [3.0])])
    assert report["rows"] == 3 and report["performances"] == 2
    assert report["raw_mae"] == pytest.approx(8.0 / 3.0)
    assert report["offset_mae"] == pytest.approx((3 * 2 + 2 * 1) / 3)
    assert report["scatter_mae"] == pytest.approx((1 + 1 + 0) / 3)
    assert report["shape_mae"] == pytest.approx((1 + 1 + 0) / 3)
    assert report["label_scatter_mae"] == pytest.approx((1 + 1 + 0) / 3)
    assert report["scatter_share_of_raw"] == pytest.approx(0.25)


def test_a_purely_biased_prediction_has_no_scatter_and_no_shape_error():
    # every row is exactly 0.75 too high: the level is wrong, the profile is perfect
    report = split_prediction_error([_pair([1.75, 2.75, 3.75], [1.0, 2.0, 3.0])])
    assert report["raw_mae"] == pytest.approx(0.75)
    assert report["offset_mae"] == pytest.approx(0.75)
    assert report["scatter_mae"] == pytest.approx(0.0)
    assert report["shape_mae"] == pytest.approx(0.0)
    assert report["label_scatter_mae"] == pytest.approx(2.0 / 3.0)


def test_a_flat_prediction_loses_the_profile_but_not_the_level():
    report = split_prediction_error([_pair([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])])
    assert report["offset_mae"] == pytest.approx(0.0)
    assert report["scatter_mae"] == pytest.approx(2.0 / 3.0)
    # centring both sides makes the flat prediction's shape error equal the label's own spread
    assert report["shape_mae"] == pytest.approx(report["label_scatter_mae"])


def test_the_median_not_the_mean_defines_the_level():
    report = split_prediction_error([_pair([1.0, 1.0, 1.0, 1.0, 42.0], [0.0, 0.0, 0.0, 0.0, 1.0])])
    # errors [1, 1, 1, 1, 41]: median 1, mean 8.2
    assert report["offset_mae"] == pytest.approx(1.0)
    assert report["raw_mae"] == pytest.approx(45.0 / 5.0)
    assert report["scatter_mae"] == pytest.approx(40.0 / 5.0)


def test_an_empty_channel_reports_zero_rows_instead_of_crashing():
    assert split_prediction_error([]) == {"rows": 0, "performances": 0}


def test_a_raw_arm_evaluates_against_untouched_labels():
    label = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, True, False])
    assert np.array_equal(evaluation_label(label, mask, centred=False), label)


def test_a_centred_arm_evaluates_against_its_own_units():
    label = np.array([1.0, 5.0, 3.0])
    mask = np.array([True, False, True])
    centred = evaluation_label(label, mask, centred=True)
    # only the supervised rows are labelled, their median is 2.0, and the unsupervised row stays put
    assert centred[mask].tolist() == pytest.approx([-1.0, 1.0])
    assert centred[~mask] == pytest.approx(5.0)
    assert label.tolist() == [1.0, 5.0, 3.0]


def test_centring_a_label_with_nothing_supervised_does_not_produce_nan():
    label = np.array([1.0, 2.0])
    mask = np.zeros(2, dtype=bool)
    assert np.array_equal(evaluation_label(label, mask, centred=True), label)
