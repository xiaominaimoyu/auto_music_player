from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.analyze_shape_explanatory_power import (
    CANDIDATE_NAMES,
    _shape_block,
    demean_within_groups,
    grouped_folds,
    ridge_out_of_fold_r2,
)
from training.shape_features import SHAPE_FEATURE_NAMES


def test_the_tool_measures_exactly_the_shipped_block():
    assert CANDIDATE_NAMES == SHAPE_FEATURE_NAMES


def test_grouped_folds_hold_out_whole_groups():
    groups = np.repeat(np.arange(10), 5)
    splits = grouped_folds(groups, folds=5, seed=7)
    assert len(splits) == 5
    for train, test in splits:
        assert not set(groups[train]).intersection(set(groups[test]))
        assert train.sum() + test.sum() == len(groups)
    covered = np.concatenate([np.flatnonzero(test) for _train, test in splits])
    assert covered.size == len(groups)


def test_grouped_folds_refuse_fewer_groups_than_folds():
    with pytest.raises(ValueError, match="at least"):
        grouped_folds(np.array([0, 0, 1, 1]), folds=5, seed=1)


def test_ridge_recovers_a_known_linear_relation_out_of_fold():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(20), 25)
    signal = rng.normal(size=len(groups))
    matrix = np.column_stack((signal * 2.0 + rng.normal(scale=0.05, size=len(groups)),
                              rng.normal(size=len(groups))))
    target = 3.0 * signal + 0.1 * matrix[:, 1]
    report = ridge_out_of_fold_r2(matrix, target, groups, folds=5, seed=11)
    assert report["r2"] is not None and report["r2"] > 0.9
    assert report["residual_mae"] < 0.2 * float(np.abs(target - target.mean()).mean())


def test_ridge_reports_a_useless_design_as_no_explanatory_power():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(12), 20)
    matrix = rng.normal(size=(len(groups), 3))
    target = rng.normal(size=len(groups))
    assert ridge_out_of_fold_r2(matrix, target, groups, folds=4, seed=2)["r2"] < 0.1


def test_ridge_refuses_mismatched_dimensions():
    with pytest.raises(ValueError, match="leading dimension"):
        ridge_out_of_fold_r2(np.zeros((10, 2)), np.zeros(11), np.repeat(np.arange(5), 2))


def test_a_constant_target_reports_no_r2_instead_of_a_number():
    # a piece-mean profile with no variance at all has nothing to explain: R^2 is undefined,
    # and inventing 0.0 would make a degenerate column look like a failed hypothesis rather
    # than a missing measurement
    target = np.zeros(8)
    report = ridge_out_of_fold_r2(np.arange(8, dtype=float).reshape(-1, 1), target,
                                  np.arange(8), folds=4, seed=1)
    assert report["r2"] is None
    assert report["residual_mae"] == pytest.approx(0.0)


def test_demean_within_groups_removes_group_levels_only():
    groups = np.array([0, 0, 0, 1, 1, 1])
    values = np.array([1.0, 2.0, 3.0, 101.0, 102.0, 103.0])
    centred = demean_within_groups(values.reshape(-1, 1), groups).ravel()
    assert centred.tolist() == pytest.approx([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])


class _Sample:
    def __init__(self, features: np.ndarray, positions: np.ndarray) -> None:
        self.note_features = features
        self.chord_features = positions
        self.note_to_chord = np.arange(len(positions), dtype=np.int32)


def test_shape_block_uses_the_sample_chord_positions_and_mapping():
    import math

    features = np.zeros((2, 15), dtype=np.float32)
    features[:, 0] = [0.0, 0.0833333]           # keys 60 and 62
    features[:, 3] = [0.0, 100.0]
    features[:, 4] = [math.log1p(2.0)] * 2
    block = _shape_block(_Sample(features, np.array([[0.0], [100.0]], dtype=np.float32)))
    assert block.shape == (2, len(SHAPE_FEATURE_NAMES))
    assert np.isfinite(block).all()
