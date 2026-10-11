from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.rhythm_objective import cumulative_ioi_loss, validate_spans
from training.sequence_expression import collate_windows, fit_input_normalizers, make_model, make_windows, predict_sample, prepare_sample
from training.train_expression import TrainingRunError
from training.train_sequence_expression import _run_epoch, train


def _arrays(chords=10):
    mapping = np.arange(chords)
    note = np.zeros((chords, 15), dtype=np.float32)
    chord = np.zeros((chords, 11), dtype=np.float32)
    note[:, 3] = mapping
    chord[:, 0] = mapping
    chord[1:, 8] = np.log1p(1.0)
    targets = np.zeros((chords, 2), dtype=np.float32)
    targets[1:, 1] = np.log(1.1)
    masks = np.ones((chords, 2), dtype=bool)
    masks[0, 1] = False
    return {
        "note_features": note, "chord_features": chord, "note_to_chord": mapping,
        "note_targets": np.zeros((chords, 6), dtype=np.float32), "note_masks": np.ones((chords, 6), dtype=bool),
        "chord_targets": targets, "chord_masks": masks,
    }


def _reference(chords=10):
    return {
        "score_seconds_per_beat_reference": 0.5,
        "chord_tempo_log_ratio": np.zeros(chords),
        "note_duration_tempo_log_ratio": np.zeros(chords),
        "chord_score_seconds": np.arange(chords) * 0.5,
        # These maliciously varying performance fields must not be consumed.
        "global_tempo_scale": 1234.0, "chord_performance_seconds": np.arange(chords) * 1000.0,
    }


def _config(sample, cumulative_weight=0.0):
    return {
        "architecture": "mlp", "hidden_dim": 8, "dropout": 0.0,
        "core_size": 5, "context_size": 2, "batch_size": 2, "target_shuffle": False,
        "note_target_means": [0.0] * 6, "note_target_scales": [1.0] * 6,
        "chord_target_means": [0.0] * 2, "chord_target_scales": [1.0] * 2,
        "onset_weight": 0.0, "micro_weight": 0.0, "duration_weight": 1.0,
        "velocity_weight": 0.0, "pedal_weight": 0.0, "ioi_weight": 1.0,
        "cumulative_weight": cumulative_weight, "cumulative_spans": [1, 3, 8],
        **fit_input_normalizers([sample]),
    }


def test_exact_targets_have_zero_cumulative_loss_and_identity_has_gradient():
    target = torch.tensor([[0.0, math.log(1.1), math.log(0.8), math.log(1.4)]], dtype=torch.float32)
    score = torch.tensor([[0.0, 0.5, 0.25, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[False, True, True, True]])
    exact = target.clone().requires_grad_()
    loss, metrics = cumulative_ioi_loss(exact, target, score, mask, spans=(1, 3))
    assert loss.item() == pytest.approx(0.0, abs=1e-15)
    assert metrics["span_counts"] == {"1": 3, "3": 1}
    identity = torch.zeros_like(target, requires_grad=True)
    different_loss, _ = cumulative_ioi_loss(identity, target, score, mask, spans=(1, 3))
    assert different_loss.item() > 0
    different_loss.backward()
    assert torch.isfinite(identity.grad).all()
    assert torch.any(identity.grad[mask] != 0)
    assert identity.grad[0, 0] == 0


def test_relative_elapsed_weights_seconds_not_beats_and_counts_spans_equally():
    prediction = torch.log(torch.tensor([[2.0, 1.0]]))
    target = torch.zeros_like(prediction)
    seconds = torch.tensor([[0.25, 0.75]])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss, result = cumulative_ioi_loss(prediction, target, seconds, mask, spans=(1, 2))
    # Span 1 has errors [1,0] and mean SmoothL1 .25; span 2 has
    # relative elapsed error .25 and SmoothL1 .03125; equal-span mean.
    assert loss.item() == pytest.approx((0.25 + 0.03125) / 2, rel=1e-6)
    assert result["span_relative_mae"]["2"] == pytest.approx(0.25, rel=1e-6)


def test_invalid_gap_and_batch_boundary_cannot_form_long_span():
    prediction = torch.zeros((2, 4), requires_grad=True)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    loss, result = cumulative_ioi_loss(prediction, torch.zeros_like(prediction), torch.ones_like(prediction),
                                       mask, spans=(1, 2, 3, 8))
    assert result["span_counts"] == {"1": 6, "2": 2, "3": 0, "8": 0}
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_padding_nan_is_ignored_but_trusted_nonfinite_or_nonpositive_fails():
    prediction = torch.tensor([[0.0, float("nan")]], requires_grad=True)
    target = torch.tensor([[0.0, float("inf")]])
    score = torch.tensor([[0.5, float("nan")]])
    mask = torch.tensor([[True, False]])
    loss, _ = cumulative_ioi_loss(prediction, target, score, mask, spans=(1,))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(TrainingRunError, match="non-finite trusted"):
        cumulative_ioi_loss(prediction, target, score, torch.ones_like(mask), spans=(1,))
    with pytest.raises(TrainingRunError, match="positive seconds"):
        cumulative_ioi_loss(torch.zeros((1, 1)), torch.zeros((1, 1)), torch.zeros((1, 1)),
                            torch.ones((1, 1), dtype=torch.bool))


def test_prediction_clamp_is_explicit_target_is_never_clamped():
    prediction = torch.tensor([[100.0]], requires_grad=True)
    target = torch.tensor([[math.log(2.0)]])
    score = torch.ones((1, 1))
    mask = torch.ones((1, 1), dtype=torch.bool)
    loss, result = cumulative_ioi_loss(prediction, target, score, mask, spans=(1,), log_ratio_clip=1.0)
    expected_error = (math.e - 2) / 2
    assert loss.item() == pytest.approx(expected_error ** 2 / 2, rel=1e-6)
    assert result["prediction_clipped_count"] == 1
    loss.backward()
    assert prediction.grad.item() == 0  # saturation, not a secretly altered label
    with pytest.raises(TrainingRunError, match="overflow"):
        cumulative_ioi_loss(prediction, torch.tensor([[1000.0]]), score, mask)


def test_empty_coverage_is_differentiable_zero_not_fake_accuracy():
    prediction = torch.zeros((2, 3), requires_grad=True)
    loss, result = cumulative_ioi_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction),
                                       torch.zeros_like(prediction, dtype=torch.bool))
    assert loss.item() == 0
    assert result["valid_window_count"] == 0
    assert all(value is None for value in result["span_relative_mae"].values())
    loss.backward()
    assert torch.all(prediction.grad == 0)


@pytest.mark.parametrize("spans", [(), (0,), (-1, 2), (1, 1), (1.5,), (True,)])
def test_invalid_spans_rejected(spans):
    with pytest.raises(TrainingRunError, match="distinct positive"):
        validate_spans(spans)


def test_core_endpoints_exclude_halo_padding_previous_core_and_native_gap():
    arrays = _arrays(10)
    arrays["chord_masks"][3, 1] = False
    sample = prepare_sample("work-a", arrays, timing_reference=_reference(10))
    short = prepare_sample("work-b", _arrays(2), timing_reference=_reference(2))
    windows = list(make_windows(sample, 5, 2)) + list(make_windows(short, 5, 2))
    batch = collate_windows(windows, _config(sample, 0.5))
    assert batch["cumulative_ioi_mask"].tolist() == [
        [False, True, True, False, True, False, False],
        [False, False, False, True, True, True, True],
        [False, True, False, False, False, False, False],
    ]
    assert batch["score_ioi_seconds"].dtype == torch.float64
    # Pure score conditions are unchanged by performance-wide calibration.
    changed = dict(_reference(10), global_tempo_scale=0.01, chord_performance_seconds=np.arange(10) * -3)
    other = prepare_sample("work-a", arrays, timing_reference=changed)
    assert np.array_equal(sample.score_ioi_seconds, other.score_ioi_seconds)
    assert np.array_equal(sample.chord_features, other.chord_features)


def test_physical_seconds_require_reference_or_explicit_geometry_and_old_samples_still_work():
    old = prepare_sample("old", _arrays())
    assert old.score_ioi_seconds is None
    assert "score_ioi_seconds" not in collate_windows(list(make_windows(old)), _config(old))
    with pytest.raises(TrainingRunError, match="requires pure score IOI"):
        collate_windows(list(make_windows(old)), _config(old, 0.5))
    arrays = _arrays()
    arrays["score_ioi_seconds"] = np.r_[0.0, np.full(9, 0.5)]
    explicit = prepare_sample("explicit", arrays)
    assert explicit.chord_features.shape[1] == 11
    assert np.all(explicit.score_ioi_seconds[1:] == 0.5)
    fallback_ref = _reference()
    fallback_ref.pop("chord_score_seconds")
    derived = prepare_sample("derived", _arrays(), timing_reference=fallback_ref)
    assert derived.score_ioi_seconds[1:] == pytest.approx(np.full(9, 0.5), rel=1e-6)
    # A training-only objective flag must not demand labels/physical sidecars
    # at inference; model channels have their own score-only contract.
    config = _config(old, 0.5)
    prediction = predict_sample(make_model(config), old, config=config)
    assert prediction[0].shape == (10, 6)
    assert prediction[1].shape == (10, 2)


def test_default_zero_weight_matches_explicit_zero_and_cumulative_only_trains():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        sample = prepare_sample("work", _arrays(), timing_reference=_reference())
        torch.manual_seed(87)
        config = _config(sample)
        model = make_model(config)
        explicit = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=1)
        old_config = dict(config)
        old_config.pop("cumulative_weight")
        default = _run_epoch(model, [sample], old_config, device=torch.device("cpu"), optimizer=None, seed=1)
        assert explicit == default
        config.update(cumulative_weight=0.5, duration_weight=0.0, ioi_weight=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        result = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=optimizer, seed=1)
        assert result["weighted_target_count"] == 0
        assert result["point_loss"] is None
        assert result["cumulative_span_counts"] == {"1": 8, "3": 4, "8": 0}
        assert result["cumulative_loss"] > 0
        assert math.isfinite(result["loss"])
        config["target_shuffle"] = True
        with pytest.raises(TrainingRunError, match="target shuffle"):
            _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=1)
    finally:
        torch.set_num_threads(previous_threads)


def test_train_rejects_no_losses_negative_nan_and_cumulative_without_reference(tmp_path):
    args = dict(dataset=tmp_path / "dataset", output=tmp_path / "run", architecture="gru",
                train_sample_id_file=tmp_path / "train.txt", validation_sample_id_file=tmp_path / "val.txt")
    for overrides in (
        dict(micro_weight=0.0, onset_weight=0.0, duration_weight=0.0, velocity_weight=0.0,
             pedal_weight=0.0, ioi_weight=0.0),
        dict(duration_weight=-1.0), dict(cumulative_weight=float("nan")),
    ):
        with pytest.raises(TrainingRunError, match="finite, nonnegative"):
            train(**args, **overrides)
    with pytest.raises(TrainingRunError, match="explicit pure-score timing"):
        train(**args, cumulative_weight=0.5)
    with pytest.raises(TrainingRunError, match="target shuffle"):
        train(**args, cumulative_weight=0.5, timing_reference=tmp_path / "reference", target_shuffle=True)
