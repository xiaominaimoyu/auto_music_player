"""CPU-only deterministic inference timing; does not exercise a live player."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from training.audit_sequence_expression import load_model
from training.playback_adoption import verify_inventory
from training.sequence_expression import SequenceSample, make_windows_from_config, predict_sample
from training.tempo_reference import _safe_path
from training.train_expression import _sha256, _write_json
from training.train_sequence_expression import write_integrity


def benchmark(*, audit: Path, checkpoint: Path, output: Path, cpu_threads: int = 4) -> dict:
    if output.exists() or cpu_threads <= 0:
        raise ValueError("new output and positive CPU threads required")
    verify_inventory(audit)
    verify_inventory(checkpoint.parent)
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    if report["checkpoint_sha256"] != _sha256(checkpoint):
        raise ValueError("CPU benchmark audit/model binding mismatch")
    torch.set_num_threads(cpu_threads)
    started = time.perf_counter()
    model, config = load_model(checkpoint, "cpu")
    load_seconds = time.perf_counter() - started
    samples = []
    policy = str(config.get("window_policy", "fixed"))
    bar_geometry_samples = 0
    for sample_id, relative in sorted(report["prediction_files"].items()):
        with np.load(_safe_path(audit, relative), allow_pickle=False) as stored:
            # Already score-only prepared feature channels. No target, mask,
            # performance scale, alignment confidence, or raw event inputs.
            note, chord, mapping = stored["note_features"], stored["chord_features"], stored["note_to_chord"]
            geometry: dict[str, np.ndarray] = {}
            if policy == "bar":
                fields = ("bar_span_rows", "bar_span_horizons", "bar_boundary_indices", "bar_requested_horizons")
                missing = [name for name in fields if name not in stored]
                if missing:
                    raise ValueError(f"bar-policy audit NPZ lacks score-only geometry: {', '.join(missing)}")
                geometry = {name: stored[name] for name in fields}
                bar_geometry_samples += 1
        samples.append(SequenceSample(sample_id, note, chord, np.zeros((len(note), 6), np.float32),
                                      np.zeros((len(chord), 2), np.float32), np.zeros((len(note), 6), bool),
                                      np.zeros((len(chord), 2), bool), mapping, None,
                                      geometry.get("bar_span_rows"), geometry.get("bar_span_horizons"),
                                      geometry.get("bar_boundary_indices"), geometry.get("bar_requested_horizons")))
    if not samples:
        raise ValueError("CPU benchmark requires score samples")
    if policy == "bar" and bar_geometry_samples != len(samples):
        raise ValueError("bar-policy benchmark lost window geometry for some samples")
    smallest = min(samples, key=lambda sample: len(sample.chord_features))
    # Zero-target inference must exercise the same layout the checkpoint trained on.
    planned = make_windows_from_config(smallest, config)
    if not planned:
        raise ValueError("window policy produced no cores for the zero-target benchmark")
    predict_sample(model, smallest, config=config, device="cpu")
    timings = []
    for sample in samples:
        started = time.perf_counter()
        first = predict_sample(model, sample, config=config, device="cpu")
        elapsed = time.perf_counter() - started
        second = predict_sample(model, sample, config=config, device="cpu")
        if any(not np.array_equal(a, b) for a, b in zip(first, second)):
            raise ValueError("CPU score inference was not deterministic")
        timings.append({"sample_id": sample.sample_id, "note_count": len(sample.note_features),
                        "chord_count": len(sample.chord_features), "seconds": elapsed})
    result = {"status": "complete", "checkpoint_sha256": _sha256(checkpoint), "architecture": config["architecture"],
              "window_policy": policy, "max_window_chords": config.get("max_window_chords"),
              "bar_geometry_sample_count": bar_geometry_samples,
              "planned_core_count_smallest_sample": len(planned),
              "fixed128_core_count_smallest_sample": -(-len(smallest.chord_features) // 128),
              "device": "cpu", "torch_version": torch.__version__, "cpu_threads": cpu_threads,
              "sample_count": len(samples), "model_load_seconds": load_seconds,
              "median_seconds": float(np.median([r["seconds"] for r in timings])),
              "p95_seconds": float(np.quantile([r["seconds"] for r in timings], .95)),
              "maximum_seconds": max(r["seconds"] for r in timings), "bitwise_repeat_passed": True,
              "zero_target_inputs": True, "actual_game_input_sent": False, "gui_responsiveness_verified": False,
              "fallback_interrupt_timeout_verified": False, "production_approved": False,
              "samples": timings, "implementation_sha256": _sha256(Path(__file__))}
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").touch()
    _write_json(output / "results.json", result)
    write_integrity(output)
    (output / "_INCOMPLETE").unlink()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    result = benchmark(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "p95_seconds", "bitwise_repeat_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
