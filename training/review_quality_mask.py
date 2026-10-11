"""Review flagged PianoCoRe samples and emit a target-level quality mask.

The mask is a sidecar: it never edits the dataset shards or the refined ZIP.
All accepted samples remain available as context.  The legacy v1 policy can
hard-mask targets, while calibrated v2 treats aggregate flags as review
evidence and emits reversible soft weights over the native event masks.  The
conservative v3 policy narrows low-alignment weighting to timing targets and
does not collateral-weight velocity or pedal targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from training.midi_features import build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import RefinedArchive, _load_alignment


TARGETS = (
    "micro_offset_ms",
    "log_key_duration_ratio",
    "velocity_0_1",
    "velocity_robust_z",
    "pedal_at_onset",
    "pedal_coverage",
    "onset_delta_ms",
    "log_ioi_ratio",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None, "p95": None}
    q1, median, q3, p95 = np.quantile(array, [0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "p95": float(p95),
    }


def _raw_review(row: Mapping[str, Any], archive: RefinedArchive) -> dict[str, Any]:
    score = parse_midi_bytes(archive.read(str(row["score_midi_path"])))
    performance = parse_midi_bytes(archive.read(str(row["performance_midi_path"])))
    perf_idx, interpolated = _load_alignment(
        archive.read(str(row["alignment_path"]))
    )
    arrays = build_aligned_features(score, performance, perf_idx, interpolated)
    note_masks = arrays.note_masks
    chord_masks = arrays.chord_masks
    duration = arrays.note_targets[:, 1][note_masks[:, 1]]
    micro = arrays.note_targets[:, 0][note_masks[:, 0]]
    ioi = arrays.chord_targets[:, 1][chord_masks[:, 1]]
    onset = arrays.chord_targets[:, 0][chord_masks[:, 0]]
    note_end = max((note.end_s for note in performance.notes), default=0.0)
    pedal_events = performance.pedal_events
    last_pedal = max((event.time_s for event in pedal_events), default=0.0)
    pedal_tail = max(0.0, last_pedal - note_end)
    final_pedal_value = pedal_events[-1].value if pedal_events else None
    pedal_channels = sorted({int(event.channel) for event in pedal_events})
    real_alignment_count = int((~interpolated).sum())
    return {
        "score_duration_s": float(score.duration_s),
        "performance_duration_s": float(performance.duration_s),
        "score_last_note_end_s": float(max((note.end_s for note in score.notes), default=0.0)),
        "performance_last_note_end_s": float(note_end),
        "score_note_count": len(score.notes),
        "performance_note_count": len(performance.notes),
        "extra_performance_note_count": max(0, len(performance.notes) - len(score.notes)),
        "real_alignment_count": real_alignment_count,
        "interpolated_count": int(interpolated.sum()),
        "alignment_validated": True,
        "trusted_chord_fraction": float(chord_masks[:, 0].mean()) if len(chord_masks) else 0.0,
        "ioi_supervised_fraction": float(chord_masks[:, 1].mean()) if len(chord_masks) else 0.0,
        "pedal_event_count": len(pedal_events),
        "pedal_channels": pedal_channels,
        "final_pedal_value": final_pedal_value,
        "last_pedal_event_s": float(last_pedal),
        "pedal_tail_after_last_note_s": float(pedal_tail),
        "global_tempo_scale": float(arrays.diagnostics["global_tempo_scale"]),
        "refined_recall": float(row.get("refined_recall") or 0.0),
        "interpolation_rate": float(arrays.diagnostics["interpolation_rate"]),
        "target_stats": {
            "micro_offset_ms": _summary(micro.tolist()),
            "log_key_duration_ratio": _summary(duration.tolist()),
            "velocity_0_1": _summary(arrays.note_targets[:, 2][note_masks[:, 2]].tolist()),
            "velocity_robust_z": _summary(arrays.note_targets[:, 3][note_masks[:, 3]].tolist()),
            "pedal_at_onset": _summary(arrays.note_targets[:, 4][note_masks[:, 4]].tolist()),
            "pedal_coverage": _summary(arrays.note_targets[:, 5][note_masks[:, 5]].tolist()),
            "onset_delta_ms": _summary(onset.tolist()),
            "log_ioi_ratio": _summary(ioi.tolist()),
        },
        "diagnostics": {
            "onset_delta_abs_p95_ms": float(arrays.diagnostics["onset_delta_abs_p95_ms"]),
            "micro_offset_abs_p95_ms": float(arrays.diagnostics["micro_offset_abs_p95_ms"]),
            "supervised_note_count": int(arrays.diagnostics["supervised_note_count"]),
            "supervised_chord_count": int(arrays.diagnostics["supervised_chord_count"]),
            "onset_delta_aux_masked_count": int(
                arrays.diagnostics["onset_delta_aux_masked_count"]
            ),
        },
    }


def _make_mask(
    row: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    policy_version: str = "quality-mask-v1",
) -> dict[str, Any]:
    duration_median = review["target_stats"]["log_key_duration_ratio"]["median"]
    micro_p95 = float(review["diagnostics"]["micro_offset_abs_p95_ms"])
    onset_p95 = float(review["diagnostics"]["onset_delta_abs_p95_ms"])
    low_alignment = (
        float(review["refined_recall"]) < 0.90
        or float(review["interpolation_rate"]) > 0.10
    )
    duration_outlier = duration_median is not None and abs(float(duration_median)) > 1.0
    pedal_tail = float(review["pedal_tail_after_last_note_s"]) > 2.0
    micro_outlier = micro_p95 > 50.0
    onset_outlier = onset_p95 > 2_000.0

    weights = {
        "micro_offset_ms": 0.5 if low_alignment else 1.0,
        "log_key_duration_ratio": 0.5 if low_alignment else 1.0,
        "velocity_0_1": 0.5 if low_alignment else 1.0,
        "velocity_robust_z": 0.5 if low_alignment else 1.0,
        "pedal_at_onset": 0.5 if pedal_tail or low_alignment else 1.0,
        "pedal_coverage": 0.5 if pedal_tail or low_alignment else 1.0,
        "onset_delta_ms": 0.5 if onset_outlier or low_alignment else 1.0,
        "log_ioi_ratio": 0.5 if low_alignment else 1.0,
    }
    masks = {
        "micro_offset_ms": not micro_outlier,
        "log_key_duration_ratio": not duration_outlier,
        "velocity_0_1": True,
        "velocity_robust_z": True,
        "pedal_at_onset": not pedal_tail,
        "pedal_coverage": not pedal_tail,
        "onset_delta_ms": not onset_outlier,
        "log_ioi_ratio": True,
    }
    reasons: list[str] = []
    findings: list[str] = []
    if low_alignment:
        reasons.append("low_alignment_quality:recall<0.90_or_interpolation>0.10")
        findings.append("对齐召回率或插值率超过训练质量门槛。")
    if duration_outlier:
        reasons.append("duration_median_abs_log_ratio>1.0")
        findings.append("键时长相对谱面时值的中位 log ratio 超过 1。")
    if micro_outlier:
        reasons.append("micro_offset_abs_p95>50ms")
        findings.append("和弦内起音偏移 p95 超过 50 ms。")
    if onset_outlier:
        reasons.append("onset_delta_abs_p95>2000ms")
        findings.append(
            "相对全局 tempo 的和弦起音残差 p95 超过 2 s；原生逐和弦 mask 已处理超限值。"
        )
    if pedal_tail:
        reasons.append("pedal_tail_after_last_note>2s")
        findings.append(
            "CC64 事件在最后一个键音结束后持续超过 2 s；这可能是延迟松踏板，也可能是尾部事件噪声。"
        )
    if not findings:
        findings.append("复核指标未触发质量门槛；保留全部目标。")

    if policy_version == "quality-mask-v2-calibrated":
        # Aggregate flags are review evidence, not proof that an entire work
        # is mislabeled. Keep native per-event masks and use soft weights.
        masks = {target: True for target in TARGETS}
        weights = {target: 1.0 for target in TARGETS}
        if low_alignment:
            weights = {target: 0.5 for target in TARGETS}
        if duration_outlier:
            weights["log_key_duration_ratio"] = min(
                weights["log_key_duration_ratio"], 0.5
            )
        if micro_outlier:
            weights["micro_offset_ms"] = min(weights["micro_offset_ms"], 0.5)
        if onset_outlier:
            weights["onset_delta_ms"] = min(weights["onset_delta_ms"], 0.5)
        if pedal_tail:
            weights["pedal_at_onset"] = min(weights["pedal_at_onset"], 0.5)
            weights["pedal_coverage"] = min(weights["pedal_coverage"], 0.5)
        if reasons:
            findings.append(
                "校准策略：保留原生逐音符/逐和弦 mask，仅对触发证据的监督目标降权，不整曲硬屏蔽。"
            )
        status = "downweight" if reasons else "keep"
    elif policy_version == "quality-mask-v3-target-specific":
        # Conservative follow-up to v2: low alignment is evidence about the
        # timing projection, not about every performance target.  Keep the
        # native event masks and isolate the soft weight to timing-related
        # targets.  Velocity and pedal remain untouched unless their own
        # concrete diagnostics fire.
        masks = {target: True for target in TARGETS}
        weights = {target: 1.0 for target in TARGETS}
        if low_alignment:
            for target in (
                "micro_offset_ms",
                "log_key_duration_ratio",
                "onset_delta_ms",
                "log_ioi_ratio",
            ):
                weights[target] = 0.5
        if duration_outlier:
            weights["log_key_duration_ratio"] = min(
                weights["log_key_duration_ratio"], 0.5
            )
        if micro_outlier:
            weights["micro_offset_ms"] = min(weights["micro_offset_ms"], 0.5)
        if onset_outlier:
            weights["onset_delta_ms"] = min(weights["onset_delta_ms"], 0.5)
        if pedal_tail:
            weights["pedal_at_onset"] = min(weights["pedal_at_onset"], 0.5)
            weights["pedal_coverage"] = min(weights["pedal_coverage"], 0.5)
        if reasons:
            findings.append(
                "保守策略：低对齐仅影响时序目标；velocity/pedal 保持原权重，独立尾音证据仍可降权 pedal。"
            )
        status = "downweight" if reasons else "keep"
    elif policy_version == "quality-mask-v1":
        status = "keep" if not reasons else (
            "mask_targets" if any(not enabled for enabled in masks.values()) else "downweight"
        )
    else:
        raise ValueError(f"unsupported quality mask policy: {policy_version}")
    hard_disabled = [name for name, enabled in masks.items() if not enabled]
    return {
        "sample_id": row["id"],
        "source": row.get("performance_dataset"),
        "split": row.get("split"),
        "reviewed": True,
        "context_keep": True,
        "status": status,
        "reasons": reasons,
        "findings": findings,
        "target_mask": masks,
        "target_weight": weights,
        "hard_disabled_targets": hard_disabled,
        "policy_version": policy_version,
    }


def _default_mask(
    row: Mapping[str, Any], *, policy_version: str = "quality-mask-v1"
) -> dict[str, Any]:
    """Leave unreviewed accepted samples unchanged; do not extrapolate a sample audit."""

    masks = {target: True for target in TARGETS}
    weights = {target: 1.0 for target in TARGETS}
    return {
        "sample_id": row["id"],
        "source": row.get("performance_dataset"),
        "split": row.get("split"),
        "reviewed": False,
        "context_keep": True,
        "status": "keep",
        "reasons": [],
        "target_mask": masks,
        "target_weight": weights,
        "hard_disabled_targets": [],
        "policy_version": policy_version,
    }


def review(
    *,
    dataset_root: str | Path,
    archive: str | Path,
    source_audit: str | Path,
    output_dir: str | Path,
    policy_version: str = "quality-mask-v1",
) -> dict[str, Any]:
    if policy_version not in {
        "quality-mask-v1",
        "quality-mask-v2-calibrated",
        "quality-mask-v3-target-specific",
    }:
        raise ValueError(f"unsupported quality mask policy: {policy_version}")
    dataset_path = Path(dataset_root).resolve()
    archive_path = Path(archive).resolve()
    source_audit_path = Path(source_audit).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(dataset_path / "manifest.jsonl")
    flagged_report = json.loads(source_audit_path.read_text(encoding="utf-8"))
    flagged = {
        str(row["id"]): row
        for row in flagged_report["samples"]
        if row.get("flags")
    }
    manifest_by_id = {str(row["id"]): row for row in rows}
    missing = sorted(set(flagged) - set(manifest_by_id))
    if missing:
        raise ValueError(f"flagged samples are absent from manifest: {missing[:3]}")
    detailed: dict[str, dict[str, Any]] = {}
    with RefinedArchive(archive_path) as source:
        for sample_id in sorted(flagged):
            detailed[sample_id] = _raw_review(manifest_by_id[sample_id], source)

    mask_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["id"])
        review_data = detailed.get(sample_id)
        mask_rows.append(
            _make_mask(row, review_data, policy_version=policy_version)
            if review_data is not None
            else _default_mask(row, policy_version=policy_version)
        )
    mask_by_id = {str(row["sample_id"]): row for row in mask_rows}

    mask_path = output_path / "quality_mask.jsonl"
    with mask_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in mask_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    detailed_path = output_path / "flagged_samples_review.json"
    detailed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": policy_version,
                "sample_count": len(detailed),
                "samples": [
                    {
                        "sample_id": sample_id,
                        "performance_dataset": manifest_by_id[sample_id].get(
                            "performance_dataset"
                        ),
                        "performance_id": manifest_by_id[sample_id].get("performance_id"),
                        "split": manifest_by_id[sample_id].get("split"),
                        "composer": manifest_by_id[sample_id].get("composer"),
                        "composition": manifest_by_id[sample_id].get("composition"),
                        "movement": manifest_by_id[sample_id].get("movement"),
                        "score_midi_path": manifest_by_id[sample_id].get("score_midi_path"),
                        "performance_midi_path": manifest_by_id[sample_id].get(
                            "performance_midi_path"
                        ),
                        **detailed[sample_id],
                        "decision": mask_by_id[sample_id],
                    }
                    for sample_id in sorted(detailed)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    status_counts = Counter(row["status"] for row in mask_rows)
    target_mask_counts = {
        target: sum(not row["target_mask"][target] for row in mask_rows)
        for target in TARGETS
    }
    is_v3 = policy_version == "quality-mask-v3-target-specific"
    policy = {
        "low_alignment": (
            "refined_recall < 0.90 OR interpolation_rate > 0.10: keep context, weight timing targets (micro/duration/onset/IOI) 0.5; leave velocity/pedal unchanged"
            if is_v3
            else "refined_recall < 0.90 OR interpolation_rate > 0.10: keep context, weight supervised targets 0.5"
        ),
        "duration": "abs(log_key_duration_ratio median) > 1.0: v1 disables duration; calibrated v2/v3 downweight it",
        "micro_offset": "micro_offset_abs_p95 > 50 ms: v1 disables micro-offset; calibrated v2/v3 downweight it",
        "onset_aux": "onset_delta_abs_p95 > 2000 ms: native per-chord mask remains authoritative; calibrated v2/v3 downweight the sample",
        "pedal": "pedal tail after last note > 2 s: v1 disables pedal targets; calibrated v2/v3 downweight them pending local review",
        "velocity_and_ioi": (
            "never hard-disable solely from these flags; v3 low-alignment weighting excludes velocity and pedal, while IOI remains a timing target"
            if is_v3
            else "never hard-disable solely from these flags; low-alignment samples receive weight 0.5"
        ),
        "context": "all accepted samples remain context_keep=true; no raw row is deleted",
        "mode": (
            "candidate soft-weight only; retain native per-event masks"
            if policy_version in {"quality-mask-v2-calibrated", "quality-mask-v3-target-specific"}
            else "legacy hard-mask candidate"
        ),
    }
    summary = {
        "schema_version": 1,
        "policy_version": policy_version,
        "status": "complete",
        "review_scope": f"Only the {len(detailed)} samples flagged by the source audit were re-parsed; unreviewed accepted samples are unchanged.",
        "review_method": f"Read-only reparse of refined score/performance/alignment; builder alignment integrity checks passed for all {len(detailed)} samples; policy decisions are conservative and reversible.",
        "dataset_root": str(dataset_path),
        "archive": str(archive_path),
        "source_audit": str(source_audit_path),
        "manifest_count": len(rows),
        "reviewed_flagged_count": len(detailed),
        "status_counts": dict(sorted(status_counts.items())),
        "target_hard_disabled_counts": target_mask_counts,
        "policy": policy,
        "source_audit_sha256": _sha256(source_audit_path),
        "quality_mask_sha256": _sha256(mask_path),
        "flagged_review_sha256": _sha256(detailed_path),
        "outputs": {
            "quality_mask": str(mask_path),
            "flagged_review": str(detailed_path),
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "_SUCCESS").write_text(
        _sha256(output_path / "summary.json") + "\n", encoding="ascii"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy",
        choices=(
            "quality-mask-v1",
            "quality-mask-v2-calibrated",
            "quality-mask-v3-target-specific",
        ),
        default="quality-mask-v1",
    )
    args = parser.parse_args(argv)
    result = review(
        dataset_root=args.dataset_root,
        archive=args.archive,
        source_audit=args.source_audit,
        output_dir=args.output_dir,
        policy_version=args.policy,
    )
    print(json.dumps({"status": result["status"], "reviewed_flagged_count": result["reviewed_flagged_count"], "output_dir": args.output_dir}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["review", "main"]
