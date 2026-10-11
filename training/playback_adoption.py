"""Conservative, development-only playback adoption gate (no game I/O)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.compare_quality_multiseed import _bootstrap, _sha256


POLICY = {
    "version": 1,
    "scope": "offline compiler shadow; not legacy Player or game acceptance",
    "minimum_seed_pairs": 3,
    "minimum_work_units": 5,
    "minimum_source_work_units": 5,
    "primary_improvement_fraction": 0.10,
    "source_noninferiority_fraction": 0.02,
    "required_baselines": ["identity", "rules", "zero_model_adapter"],
    "primary_metrics": ["projected:fixed_score:ioi_mae_ms",
                        "projected:fixed_score:key_duration_mae_ms"],
    "noninferiority_metrics": [f"projected:fixed_score:bar{h}_mae_ms" for h in (1, 4, 16)],
    "micro_enabled": False,
    "velocity_enabled": False,
    "pedal_enabled": False,
    "human_preference_required_for_default_enable": True,
    "threshold_origin": "project engineering target, not an industry standard",
}


def verify_inventory(root: str | Path) -> str:
    """Verify every recorded child artifact before using its observations."""
    root = Path(root).resolve()
    inventory = root / "integrity.json"
    if (root / "_INCOMPLETE").exists() or not (root / "_SUCCESS").is_file():
        raise ValueError(f"incomplete artifact: {root}")
    digest = _sha256(inventory)
    if (root / "_SUCCESS").read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"inventory fingerprint mismatch: {root}")
    rows = json.loads(inventory.read_text(encoding="utf-8"))["files"]
    if not rows:
        raise ValueError("artifact inventory is empty")
    seen = set()
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or path == root or row["path"] in seen:
            raise ValueError("unsafe or duplicate artifact path")
        seen.add(row["path"])
        if path.stat().st_size != row["size_bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError(f"artifact modified: {path}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
              and path.name not in {"_SUCCESS", "_INCOMPLETE", "integrity.json"}}
    if seen != actual:
        raise ValueError("artifact inventory coverage mismatch")
    return digest


def compare_work_metrics(baselines: list[dict], candidates: list[dict], *, seed: int = 20261003) -> dict:
    """Pair identical seeds/selections/works; never silently intersect coverage."""
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("need nonempty matched seed pairs")
    if len({row["seed"] for row in baselines}) != len(baselines):
        raise ValueError("seed pairs must use distinct seeds")
    result: dict[str, Any] = {}
    names = set(baselines[0]["work_values"])
    for base, candidate in zip(baselines, candidates):
        if base["selection_sha256"] != candidate["selection_sha256"] or base["seed"] != candidate["seed"]:
            raise ValueError("selection or seed pairing mismatch")
        if set(base["work_values"]) != names or set(candidate["work_values"]) != names:
            raise ValueError("metric sets mismatch")
        if base["source_by_work"] != candidate["source_by_work"]:
            raise ValueError("source/work identity mismatch")
    for name in sorted(names):
        keys = set(baselines[0]["work_values"][name])
        if any(set(row["work_values"][name]) != keys for row in baselines + candidates):
            raise ValueError(f"work coverage mismatch: {name}")
        works = sorted(keys)
        if not works:
            result[name] = {"n_works": 0, "baseline_mean": None, "candidate_mean": None,
                            "relative_improvement": None, "delta_by_seed": [],
                            "fixed_seed_work_bootstrap": _bootstrap([], seed=seed, rounds=3000),
                            "sources": {}}
            continue
        base = np.array([[r["work_values"][name][w] for w in works] for r in baselines], dtype=float)
        candidate = np.array([[r["work_values"][name][w] for w in works] for r in candidates], dtype=float)
        if not np.isfinite(base).all() or not np.isfinite(candidate).all() or np.any(base < 0) or np.any(candidate < 0):
            raise ValueError("error metrics must be finite and nonnegative")
        delta = candidate - base
        by_source = {}
        for source in sorted(set(baselines[0]["source_by_work"][w] for w in works)):
            indices = [i for i, w in enumerate(works) if baselines[0]["source_by_work"][w] == source]
            bm = float(base[:, indices].mean())
            cm = float(candidate[:, indices].mean())
            by_source[source] = {"n_works": len(indices), "baseline_mean": bm, "candidate_mean": cm,
                                 "relative_regression": (cm - bm) / bm if bm > 0 else None,
                                 "fixed_seed_work_bootstrap": _bootstrap(delta[:, indices].mean(axis=0).tolist(),
                                                                          seed=seed, rounds=3000)}
        bm, cm = float(base.mean()), float(candidate.mean())
        result[name] = {"n_works": len(works), "baseline_mean": bm, "candidate_mean": cm,
                        "relative_improvement": (bm - cm) / bm if bm > 0 else None,
                        "delta_by_seed": delta.mean(axis=1).tolist(),
                        "fixed_seed_work_bootstrap": _bootstrap(delta.mean(axis=0).tolist(), seed=seed, rounds=3000),
                        "sources": by_source}
    return {"seed_pairs": len(baselines), "seed_ids": [r["seed"] for r in baselines],
            "source_names": sorted(set(baselines[0]["source_by_work"].values())), "metrics": result,
            "interval_scope": "paired work bootstrap conditional on fixed seeds; not total seed uncertainty"}


def adoption_decision(comparisons: dict[str, dict], *, safety_passed: bool,
                      expected_sources: list[str], human_preferences_passed: bool = False,
                      untouched_holdout_passed: bool = False, actual_game_passed: bool = False) -> dict:
    """Missing evidence blocks promotion. Numerical passage is not release acceptance."""
    reasons = []
    if not safety_passed:
        reasons.append("shadow_safety_or_coverage_failed")
    if not set(POLICY["required_baselines"]).issubset(comparisons):
        reasons.append("identity_rules_or_zero_adapter_baseline_missing")
    if not expected_sources:
        reasons.append("expected_source_evidence_missing")
    for baseline, comparison in comparisons.items():
        if comparison["seed_pairs"] < POLICY["minimum_seed_pairs"]:
            reasons.append(f"{baseline}:insufficient_seeds")
        if len(set(comparison.get("seed_ids", []))) != comparison["seed_pairs"]:
            reasons.append(f"{baseline}:distinct_seed_evidence_missing")
        if set(expected_sources) != set(comparison.get("source_names", [])):
            reasons.append(f"{baseline}:expected_sources_mismatch")
        for name in POLICY["primary_metrics"] + POLICY["noninferiority_metrics"]:
            metric = comparison["metrics"].get(name)
            if metric is None or metric["n_works"] < POLICY["minimum_work_units"]:
                reasons.append(f"{baseline}:{name}:insufficient_work_coverage")
                continue
            if name in POLICY["primary_metrics"]:
                fraction = metric["relative_improvement"]
                upper = metric["fixed_seed_work_bootstrap"]["upper_95"]
                if fraction is None or fraction < POLICY["primary_improvement_fraction"] or upper is None or upper >= 0:
                    reasons.append(f"{baseline}:{name}:primary_improvement_failed")
                if not all(value < 0 for value in metric["delta_by_seed"]):
                    reasons.append(f"{baseline}:{name}:seed_consistency_failed")
                for source in expected_sources:
                    part = metric["sources"].get(source)
                    if part is None or part["n_works"] < POLICY["minimum_source_work_units"]:
                        reasons.append(f"{baseline}:{name}:{source}:source_evidence_insufficient")
                    elif part["relative_regression"] is None or part["relative_regression"] > POLICY["source_noninferiority_fraction"]:
                        reasons.append(f"{baseline}:{name}:{source}:source_noninferiority_failed")
            elif metric["candidate_mean"] > metric["baseline_mean"] or any(v > 0 for v in metric["delta_by_seed"]):
                reasons.append(f"{baseline}:{name}:long_horizon_regression")
    numerical = not reasons
    production_reasons = list(reasons)
    for passed, reason in ((human_preferences_passed, "genuine_human_preference_not_accepted"),
                           (untouched_holdout_passed, "untouched_holdout_not_accepted"),
                           (actual_game_passed, "actual_game_execution_not_accepted")):
        if not passed:
            production_reasons.append(reason)
    return {"numerical_gate_passed": numerical, "experimental_entry_authorized": False,
            "production_approved": not production_reasons, "numerical_blockers": reasons,
            "production_blockers": production_reasons,
            "next_stage": "separate CPU/fallback/interrupt acceptance before optional default-off UI"
            if numerical else "retain existing playback; research only"}
