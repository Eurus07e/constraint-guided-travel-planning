#!/usr/bin/env python3
"""Compare the internal audit decision with released-evaluator Final Pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.plan_audit import audit_plan


METHOD_FILES = {
    "direct": "validation_deepseek-v4-flash_direct_sole-planning_submission.jsonl",
    "verifier_hybrid_agent": "validation_deepseek-v4-flash_verifier_hybrid_agent_sole-planning_submission.jsonl",
    "seeded_repair_probe": "validation_deepseek-v4-flash_seeded_repair_probe_sole-planning_submission.jsonl",
    "multi_agent_seeded_r2_n3": "validation_deepseek-v4-flash_multi_agent_seeded_r2_n3_sole-planning_submission.jsonl",
    "cc_mar_r3": "validation_deepseek-v4-flash_cc_mar_r3_sole-planning_submission.jsonl",
}


def confusion_counts(predicted: list[bool], observed: list[bool]) -> dict[str, int]:
    if len(predicted) != len(observed):
        raise ValueError("Predicted and observed vectors must have equal length")
    if any(type(value) is not bool for value in predicted + observed):
        raise TypeError("Confusion vectors must contain booleans")
    return {
        "tp": sum(p and o for p, o in zip(predicted, observed)),
        "fp": sum(p and not o for p, o in zip(predicted, observed)),
        "fn": sum(not p and o for p, o in zip(predicted, observed)),
        "tn": sum(not p and not o for p, o in zip(predicted, observed)),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def confusion_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "negative_predictive_value": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, total),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_final_vectors(path: Path) -> dict[str, list[bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {method: details["per_case"] for method, details in payload["breakdown"].items()}


def build_task(row: pd.Series) -> dict:
    import seeded_multi_agent_planner as planner

    return planner.build_task_context(row)


def analyze(root: Path) -> dict:
    root = root.resolve()
    records = pd.read_csv(root / "database" / "validation.csv")
    vectors = load_final_vectors(root / "evaluation" / "experiment_results" / "paper_artifact_summary.json")
    methods: dict[str, dict] = {}
    for method, filename in METHOD_FILES.items():
        rows = load_jsonl(root / "evaluation" / filename)
        if len(rows) != len(records):
            raise ValueError(f"Expected {len(records)} rows for {method}, got {len(rows)}")
        predicted: list[bool] = []
        audit_scores: list[int] = []
        fatal_counts: list[int] = []
        for position, (record_idx, record) in enumerate(records.iterrows(), 1):
            row = rows[position - 1]
            if int(row["idx"]) != position:
                raise ValueError(f"Non-contiguous submission idx for {method} at {position}")
            audit = audit_plan(build_task(record), row["plan"])
            predicted.append(bool(audit["likely_pass"]))
            audit_scores.append(int(audit["score"]))
            fatal_counts.append(int(audit["fatal_count"]))
        observed = list(vectors[method])
        counts = confusion_counts(predicted, observed)
        methods[method] = {
            "n": len(observed),
            "audit_likely_pass": sum(predicted),
            "scorer_final_pass": sum(observed),
            "confusion": counts,
            "metrics": confusion_metrics(counts),
            "mean_audit_score": sum(audit_scores) / len(audit_scores),
            "mean_fatal_count": sum(fatal_counts) / len(fatal_counts),
        }
    return {"methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

