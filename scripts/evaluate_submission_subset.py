#!/usr/bin/env python3
"""Evaluate an indexed subset with the released TravelPlanner constraints."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.repair_submission import load_records


def _criterion_pass(info: dict[str, Any] | None) -> tuple[bool, list[str], int, int]:
    if not info:
        return False, ["not_evaluated"], 0, 0
    failures: list[str] = []
    passed = 0
    applicable = 0
    for name, raw in info.items():
        value = raw[0] if isinstance(raw, (tuple, list)) and raw else None
        if value is None:
            continue
        applicable += 1
        if bool(value):
            passed += 1
        else:
            failures.append(name)
    return not failures, failures, passed, applicable


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    if not n:
        raise ValueError("No cases to summarize")
    common_true = sum(case["commonsense_pass"] for case in cases)
    hard_true = sum(case["hard_pass"] for case in cases)
    final_true = sum(case["final_pass"] for case in cases)
    common_passed = sum(case["commonsense_micro_passed"] for case in cases)
    common_total = sum(case["commonsense_micro_total"] for case in cases)
    hard_passed = sum(case["hard_micro_passed"] for case in cases)
    hard_total = sum(case["hard_micro_total"] for case in cases)
    return {
        "n": n,
        "delivery_rate": sum(case["delivered"] for case in cases) / n,
        "commonsense_macro_rate": common_true / n,
        "hard_macro_rate": hard_true / n,
        "final_pass_rate": final_true / n,
        "final_pass_count": final_true,
        "commonsense_micro_rate": common_passed / common_total if common_total else None,
        "hard_micro_rate": hard_passed / hard_total if hard_total else None,
    }


def evaluate(input_path: Path, set_type: str) -> dict[str, Any]:
    input_path = input_path.resolve()
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = load_records(set_type)

    from evaluation.commonsense_constraint import evaluation as commonsense_eval
    from evaluation.hard_constraint import evaluation as hard_eval

    cases: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        idx = int(row["idx"])
        if idx in seen:
            raise ValueError(f"Duplicate submission idx: {idx}")
        seen.add(idx)
        if not 1 <= idx <= len(records):
            raise IndexError(f"Submission idx outside dataset: {idx}")
        record = dict(records[idx - 1])
        if isinstance(record.get("local_constraint"), str):
            record["local_constraint"] = ast.literal_eval(record["local_constraint"])
        plan = row.get("plan")
        delivered = isinstance(plan, list) and bool(plan)
        common_info = commonsense_eval(record, plan) if delivered else None
        hard_info = None
        if common_info and common_info["is_not_absent"][0] and common_info["is_valid_information_in_sandbox"][0]:
            hard_info = hard_eval(record, plan)
        common_pass, common_failures, common_micro_passed, common_micro_total = _criterion_pass(common_info)
        hard_pass, hard_failures, hard_micro_passed, hard_micro_total = _criterion_pass(hard_info)
        cases.append(
            {
                "idx": idx,
                "level": record["level"],
                "days": int(record["days"]),
                "delivered": delivered,
                "commonsense_pass": common_pass,
                "hard_pass": hard_pass,
                "final_pass": common_pass and hard_pass,
                "commonsense_failures": common_failures,
                "hard_failures": hard_failures,
                "commonsense_micro_passed": common_micro_passed,
                "commonsense_micro_total": common_micro_total,
                "hard_micro_passed": hard_micro_passed,
                "hard_micro_total": hard_micro_total,
            }
        )

    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_days: dict[int, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    for case in cases:
        by_level[case["level"]].append(case)
        by_days[case["days"]].append(case)
        failure_counts.update(f"commonsense:{name}" for name in case["commonsense_failures"])
        failure_counts.update(f"hard:{name}" for name in case["hard_failures"])
    return {
        "set_type": set_type,
        "input": str(input_path),
        "summary": summarize_cases(cases),
        "by_level": {level: summarize_cases(group) for level, group in sorted(by_level.items())},
        "by_days": {str(days): summarize_cases(group) for days, group in sorted(by_days.items())},
        "failure_counts": dict(failure_counts.most_common()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--set-type", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.input, args.set_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

