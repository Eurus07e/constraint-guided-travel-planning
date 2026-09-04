#!/usr/bin/env python3
"""Run deterministic repair operator subsets on the stored validation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import seeded_multi_agent_planner as planner


OPERATORS = {
    "accommodation": planner.repair_accommodations,
    "cuisine": planner.repair_missing_cuisines,
    "budget": planner.repair_budget,
}
CONFIGS = {
    "selector_only": (),
    "accommodation_only": ("accommodation",),
    "cuisine_only": ("cuisine",),
    "budget_only": ("budget",),
    "without_accommodation": ("cuisine", "budget"),
    "without_cuisine": ("accommodation", "budget"),
    "without_budget": ("accommodation", "cuisine"),
    "all_operators": ("accommodation", "cuisine", "budget"),
}


def apply_operators(task: dict, seed: dict, operator_names: tuple[str, ...]) -> dict | None:
    working = planner.clone_plan(seed["plan"])
    changes: list[dict] = []
    for name in operator_names:
        working, operator_changes = OPERATORS[name](task, working)
        changes.extend(operator_changes)
    repaired_audit = planner.audit_plan(task, working)
    repaired = planner.build_candidate(
        seed["candidate_id"] + "_" + "_".join(operator_names or ("none",)),
        seed["seed_source"],
        "revised",
        1,
        task,
        working,
        planner.render_plan_text(working),
        provenance={"operators": list(operator_names), "change_log": changes},
    )
    improved = planner.candidate_rank_tuple(repaired) > planner.candidate_rank_tuple(seed)
    return repaired if improved else None


def run_config(
    rows: pd.DataFrame,
    direct: list[dict],
    program: list[dict],
    operator_names: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    output: list[dict] = []
    debug: list[dict] = []
    for row_index, row in rows.iterrows():
        idx = row_index + 1
        task = planner.build_task_context(row)
        seeds = planner.load_seed_candidates(idx, task, direct, program)
        candidates = list(seeds)
        repaired_ids: list[str] = []
        for seed in seeds:
            repaired = apply_operators(task, seed, operator_names)
            if repaired is not None:
                candidates.append(repaired)
                repaired_ids.append(repaired["candidate_id"])
        chosen = planner.rank_candidates(candidates)[0]
        output.append({"idx": idx, "query": task["query"], "plan": chosen["plan"]})
        debug.append(
            {
                "idx": idx,
                "operators": list(operator_names),
                "eligible_repaired_candidates": repaired_ids,
                "chosen_candidate": chosen["candidate_id"],
                "chosen_audit": chosen["audit"],
            }
        )
    return output, debug


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/operator_ablations"))
    args = parser.parse_args()
    rows = pd.read_csv(planner.DATA_PATH)
    direct = planner.load_jsonl(planner.DIRECT_SUBMISSION_FILE)
    program = planner.load_jsonl(planner.PROGRAM_SUBMISSION_FILE)
    for name, operator_names in CONFIGS.items():
        output, debug = run_config(rows, direct, program, operator_names)
        write_jsonl(args.output_dir / f"validation_{name}_submission.jsonl", output)
        write_jsonl(args.output_dir / f"validation_{name}_debug.jsonl", debug)
        print(name, len(output), sum(bool(row["eligible_repaired_candidates"]) for row in debug))


if __name__ == "__main__":
    main()

