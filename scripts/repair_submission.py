#!/usr/bin/env python3
"""Apply the stored deterministic repair operators to one submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from scripts.heldout_query_adapter import recover_evaluator_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-output", type=Path, required=True)
    parser.add_argument("--set-type", choices=("train", "validation", "test"), default="test")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records(set_type: str) -> list[dict]:
    if set_type == "validation":
        import pandas as pd

        return pd.read_csv("database/validation.csv").to_dict("records")
    cache_paths = sorted(
        Path.home().glob(
            f".cache/huggingface/datasets/osunlp___travel_planner/{set_type}/*/*/travel_planner-{set_type}.arrow"
        )
    )
    if cache_paths:
        dataset = Dataset.from_file(str(cache_paths[-1]))
    else:
        dataset = load_dataset("osunlp/TravelPlanner", set_type, download_mode="reuse_cache_if_exists")[set_type]
    records = [dict(record) for record in dataset]
    return [recover_evaluator_fields(record) for record in records] if set_type == "test" else records


def repair_rows(submission: list[dict], records: list[dict]) -> tuple[list[dict], list[dict]]:
    import seeded_multi_agent_planner as planner

    output_rows: list[dict] = []
    debug_rows: list[dict] = []
    for row in submission:
        idx = int(row["idx"])
        if not 1 <= idx <= len(records):
            raise IndexError(f"Submission idx outside dataset: {idx}")
        task = planner.build_task_context(records[idx - 1])
        plan = planner.normalize_plan_list(row.get("plan"), int(task["days"]))
        if len(plan) != int(task["days"]):
            raise ValueError(f"Invalid plan length at idx={idx}")
        candidate = planner.build_candidate(
            "input_seed",
            "direct",
            "seed",
            0,
            task,
            plan,
            planner.render_plan_text(plan),
            provenance={"source": "input_submission"},
        )
        repaired = planner.deterministic_repair(task, candidate)
        selected_plan = repaired["plan"] if repaired["improved"] else plan
        output_rows.append({"idx": idx, "query": task["query"], "plan": selected_plan})
        debug_rows.append(
            {
                "idx": idx,
                "improved": repaired["improved"],
                "change_log": repaired["change_log"],
                "original_audit": repaired["original_audit"],
                "repaired_audit": repaired["audit"],
            }
        )
    return output_rows, debug_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    submission = load_jsonl(args.input)
    records = load_records(args.set_type)
    output_rows, debug_rows = repair_rows(submission, records)
    write_jsonl(args.output, output_rows)
    write_jsonl(args.debug_output, debug_rows)
    print(f"repaired {len(output_rows)} rows; improved {sum(row['improved'] for row in debug_rows)}")


if __name__ == "__main__":
    main()

