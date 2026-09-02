import ast
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_RESULTS_DIR = ROOT / "evaluation" / "experiment_results"
STYLES_DIR = ROOT / "paper" / "source"
ASSETS_DIR = STYLES_DIR / "assets"
MULTI_AGENT_DEBUG_DIR = ROOT / "outputs_multi_agent_seeded_r2_n3" / "debug"

METHOD_SPECS = [
    {
        "display_name": "Direct Prompt",
        "method_id": "direct",
        "metrics_json": EVAL_RESULTS_DIR / "direct_prompt_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_direct_sole-planning_submission.jsonl",
        "family": "baseline",
    },
    {
        "display_name": "Constraint-Checklist Direct",
        "method_id": "constraint_direct_json",
        "metrics_json": EVAL_RESULTS_DIR / "constraint_direct_json_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_constraint_direct_json_sole-planning_submission.jsonl",
        "family": "strong_baseline",
    },
    {
        "display_name": "Generic Self-Refine",
        "method_id": "generic_self_refine_r1",
        "metrics_json": EVAL_RESULTS_DIR / "generic_self_refine_r1_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_generic_self_refine_r1_sole-planning_submission.jsonl",
        "family": "strong_baseline",
    },
    {
        "display_name": "Program Planner v1",
        "method_id": "program_v1",
        "metrics_json": EVAL_RESULTS_DIR / "program_v1_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_program_sole-planning_submission.jsonl",
        "family": "baseline",
    },
    {
        "display_name": "LLM-Assisted Program Planner",
        "method_id": "program_v23_best",
        "metrics_json": EVAL_RESULTS_DIR / "program_v23_best_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_program_v23_best_sole-planning_submission.jsonl",
        "family": "baseline",
    },
    {
        "display_name": "Verifier-Guided Hybrid Selector",
        "method_id": "verifier_hybrid_agent",
        "metrics_json": EVAL_RESULTS_DIR / "verifier_hybrid_agent_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_verifier_hybrid_agent_sole-planning_submission.jsonl",
        "family": "control",
    },
    {
        "display_name": "Verifier-Directed Repair Control",
        "method_id": "seeded_repair_probe",
        "metrics_json": EVAL_RESULTS_DIR / "seeded_repair_probe_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_seeded_repair_probe_sole-planning_submission.jsonl",
        "family": "control",
    },
    {
        "display_name": "Seeded Multi-Agent Planner",
        "method_id": "multi_agent_seeded_r2_n3",
        "metrics_json": EVAL_RESULTS_DIR / "multi_agent_seeded_r2_n3_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_multi_agent_seeded_r2_n3_sole-planning_submission.jsonl",
        "family": "main",
    },
    {
        "display_name": "CC-MAR",
        "method_id": "cc_mar_r3",
        "metrics_json": EVAL_RESULTS_DIR / "cc_mar_r3_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_cc_mar_r3_sole-planning_submission.jsonl",
        "family": "cc_mar",
    },
    {
        "display_name": "CC-MAR w/o Critic",
        "method_id": "cc_mar_no_critic",
        "metrics_json": EVAL_RESULTS_DIR / "cc_mar_no_critic_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_cc_mar_no_critic_sole-planning_submission.jsonl",
        "family": "cc_mar_ablation",
    },
    {
        "display_name": "CC-MAR w/o Board",
        "method_id": "cc_mar_no_board",
        "metrics_json": EVAL_RESULTS_DIR / "cc_mar_no_board_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_cc_mar_no_board_sole-planning_submission.jsonl",
        "family": "cc_mar_ablation",
    },
    {
        "display_name": "CC-MAR w/o Specialization",
        "method_id": "cc_mar_no_specialization",
        "metrics_json": EVAL_RESULTS_DIR / "cc_mar_no_specialization_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_cc_mar_no_specialization_sole-planning_submission.jsonl",
        "family": "cc_mar_ablation",
    },
    {
        "display_name": "Single Generic Repair Agent",
        "method_id": "single_generic_repair_agent",
        "metrics_json": EVAL_RESULTS_DIR / "single_generic_repair_agent_metrics_local.json",
        "submission_file": ROOT / "evaluation" / "validation_deepseek-v4-flash_single_generic_repair_agent_sole-planning_submission.jsonl",
        "family": "cc_mar_ablation",
    },
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value):
    return round(float(value) * 100.0, 2)


def available_methods():
    methods = []
    for spec in METHOD_SPECS:
        if spec["metrics_json"].exists():
            payload = load_json(spec["metrics_json"])
            scores = payload["scores"]
            methods.append(
                {
                    **spec,
                    "scores": scores,
                    "delivery_rate": pct(scores["Delivery Rate"]),
                    "commonsense_micro": pct(scores["Commonsense Constraint Micro Pass Rate"]),
                    "commonsense_macro": pct(scores["Commonsense Constraint Macro Pass Rate"]),
                    "hard_micro": pct(scores["Hard Constraint Micro Pass Rate"]),
                    "hard_macro": pct(scores["Hard Constraint Macro Pass Rate"]),
                    "final_pass_rate": pct(scores["Final Pass Rate"]),
                }
            )
    return methods


def write_metrics_summary(methods):
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method["display_name"],
                "method_id": method["method_id"],
                "delivery_rate": method["delivery_rate"],
                "commonsense_micro": method["commonsense_micro"],
                "commonsense_macro": method["commonsense_macro"],
                "hard_micro": method["hard_micro"],
                "hard_macro": method["hard_macro"],
                "final_pass_rate": method["final_pass_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_RESULTS_DIR / "metrics_summary.csv", index=False)

    table_rows = [
        row for row in rows
        if next(spec for spec in METHOD_SPECS if spec["method_id"] == row["method_id"])["family"] != "cc_mar_ablation"
    ]
    best = {
        key: max((row[key] for row in table_rows), default=0.0)
        for key in ["final_pass_rate", "commonsense_macro", "hard_macro", "commonsense_micro", "hard_micro"]
    }
    second = {}
    for key in best:
        values = sorted({row[key] for row in table_rows}, reverse=True)
        second[key] = values[1] if len(values) > 1 else None

    def fmt(value, key):
        text = f"{value:.2f}"
        if abs(value - best[key]) < 1e-9:
            return rf"\textbf{{{text}}}"
        if second[key] is not None and abs(value - second[key]) < 1e-9:
            return rf"\underline{{{text}}}"
        return text

    tex_lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Final Pass (\%) $\uparrow$ & Common.\ Macro (\%) $\uparrow$ & Hard Macro (\%) $\uparrow$ & Common.\ Micro (\%) $\uparrow$ & Hard Micro (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    family_labels = [
        ("baseline", "Prompt-only and program baselines"),
        ("strong_baseline", "Stronger prompting baselines"),
        ("control", "Verifier and repair controls"),
        ("main", "Seeded pipeline"),
        ("cc_mar", "Contract multi-agent repair"),
    ]
    for family, label in family_labels:
        family_rows = [row for row in table_rows if next(spec for spec in METHOD_SPECS if spec["method_id"] == row["method_id"])["family"] == family]
        if not family_rows:
            continue
        tex_lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{label}}}}} \\")
        for row in family_rows:
            tex_lines.append(
                "{} & {} & {} & {} & {} & {} \\\\".format(
                    row["method"],
                    fmt(row["final_pass_rate"], "final_pass_rate"),
                    fmt(row["commonsense_macro"], "commonsense_macro"),
                    fmt(row["hard_macro"], "hard_macro"),
                    fmt(row["commonsense_micro"], "commonsense_micro"),
                    fmt(row["hard_micro"], "hard_micro"),
                )
            )
        tex_lines.append(r"\midrule")
    if tex_lines[-1] == r"\midrule":
        tex_lines.pop()
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ASSETS_DIR / "results_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return rows


def build_final_pass_plot(rows):
    rows = [row for row in rows if next(spec for spec in METHOD_SPECS if spec["method_id"] == row["method_id"])["family"] != "cc_mar_ablation"]
    methods = [row["method"] for row in rows]
    final_scores = [row["final_pass_rate"] for row in rows]

    plt.figure(figsize=(10.5, 4.6))
    colors = ["#8da0cb", "#fc8d62", "#66c2a5", "#e78ac3", "#a6d854", "#ffd92f", "#4c72b0", "#55a868"][: len(rows)]
    bars = plt.bar(methods, final_scores, color=colors)
    plt.ylabel("Final Pass Rate (%)")
    plt.xticks(rotation=18, ha="right")
    plt.ylim(0, max(final_scores) + 8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.6, f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "final_pass_rate.png", dpi=220)
    plt.close()


def build_macro_plot(rows):
    rows = [row for row in rows if next(spec for spec in METHOD_SPECS if spec["method_id"] == row["method_id"])["family"] != "cc_mar_ablation"]
    methods = [row["method"] for row in rows]
    c_macro = [row["commonsense_macro"] for row in rows]
    h_macro = [row["hard_macro"] for row in rows]
    x = range(len(rows))

    plt.figure(figsize=(10.5, 4.8))
    width = 0.36
    plt.bar([i - width / 2 for i in x], c_macro, width=width, label="Commonsense Macro", color="#4c72b0")
    plt.bar([i + width / 2 for i in x], h_macro, width=width, label="Hard Macro", color="#dd8452")
    plt.xticks(list(x), methods, rotation=18, ha="right")
    plt.ylabel("Pass Rate (%)")
    plt.ylim(0, max(c_macro + h_macro) + 10)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "macro_breakdown.png", dpi=220)
    plt.close()


def incremental_llm_calls(debug_payload):
    total = 0
    for stage in debug_payload.get("stages", []):
        name = stage.get("stage")
        if name == "planner_round1":
            provenance = ((stage.get("candidate") or {}).get("provenance")) or {}
            if isinstance(provenance, dict) and provenance.get("planner_status") != "skipped_high_confidence_seed_pair":
                total += 1
        elif name == "verifier_round1":
            for candidate in stage.get("candidates", []) or []:
                llm_verifier = candidate.get("llm_verifier") or {}
                if isinstance(llm_verifier, dict) and llm_verifier.get("summary") != "deterministic_audit_only":
                    total += 1
        elif name == "reflector_round2":
            reflection = stage.get("reflection") or {}
            if isinstance(reflection, dict) and reflection.get("reflection_mode") != "deterministic":
                total += 1
        elif name == "reviser_round2":
            provenance = stage.get("provenance") or {}
            if isinstance(provenance, dict) and provenance.get("repair_mode") != "deterministic_short_circuit":
                if "reviser_output" in provenance or provenance.get("reviser_status") == "fallback_original_candidate":
                    total += 1
    return total


def summarize_multi_agent_debug():
    if not MULTI_AGENT_DEBUG_DIR.exists():
        return None

    debug_paths = sorted(
        MULTI_AGENT_DEBUG_DIR.glob("debug_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not debug_paths:
        return None

    final_choice_counts = Counter()
    reflection_mode_counts = Counter()
    llm_call_counts = []
    revised_selected = 0
    seed_selected = 0
    scratch_generated = 0
    scratch_skipped = 0

    for path in debug_paths:
        payload = load_json(path)
        final_choice = str(payload.get("final_choice", ""))
        final_choice_counts[final_choice] += 1
        if final_choice.endswith("_revised_r2"):
            revised_selected += 1
        else:
            seed_selected += 1

        llm_call_counts.append(incremental_llm_calls(payload))

        for stage in payload.get("stages", []):
            stage_name = stage.get("stage")
            if stage_name == "planner_round1":
                provenance = ((stage.get("candidate") or {}).get("provenance")) or {}
                if isinstance(provenance, dict) and provenance.get("planner_status") == "skipped_high_confidence_seed_pair":
                    scratch_skipped += 1
                else:
                    scratch_generated += 1
            elif stage_name == "reflector_round2":
                reflection = stage.get("reflection") or {}
                if isinstance(reflection, dict):
                    reflection_mode_counts[reflection.get("reflection_mode", "llm")] += 1

    return {
        "num_instances": len(debug_paths),
        "final_choice_counts": dict(final_choice_counts),
        "revised_selected": revised_selected,
        "seed_selected": seed_selected,
        "scratch_generated": scratch_generated,
        "scratch_skipped": scratch_skipped,
        "deterministic_reflection": reflection_mode_counts.get("deterministic", 0),
        "llm_reflection": reflection_mode_counts.get("llm", 0),
        "avg_incremental_llm_calls": round(sum(llm_call_counts) / len(llm_call_counts), 2),
        "max_incremental_llm_calls": max(llm_call_counts),
    }


PAPER_TERM_MAP = {
    "is_valid_information_in_current_city": "Within Current City",
    "is_valid_information_in_sandbox": "Within Sandbox",
    "is_reasonable_visiting_city": "Reasonable City Route",
    "is_valid_restaurants": "Diverse Restaurants",
    "is_valid_transportation": "Non-conf. Transportation",
    "is_valid_attractions": "Diverse Attractions",
    "is_valid_accommodation": "Minimum Nights Stay",
    "is_not_absent": "Complete Information",
    "valid_cost": "Budget",
    "valid_room_rule": "Room Rule",
    "valid_cuisine": "Cuisine",
    "valid_room_type": "Room Type",
    "valid_transportation": "Transportation",
}


def evaluate_plan_failures(query_data, plan, commonsense_eval, hard_eval):
    if isinstance(query_data["local_constraint"], str):
        query_data["local_constraint"] = ast.literal_eval(query_data["local_constraint"])

    commonsense_box = commonsense_eval(query_data, plan) if plan else None
    if not commonsense_box:
        return False, ["Complete Information"]

    failed_categories = []
    for key, value in commonsense_box.items():
        if value[0] is False:
            failed_categories.append(PAPER_TERM_MAP[key])

    if commonsense_box and commonsense_box["is_not_absent"][0] and commonsense_box["is_valid_information_in_sandbox"][0]:
        hard_box = hard_eval(query_data, plan)
    else:
        hard_box = None

    if not hard_box:
        return len(failed_categories) == 0, sorted(set(failed_categories or ["Complete Information"]))
    for key, value in hard_box.items():
        if value[0] is False:
            failed_categories.append(PAPER_TERM_MAP[key])

    return len(failed_categories) == 0, sorted(set(failed_categories))


def load_cached_breakdown(methods):
    path = EVAL_RESULTS_DIR / "paper_artifact_summary.json"
    if not path.exists():
        return None
    payload = load_json(path)
    breakdown = payload.get("breakdown")
    if not isinstance(breakdown, dict):
        return None

    required = {method["method_id"] for method in methods}
    if not required.issubset(set(breakdown.keys())):
        return None
    return breakdown


def compute_breakdowns(methods):
    cached = load_cached_breakdown(methods)
    if cached:
        return cached

    sys.path.append(str(ROOT))
    sys.path.append(str(ROOT / "evaluation"))
    from commonsense_constraint import evaluation as commonsense_eval
    from hard_constraint import evaluation as hard_eval

    validation_df = pd.read_csv(ROOT / "database" / "validation.csv")
    breakdown = {}

    for method in methods:
        if not method["submission_file"].exists():
            continue

        rows = load_jsonl(method["submission_file"])
        by_difficulty = defaultdict(lambda: [0, 0])
        by_days = defaultdict(lambda: [0, 0])
        per_case = []

        for idx, row in validation_df.iterrows():
            query_data = dict(row)
            passed, _ = evaluate_plan_failures(query_data, rows[idx]["plan"], commonsense_eval, hard_eval)
            difficulty = str(row["level"])
            days = str(row["days"])
            by_difficulty[difficulty][1] += 1
            by_days[days][1] += 1
            if passed:
                by_difficulty[difficulty][0] += 1
                by_days[days][0] += 1
            per_case.append(bool(passed))

        breakdown[method["method_id"]] = {
            "difficulty": {key: round(value[0] / value[1] * 100, 2) for key, value in by_difficulty.items()},
            "days": {key: round(value[0] / value[1] * 100, 2) for key, value in by_days.items()},
            "per_case": per_case,
        }

    return breakdown


def compute_error_taxonomy(methods):
    target = next((method for method in methods if method["method_id"] == "multi_agent_seeded_r2_n3"), None)
    if not target or not target["metrics_json"].exists():
        return None

    counter = Counter()
    payload = load_json(target["metrics_json"])
    detailed_scores = payload.get("detailed_scores", {})

    for constraint_family in detailed_scores.values():
        for level_dict in constraint_family.values():
            for day_dict in level_dict.values():
                for category, values in day_dict.items():
                    counter[category] += int(values.get("false", 0))

    total_false_checks = sum(counter.values())

    return {
        "total_false_checks": total_false_checks,
        "rows": [
            {
                "category": category,
                "count": count,
                "share": round(count / total_false_checks * 100.0, 2) if total_false_checks else 0.0,
            }
            for category, count in counter.most_common()
        ],
    }


def write_breakdown_tables(methods, breakdown):
    difficulty_cols = ["easy", "medium", "hard"]
    days_cols = ["3", "5", "7"]

    diff_lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Easy (\%) $\uparrow$ & Medium (\%) $\uparrow$ & Hard (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    day_lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & 3 days (\%) $\uparrow$ & 5 days (\%) $\uparrow$ & 7 days (\%) $\uparrow$ \\",
        r"\midrule",
    ]

    for method in methods:
        if method["family"] == "cc_mar_ablation":
            continue
        item = breakdown.get(method["method_id"])
        if not item:
            continue
        diff_lines.append(
            "{} & {} & {} & {} \\\\".format(
                method["display_name"],
                *(f"{item['difficulty'].get(col, 0):.2f}" for col in difficulty_cols),
            )
        )
        day_lines.append(
            "{} & {} & {} & {} \\\\".format(
                method["display_name"],
                *(f"{item['days'].get(col, 0):.2f}" for col in days_cols),
            )
        )

    diff_lines.extend([r"\bottomrule", r"\end{tabular}"])
    day_lines.extend([r"\bottomrule", r"\end{tabular}"])

    (ASSETS_DIR / "difficulty_breakdown_table.tex").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
    (ASSETS_DIR / "days_breakdown_table.tex").write_text("\n".join(day_lines) + "\n", encoding="utf-8")


def write_error_taxonomy_table(error_taxonomy):
    if not error_taxonomy:
        return

    tex_lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Failure category & False checks & Share of all false checks (\%) \\",
        r"\midrule",
    ]
    for row in error_taxonomy["rows"][:8]:
        tex_lines.append(
            "{} & {} & {:.2f} \\\\".format(
                row["category"],
                row["count"],
                row["share"],
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ASSETS_DIR / "error_taxonomy_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_progressive_ablation_table(methods):
    lookup = {method["method_id"]: method for method in methods}
    rows = [
        ("Direct Prompt", "direct"),
        ("+ Verifier-Based Seed Selection", "verifier_hybrid_agent"),
        ("+ Deterministic Repair", "seeded_repair_probe"),
        ("+ Full Seeded Multi-Agent Loop", "multi_agent_seeded_r2_n3"),
    ]

    tex_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Variant & Final Pass (\%) $\uparrow$ & $\Delta$ Final vs.\ prev. & Common.\ Macro (\%) $\uparrow$ & Hard Macro (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    prev_final = None
    for label, method_id in rows:
        method = lookup.get(method_id)
        if not method:
            continue
        delta = "--" if prev_final is None else f"{method['final_pass_rate'] - prev_final:+.2f}"
        prev_final = method["final_pass_rate"]
        tex_lines.append(
            "{} & {:.2f} & {} & {:.2f} & {:.2f} \\\\".format(
                label,
                method["final_pass_rate"],
                delta,
                method["commonsense_macro"],
                method["hard_macro"],
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ASSETS_DIR / "ablation_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_cc_mar_ablation_table(methods):
    lookup = {method["method_id"]: method for method in methods}
    rows = [
        ("CC-MAR", "cc_mar_r3"),
        ("w/o Critic Agent", "cc_mar_no_critic"),
        ("w/o Communication Board", "cc_mar_no_board"),
        ("w/o Role Specialization", "cc_mar_no_specialization"),
        ("Single Generic Repair Agent", "single_generic_repair_agent"),
    ]
    present = [(label, lookup[method_id]) for label, method_id in rows if method_id in lookup]
    if not present:
        return

    tex_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Variant & Final Pass (\%) $\uparrow$ & Common.\ Macro (\%) $\uparrow$ & Hard Macro (\%) $\uparrow$ & Hard Micro (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    for label, method in present:
        tex_lines.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                label,
                method["final_pass_rate"],
                method["commonsense_macro"],
                method["hard_macro"],
                method["hard_micro"],
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ASSETS_DIR / "cc_mar_ablation_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_agent_profile_assets(summary):
    if not summary:
        return

    n = summary["num_instances"]
    tex_lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Statistic & Count / 180 & Share / Value \\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{Selection behavior}} \\",
        f"Final revised candidate selected & {summary['revised_selected']} & {summary['revised_selected'] / n * 100:.1f}\\% \\\\",
        f"Original seed selected & {summary['seed_selected']} & {summary['seed_selected'] / n * 100:.1f}\\% \\\\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{Adaptive generation}} \\",
        f"Scratch planner generated & {summary['scratch_generated']} & {summary['scratch_generated'] / n * 100:.1f}\\% \\\\",
        f"Scratch planner skipped & {summary['scratch_skipped']} & {summary['scratch_skipped'] / n * 100:.1f}\\% \\\\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{Reflection mode}} \\",
        f"Deterministic reflection & {summary['deterministic_reflection']} & {summary['deterministic_reflection'] / n * 100:.1f}\\% \\\\",
        f"LLM reflection & {summary['llm_reflection']} & {summary['llm_reflection'] / n * 100:.1f}\\% \\\\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{LLM budget}} \\",
        rf"Average incremental LLM calls / instance & \multicolumn{{2}}{{c@{{}}}}{{{summary['avg_incremental_llm_calls']:.2f}}} \\",
        rf"Maximum incremental LLM calls & \multicolumn{{2}}{{c@{{}}}}{{{summary['max_incremental_llm_calls']}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ASSETS_DIR / "agent_profile_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    ordered = sorted(summary["final_choice_counts"].items(), key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in ordered]
    values = [value for _, value in ordered]
    pretty_labels = [label.replace("_", "\n") for label in labels]

    plt.figure(figsize=(8.8, 4.6))
    bars = plt.bar(pretty_labels, values, color=["#55a868", "#4c72b0", "#dd8452", "#c44e52"][: len(values)])
    plt.ylabel("Validation instances")
    plt.ylim(0, max(values) + 12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "selection_distribution.png", dpi=220)
    plt.close()

    (EVAL_RESULTS_DIR / "multi_agent_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_breakdown_plot(methods, breakdown):
    if "direct" not in breakdown:
        return

    selected_ids = [method["method_id"] for method in methods if method["method_id"] in {"direct", "verifier_hybrid_agent", "seeded_repair_probe", "multi_agent_seeded_r2_n3"}]
    labels = [method["display_name"] for method in methods if method["method_id"] in selected_ids]
    day_cols = ["3", "5", "7"]
    x = range(len(day_cols))
    width = 0.18

    plt.figure(figsize=(9.6, 4.8))
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for offset, method_id in enumerate(selected_ids):
        values = [breakdown[method_id]["days"].get(day, 0) for day in day_cols]
        positions = [i + (offset - (len(selected_ids) - 1) / 2) * width for i in x]
        plt.bar(positions, values, width=width, label=next(method["display_name"] for method in methods if method["method_id"] == method_id), color=palette[offset])
    plt.xticks(list(x), ["3-day", "5-day", "7-day"])
    plt.ylabel("Final Pass Rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "days_breakdown.png", dpi=220)
    plt.close()


def write_analysis_summary(methods, breakdown, multi_agent_summary, error_taxonomy):
    payload = {
        "methods_present": [method["method_id"] for method in methods],
        "metrics_rows": [
            {
                "method_id": method["method_id"],
                "display_name": method["display_name"],
                "final_pass_rate": method["final_pass_rate"],
                "commonsense_macro": method["commonsense_macro"],
                "hard_macro": method["hard_macro"],
            }
            for method in methods
        ],
        "breakdown": breakdown,
        "multi_agent_profile": multi_agent_summary,
        "error_taxonomy": error_taxonomy,
    }
    (EVAL_RESULTS_DIR / "paper_artifact_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = available_methods()
    if not methods:
        raise RuntimeError("No metrics JSON files found in evaluation/experiment_results.")

    rows = write_metrics_summary(methods)
    build_final_pass_plot(rows)
    build_macro_plot(rows)
    breakdown = compute_breakdowns(methods)
    write_breakdown_tables(methods, breakdown)
    build_breakdown_plot(methods, breakdown)
    error_taxonomy = compute_error_taxonomy(methods)
    write_error_taxonomy_table(error_taxonomy)
    write_progressive_ablation_table(methods)
    write_cc_mar_ablation_table(methods)
    multi_agent_summary = summarize_multi_agent_debug()
    write_agent_profile_assets(multi_agent_summary)
    write_analysis_summary(methods, breakdown, multi_agent_summary, error_taxonomy)


if __name__ == "__main__":
    main()
