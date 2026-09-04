import ast
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
from utils.llm_client import complete, ModelRequestError
from utils.run_state import configure_run, atomic_write_text, exclusive_run
from utils.runner_inputs import select_rows, valid_plan_container, load_seed_submission as load_jsonl
from utils.paths import DATABASE, seed_path

from utils.plan_audit import (
    _check_local_constraints,
    _estimate_total_cost,
    _load_accommodations,
    _load_restaurants,
    _lookup_accommodation,
    _normalize_budget,
    audit_plan,
    get_valid_name_city,
)


MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
ROUNDS = int(os.getenv("ROUNDS", "2"))
CANDIDATES = int(os.getenv("CANDIDATES", "3"))
STRATEGY = os.getenv("STRATEGY", f"multi_agent_seeded_r{ROUNDS}_n{CANDIDATES}")
MODE = "sole-planning"

DATA_PATH = DATABASE / "validation.csv"

DIRECT_OUTPUT_DIR = Path("outputs/validation")
PROGRAM_OUTPUT_DIR = Path("outputs_program_v23_best/validation")
DIRECT_SUBMISSION_FILE = seed_path("direct")
PROGRAM_SUBMISSION_FILE = seed_path("program")

OUT_ROOT = Path(os.getenv("OUT_ROOT", f"outputs_{STRATEGY}"))
OUTPUT_DIR = OUT_ROOT / "validation"
DEBUG_DIR = OUT_ROOT / "debug"
CACHE_DIR = OUT_ROOT / "llm_cache"
SUBMISSION_DIR = Path("evaluation")
SUBMISSION_FILE = SUBMISSION_DIR / f"validation_{MODEL_NAME}_{STRATEGY}_{MODE}_submission.jsonl"

MAX_REFERENCE_CHARS = int(os.getenv("MAX_REFERENCE_CHARS", "12000"))
ALLOW_LLM = os.getenv("ALLOW_LLM", "1") == "1"
RESUME = os.getenv("RESUME", "1") == "1"
STRICT_ROUTE_PATCH_ONLY = os.getenv("STRICT_ROUTE_PATCH_ONLY", "0") == "1"
NO_FREE_REGEN = os.getenv("NO_FREE_REGEN", "0") == "1"
PROMOTION_GAP = int(os.getenv("PROMOTION_GAP", "8"))
TIE_BREAK_GAP = int(os.getenv("TIE_BREAK_GAP", "4"))
REPAIR_TARGETS = int(os.getenv("REPAIR_TARGETS", "2"))
SELF_GEN_TEMPERATURE = float(os.getenv("SELF_GEN_TEMPERATURE", "0.25"))
REVISE_TEMPERATURE = float(os.getenv("REVISE_TEMPERATURE", "0.05"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REPAIR_SCORE_FLOOR = int(os.getenv("REPAIR_SCORE_FLOOR", "48"))
HOPELESS_FATAL_THRESHOLD = int(os.getenv("HOPELESS_FATAL_THRESHOLD", "4"))

STATE_NAMES = {
    "Florida", "Texas", "California", "Illinois", "Michigan", "New York",
    "South Carolina", "North Carolina", "Georgia", "Washington", "Virginia",
    "Ohio", "Pennsylvania", "Missouri", "Arizona", "Nevada", "Colorado",
    "Tennessee", "Louisiana", "Alabama", "Oregon", "Maryland", "Indiana",
}

REQUIRED_PLAN_KEYS = [
    "days", "current_city", "transportation", "breakfast",
    "attraction", "lunch", "dinner", "accommodation",
]


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def safe_literal_eval(value, default):
    try:
        if pd.isna(value):
            return default
        return ast.literal_eval(str(value))
    except Exception:
        return default


def parse_any_json(text):
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    candidates = []
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    for raw in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return None


def chat_completions_url(api_base):
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def call_deepseek_text(system_prompt, user_prompt, max_tokens=4096, temperature=0.0, cache_tag="default"):
    return complete(system_prompt, user_prompt, cache_dir=CACHE_DIR, model=MODEL_NAME,
                    max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag, json_mode=False)


def call_deepseek_json(system_prompt, user_prompt, max_tokens=4096, temperature=0.0, cache_tag="default"):
    text = call_deepseek_text(
        system_prompt,
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_tag=cache_tag,
    )
    return parse_any_json(text)


def extract_candidate_cities(ref):
    sections = safe_literal_eval(ref, [])
    cities = []
    if not isinstance(sections, list):
        return cities

    patterns = [
        r"Attractions in (.+)",
        r"Restaurants in (.+)",
        r"Accommodations in (.+)",
        r"Flight from (.+?) to (.+?) on",
        r"Self-driving from (.+?) to (.+)",
        r"Taxi from (.+?) to (.+)",
    ]

    for sec in sections:
        desc = str(sec.get("Description", ""))
        for pat in patterns:
            match = re.match(pat, desc)
            if not match:
                continue
            for group in match.groups():
                city = re.sub(r"\s+on\s+.*$", "", str(group).strip()).strip()
                if city and city not in STATE_NAMES and city not in cities:
                    cities.append(city)
    return cities


def compact_reference_information(ref, max_chars=MAX_REFERENCE_CHARS):
    sections = safe_literal_eval(ref, [])
    if not isinstance(sections, list):
        return str(ref)[:max_chars]

    useful = []
    for sec in sections:
        desc = str(sec.get("Description", ""))
        content = str(sec.get("Content", ""))
        lines = content.splitlines()
        if len(lines) > 14:
            content = "\n".join(lines[:14])
        useful.append(f"[{desc}]\n{content}")
    return "\n\n".join(useful)[:max_chars]


def build_task_context(row):
    local_constraint = safe_literal_eval(row.get("local_constraint", "{}"), {})
    if not isinstance(local_constraint, dict):
        local_constraint = {}

    return {
        "org": str(row["org"]),
        "dest": str(row["dest"]),
        "days": int(row["days"]),
        "visiting_city_number": int(row["visiting_city_number"]),
        "date": str(row["date"]),
        "people_number": int(row["people_number"]),
        "budget": str(row["budget"]),
        "query": str(row["query"]),
        "level": str(row["level"]),
        "local_constraint": local_constraint,
        "candidate_cities_from_reference": extract_candidate_cities(row["reference_information"]),
        "reference_information_compact": compact_reference_information(row["reference_information"]),
    }


def load_generated_text(output_dir, idx):
    path = Path(output_dir) / f"generated_plan_{idx}.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, list) or not payload:
        return ""
    record = payload[-1]
    if not isinstance(record, dict):
        return ""
    for key, value in record.items():
        if key.endswith("_results") and not key.endswith("_parsed_results"):
            return str(value)
    return ""


def normalize_plan_list(plan, expected_days):
    if isinstance(plan, dict) and "plan" in plan:
        plan = plan["plan"]
    if not isinstance(plan, list) or not plan:
        return []

    normalized = []
    for idx, raw_day in enumerate(plan[:expected_days], start=1):
        if not isinstance(raw_day, dict):
            return []
        day = {}
        for key in REQUIRED_PLAN_KEYS:
            if key == "days":
                try:
                    day[key] = int(raw_day.get("days", idx))
                except Exception:
                    day[key] = idx
            else:
                value = raw_day.get(key, "-")
                day[key] = "-" if value in [None, ""] else str(value).strip()
        normalized.append(day)
    return normalized


def render_plan_text(plan):
    lines = ["Travel Plan:"]
    if not plan:
        lines.extend(["", "No valid plan available."])
        return "\n".join(lines)
    for day in plan:
        lines.append("")
        lines.append(f"Day {day['days']}:")
        lines.append(f"Current City: {day['current_city']}")
        lines.append(f"Transportation: {day['transportation']}")
        lines.append(f"Breakfast: {day['breakfast']}")
        lines.append(f"Attraction: {day['attraction']}")
        lines.append(f"Lunch: {day['lunch']}")
        lines.append(f"Dinner: {day['dinner']}")
        lines.append(f"Accommodation: {day['accommodation']}")
    return "\n".join(lines)


def clone_plan(plan):
    return json.loads(json.dumps(plan, ensure_ascii=False))


def iter_runs(values):
    runs = []
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[start]:
            end += 1
        runs.append((start, end, values[start], end - start + 1))
        start = end + 1
    return runs


def local_constraint_dict(task):
    value = task.get("local_constraint", {}) or {}
    return value if isinstance(value, dict) else {}


def row_matches_room_type(row, room_type):
    room_type = str(room_type or "").strip()
    if not room_type:
        return True
    value = str(row.get("room type", "") or "")
    if room_type == "shared room":
        return value == "Shared room"
    if room_type == "private room":
        return value == "Private room"
    if room_type == "entire room":
        return value == "Entire home/apt"
    if room_type == "not shared room":
        return value != "Shared room"
    return True


def row_matches_house_rule(row, house_rule):
    house_rule = str(house_rule or "").strip()
    if not house_rule:
        return True
    rules = str(row.get("house_rules", "") or "")
    if house_rule == "smoking":
        return "No smoking" not in rules
    if house_rule == "parties":
        return "No parties" not in rules
    if house_rule == "children under 10":
        return "No children under 10" not in rules
    if house_rule == "visitors":
        return "No visitors" not in rules
    if house_rule == "pets":
        return "No pets" not in rules
    return True


def accommodation_cost(row, people):
    occupancy = max(1.0, float(row.get("maximum occupancy", 1) or 1))
    return float(row.get("price", 0) or 0) * math.ceil(int(people or 1) / occupancy)


def format_accommodation(row):
    return f"{row['NAME']}, {row['city']}"


def candidate_accommodations(task, city, nights, current_value=""):
    constraints = local_constraint_dict(task)
    people = int(task.get("people_number", 1) or 1)
    rows = _load_accommodations()
    rows = rows[rows["city"].astype(str).str.casefold() == str(city).casefold()]
    rows = rows[rows["maximum occupancy"].astype(float) >= people]
    rows = rows[rows["minimum nights"].astype(float) <= nights]

    if constraints.get("room type"):
        rows = rows[rows.apply(lambda row: row_matches_room_type(row, constraints.get("room type")), axis=1)]
    if constraints.get("house rule"):
        rows = rows[rows.apply(lambda row: row_matches_house_rule(row, constraints.get("house rule")), axis=1)]

    if len(rows) == 0:
        return []

    rows = rows.assign(
        formatted=rows.apply(format_accommodation, axis=1),
        estimated_cost=rows.apply(lambda row: accommodation_cost(row, people), axis=1),
    )
    rows = rows.sort_values(by=["estimated_cost", "minimum nights", "price", "NAME"])
    formatted_rows = rows.to_dict("records")
    preferred = [row for row in formatted_rows if row["formatted"] != current_value]
    return preferred or formatted_rows


def cheapest_restaurants_for_city(city, cuisines=None, exclude=None):
    rows = _load_restaurants()
    rows = rows[rows["City"].astype(str).str.casefold() == str(city).casefold()]
    if cuisines:
        cuisines = [str(c).strip() for c in cuisines if str(c).strip()]
        rows = rows[
            rows["Cuisines"].astype(str).apply(
                lambda value: all(cuisine in str(value) for cuisine in cuisines)
            )
        ]
    if exclude:
        exclude = {str(item) for item in exclude}
        rows = rows[~rows.apply(lambda row: f"{row['Name']}, {row['City']}" in exclude, axis=1)]
    if len(rows) == 0:
        return []
    rows = rows.assign(formatted=rows.apply(lambda row: f"{row['Name']}, {row['City']}", axis=1))
    rows = rows.sort_values(by=["Average Cost", "Aggregate Rating", "Name"], ascending=[True, False, True])
    return rows.to_dict("records")


def repair_accommodations(task, plan):
    repaired = clone_plan(plan)
    change_log = []
    people = int(task.get("people_number", 1) or 1)

    accommodations = [str(day.get("accommodation", "") or "").strip() for day in repaired]
    for start, end, accommodation, nights in iter_runs(accommodations):
        if not accommodation or accommodation == "-":
            continue
        _, city = get_valid_name_city(accommodation)
        lookup = _lookup_accommodation(accommodation)
        current_row = lookup.iloc[0] if len(lookup) > 0 else None

        current_valid = current_row is not None
        if current_valid:
            current_valid = (
                float(current_row["minimum nights"]) <= nights
                and float(current_row["maximum occupancy"]) >= people
                and row_matches_room_type(current_row, local_constraint_dict(task).get("room type"))
                and row_matches_house_rule(current_row, local_constraint_dict(task).get("house rule"))
            )

        if current_valid:
            continue

        options = candidate_accommodations(task, city, nights, current_value=accommodation)
        if not options:
            continue

        replacement = options[0]["formatted"]
        for idx in range(start, end + 1):
            repaired[idx]["accommodation"] = replacement
        change_log.append(
            {
                "type": "accommodation_patch",
                "days": [day["days"] for day in repaired[start : end + 1]],
                "from": accommodation,
                "to": replacement,
            }
        )
    return repaired, change_log


def repair_missing_cuisines(task, plan):
    repaired = clone_plan(plan)
    change_log = []
    constraints = local_constraint_dict(task)
    required = list(constraints.get("cuisine") or [])
    if not required:
        return repaired, change_log

    misses = [item.split(":", 1)[1] for item in _check_local_constraints(task, repaired) if item.startswith("cuisine:")]
    used_restaurants = {
        str(day.get(key, "") or "").strip()
        for day in repaired
        for key in ["breakfast", "lunch", "dinner"]
        if str(day.get(key, "") or "").strip() not in {"", "-"}
    }

    for cuisine in misses:
        placed = False
        for day in repaired:
            current_city = str(day.get("current_city", "") or "")
            city_candidates = [part.strip() for part in re.split(r"from | to ", current_city) if part.strip()]
            city = city_candidates[-1] if city_candidates else ""
            if not city or city == task.get("org"):
                continue
            options = cheapest_restaurants_for_city(city, cuisines=[cuisine], exclude=used_restaurants)
            if not options:
                continue
            replacement = options[0]["formatted"]
            old_value = day.get("dinner", "-")
            day["dinner"] = replacement
            used_restaurants.add(replacement)
            change_log.append(
                {
                    "type": "cuisine_patch",
                    "day": day["days"],
                    "field": "dinner",
                    "from": old_value,
                    "to": replacement,
                    "cuisine": cuisine,
                }
            )
            placed = True
            break
        if not placed:
            change_log.append({"type": "cuisine_patch_failed", "cuisine": cuisine})
    return repaired, change_log


def repair_budget(task, plan):
    repaired = clone_plan(plan)
    change_log = []
    budget = _normalize_budget(task.get("budget"))
    if budget is None:
        return repaired, change_log

    current_cost = _estimate_total_cost(task, repaired)
    if current_cost <= budget:
        return repaired, change_log

    people = int(task.get("people_number", 1) or 1)
    constraints = local_constraint_dict(task)
    for start, end, accommodation, nights in iter_runs([str(day.get("accommodation", "") or "").strip() for day in repaired]):
        if current_cost <= budget or not accommodation or accommodation == "-":
            break
        _, city = get_valid_name_city(accommodation)
        current_rows = _lookup_accommodation(accommodation)
        if len(current_rows) == 0:
            continue
        current_row = current_rows.iloc[0]
        current_total = accommodation_cost(current_row, people)
        options = candidate_accommodations(task, city, nights, current_value=accommodation)
        cheaper = [row for row in options if row["estimated_cost"] < current_total]
        if not cheaper:
            continue
        replacement = cheaper[0]["formatted"]
        for idx in range(start, end + 1):
            repaired[idx]["accommodation"] = replacement
        current_cost = _estimate_total_cost(task, repaired)
        change_log.append(
            {
                "type": "budget_accommodation_patch",
                "days": [day["days"] for day in repaired[start : end + 1]],
                "from": accommodation,
                "to": replacement,
            }
        )

    if current_cost > budget:
        meal_fields = ["dinner", "lunch", "breakfast"]
        for day in repaired:
            if current_cost <= budget:
                break
            current_city = str(day.get("current_city", "") or "")
            city_candidates = [part.strip() for part in re.split(r"from | to ", current_city) if part.strip()]
            city = city_candidates[-1] if city_candidates else ""
            if not city:
                continue
            for field in meal_fields:
                current_meal = str(day.get(field, "") or "").strip()
                if not current_meal or current_meal == "-":
                    continue
                cuisines = []
                if constraints.get("cuisine"):
                    misses_after_drop = _check_local_constraints(task, repaired)
                    cuisine_misses = [item for item in misses_after_drop if item.startswith("cuisine:")]
                    if cuisine_misses:
                        continue
                options = cheapest_restaurants_for_city(city)
                if not options:
                    continue
                replacement = options[0]["formatted"]
                if replacement == current_meal:
                    continue
                day[field] = replacement
                new_cost = _estimate_total_cost(task, repaired)
                if new_cost < current_cost:
                    change_log.append(
                        {
                            "type": "budget_meal_patch",
                            "day": day["days"],
                            "field": field,
                            "from": current_meal,
                            "to": replacement,
                        }
                    )
                    current_cost = new_cost
                    break
                day[field] = current_meal

    return repaired, change_log


def deterministic_repair(task, candidate):
    original_plan = candidate["plan"]
    working_plan = clone_plan(original_plan)
    repair_log = []

    for patcher in [repair_accommodations, repair_missing_cuisines, repair_budget]:
        working_plan, changes = patcher(task, working_plan)
        repair_log.extend(changes)

    original_audit = candidate["audit"]
    repaired_audit = audit_plan(task, working_plan)

    improved = candidate_rank_tuple(
        {"audit": repaired_audit, "seed_source": candidate.get("seed_source"), "stage": "revised"}
    ) > candidate_rank_tuple(candidate)

    return {
        "plan": working_plan,
        "audit": repaired_audit,
        "change_log": repair_log,
        "improved": improved,
        "original_audit": original_audit,
    }


def repair_priority(candidate):
    audit = candidate["audit"]
    invalid_total = int(audit.get("invalid_entity_count", 0)) + int(audit.get("invalid_transportation_count", 0))
    if candidate.get("seed_source") == "scratch" and (
        int(audit.get("score", 0)) < REPAIR_SCORE_FLOOR
        or int(audit.get("fatal_count", 0)) >= HOPELESS_FATAL_THRESHOLD
        or invalid_total >= HOPELESS_FATAL_THRESHOLD
    ):
        return None

    priority = int(audit.get("score", 0))
    priority -= 14 * int(audit.get("fatal_count", 0))
    priority -= 10 * invalid_total
    if audit.get("min_nights_issues"):
        priority += 20
    if audit.get("local_constraint_misses"):
        priority += 18
    if candidate.get("seed_source") in {"direct", "program"}:
        priority += 8
    if audit.get("likely_pass"):
        priority -= 25
    return priority


def is_transport_conflict_only(candidate):
    fatal_issues = list(candidate["audit"].get("fatal_issues", []))
    return fatal_issues == ["mixed transportation modes conflict"]


def should_generate_scratch(task, seed_candidates):
    best_seed = rank_candidates(seed_candidates)[0]
    best_audit = best_seed["audit"]

    if best_audit.get("likely_pass") and int(best_audit.get("score", 0)) >= 90:
        return False

    deterministic_passes = 0
    for candidate in seed_candidates:
        repaired = deterministic_repair(task, candidate)
        if repaired["improved"] and repaired["audit"].get("likely_pass"):
            deterministic_passes += 1
    if deterministic_passes > 0:
        return False

    if best_audit.get("likely_pass") and int(best_audit.get("score", 0)) >= 84:
        return False

    if any(is_transport_conflict_only(candidate) for candidate in seed_candidates):
        return False

    return True


def deterministic_reflection(candidate):
    audit = candidate["audit"]
    instructions = []

    for issue in audit.get("min_nights_issues", []):
        instructions.append(
            f"Replace accommodation causing minimum-night failure: {issue}."
        )
    for miss in audit.get("local_constraint_misses", []):
        instructions.append(f"Resolve local constraint miss: {miss}.")
    for issue in audit.get("fatal_issues", []):
        if "estimated cost" in issue:
            instructions.append("Reduce total cost by replacing expensive meals or accommodations with cheaper valid options.")
        elif "mixed transportation modes conflict" in issue:
            instructions.append("Use a single transportation mode consistently across the trip.")
        elif "invalid transportation" in issue:
            instructions.append("Replace invalid transportation with a reference-supported route between the same cities.")
        elif "expected visiting_city_number" in issue:
            instructions.append("Adjust the itinerary so that the number of non-origin cities matches visiting_city_number exactly.")
        elif "city sequence" in issue or "trip should start" in issue or "closed loop" in issue:
            instructions.append("Repair the route so that it starts at the origin, visits contiguous cities, and returns to the origin.")

    unique = []
    seen = set()
    for item in instructions:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return {
        "candidate_id": candidate["candidate_id"],
        "instructions": unique[:6],
    }


def candidate_summary(plan):
    restaurants = []
    attractions = []
    transport_types = set()
    for day in plan:
        for key in ["breakfast", "lunch", "dinner"]:
            value = day.get(key, "-")
            if value and value != "-":
                restaurants.append(value)
        attraction = day.get("attraction", "-")
        if attraction and attraction != "-":
            parts = [x.strip() for x in attraction.split(";") if x.strip()]
            attractions.extend(parts)
        transportation = str(day.get("transportation", "")).lower()
        if "flight" in transportation:
            transport_types.add("flight")
        if "self-driving" in transportation:
            transport_types.add("self-driving")
        if "taxi" in transportation:
            transport_types.add("taxi")

    return {
        "days": len(plan),
        "unique_restaurants": len(set(restaurants)),
        "unique_attractions": len(set(attractions)),
        "restaurant_duplicates": max(0, len(restaurants) - len(set(restaurants))),
        "attraction_duplicates": max(0, len(attractions) - len(set(attractions))),
        "transport_types": sorted(transport_types),
        "last_day_accommodation": plan[-1].get("accommodation", "-") if plan else "-",
    }


def build_candidate(name, seed_source, stage, round_idx, task, plan, text, provenance=None):
    return {
        "candidate_id": name,
        "seed_source": seed_source,
        "stage": stage,
        "round": round_idx,
        "plan": plan,
        "text": text,
        "summary": candidate_summary(plan),
        "audit": audit_plan(task, plan),
        "provenance": provenance or {},
    }


def candidate_rank_tuple(candidate):
    audit = candidate["audit"]
    return (
        1 if audit.get("likely_pass") else 0,
        -int(audit.get("fatal_count", 0)),
        int(audit.get("score", 0)),
        -int(audit.get("warning_count", 0)),
        1 if candidate.get("seed_source") in {"direct", "program"} else 0,
        1 if candidate.get("stage") == "revised" else 0,
    )


def rank_candidates(candidates):
    return sorted(candidates, key=candidate_rank_tuple, reverse=True)


def get_base_seed_best(candidates):
    seed_candidates = [c for c in candidates if c.get("seed_source") in {"direct", "program"} and c.get("stage") == "seed"]
    ranked = rank_candidates(seed_candidates)
    if not ranked:
        ranked = rank_candidates(candidates)
    return ranked[0]


def should_promote_candidate(candidate, base_best):
    if candidate["candidate_id"] == base_best["candidate_id"]:
        return True

    candidate_audit = candidate["audit"]
    base_audit = base_best["audit"]
    candidate_likely = bool(candidate_audit.get("likely_pass"))
    base_likely = bool(base_audit.get("likely_pass"))
    candidate_score = int(candidate_audit.get("score", 0))
    base_score = int(base_audit.get("score", 0))
    candidate_fatal = int(candidate_audit.get("fatal_count", 0))
    base_fatal = int(base_audit.get("fatal_count", 0))

    if candidate_likely and not base_likely:
        return True
    if candidate_likely and base_likely and candidate_score >= base_score:
        return True
    if candidate_score >= base_score + PROMOTION_GAP and candidate_fatal < base_fatal:
        return True
    return False


def load_seed_candidates(idx, task, direct_submission_rows, program_submission_rows):
    expected_days = int(task["days"])
    direct_plan = normalize_plan_list(direct_submission_rows[idx - 1]["plan"], expected_days)
    program_plan = normalize_plan_list(program_submission_rows[idx - 1]["plan"], expected_days)
    if not direct_plan and not program_plan:
        raise RuntimeError(f"Missing valid direct/program seed candidates at idx={idx}")

    candidates = []
    if direct_plan:
        candidates.append(
            build_candidate(
                "direct_seed",
                "direct",
                "seed",
                0,
                task,
                direct_plan,
                load_generated_text(DIRECT_OUTPUT_DIR, idx) or render_plan_text(direct_plan),
                provenance={"source_submission": str(DIRECT_SUBMISSION_FILE)},
            )
        )
    if program_plan and CANDIDATES >= 2:
        candidates.append(
            build_candidate(
                "program_seed",
                "program",
                "seed",
                0,
                task,
                program_plan,
                load_generated_text(PROGRAM_OUTPUT_DIR, idx) or render_plan_text(program_plan),
                provenance={"source_submission": str(PROGRAM_SUBMISSION_FILE)},
            )
        )
    return candidates


def planner_prompt(task):
    system_prompt = (
        "You are the planner agent in a multi-agent TravelPlanner system. "
        "Generate a complete itinerary in structured JSON. "
        "Use only entities from the provided reference information. "
        "Do not output markdown. Return one JSON object."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "instructions": [
                "Output a complete JSON itinerary with exactly the required number of days.",
                "Visit exactly visiting_city_number concrete cities from candidate_cities_from_reference.",
                "If dest is a state or region, do not use the state name itself as a city.",
                "Plan route first, then transportation, then meals, attractions, and accommodation.",
                "Avoid repeated restaurants and attractions whenever possible.",
                "Respect budget and hard constraints.",
                "If the traveler returns to origin on the last day, accommodation may be '-'.",
            ],
            "output_schema": {
                "plan": [
                    {
                        "days": 1,
                        "current_city": "string",
                        "transportation": "string",
                        "breakfast": "string",
                        "attraction": "string",
                        "lunch": "string",
                        "dinner": "string",
                        "accommodation": "string",
                    }
                ],
                "route_summary": "short route explanation",
            },
            "reference_information_compact": task["reference_information_compact"],
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def verifier_prompt(task, candidate):
    system_prompt = (
        "You are the verifier agent in a multi-agent TravelPlanner system. "
        "Deterministic audit findings are authoritative. "
        "Use them to produce precise, field-level repair guidance. "
        "Return one JSON object."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "candidate_id": candidate["candidate_id"],
            "candidate_plan": candidate["plan"],
            "candidate_text": candidate["text"],
            "deterministic_audit": candidate["audit"],
            "instructions": [
                "Do not override the deterministic audit on entity validity, city count, route closure, transportation validity, minimum nights, budget, or hard constraints.",
                "Identify only repairable issues and describe concrete field or route changes.",
                "If the candidate is already strong, say so briefly.",
                "Last-day accommodation '-' is valid when the trip returns to the origin city.",
            ],
            "output_schema": {
                "repairable": True,
                "confidence": "low | medium | high",
                "summary": "short judgment",
                "priority_issues": [
                    {
                        "scope": "route | day-field",
                        "day": 1,
                        "field": "transportation or accommodation or null",
                        "problem": "specific issue",
                        "instruction": "specific repair instruction",
                    }
                ],
                "tie_break_notes": ["optional higher-level observations"],
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def reflector_prompt(task, judged_candidates):
    system_prompt = (
        "You are the reflector agent in a multi-agent TravelPlanner system. "
        "Summarize only actionable repair instructions. "
        "Return one JSON object."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "judged_candidates": judged_candidates,
            "instructions": [
                "Produce concise route-level and field-level repair memory.",
                "Avoid generic advice.",
                "Candidate memories should be specific enough for a reviser agent to patch the plan directly.",
            ],
            "output_schema": {
                "global_lessons": ["list of shared lessons"],
                "candidate_memories": [
                    {
                        "candidate_id": "string",
                        "instructions": ["field-level or route-level fixes"],
                    }
                ],
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def reviser_prompt(task, candidate, memory):
    system_prompt = (
        "You are the reviser agent in a multi-agent TravelPlanner system. "
        "Patch only failing fields or failing route segments. "
        "Do not regenerate a completely new plan unless the current route is irrecoverable. "
        "Return one JSON object."
    )

    repair_mode = {
        "strict_route_patch_only": STRICT_ROUTE_PATCH_ONLY,
        "no_free_regeneration": NO_FREE_REGEN,
    }
    user_prompt = json.dumps(
        {
            "task": task,
            "candidate_id": candidate["candidate_id"],
            "current_plan": candidate["plan"],
            "candidate_text": candidate["text"],
            "deterministic_audit": candidate["audit"],
            "verifier_feedback": candidate.get("llm_verifier", {}),
            "reflection_memory": memory,
            "repair_mode": repair_mode,
            "instructions": [
                "Preserve all valid fields unless there is a concrete reason to modify them.",
                "If strict_route_patch_only is true, only fix route and transportation fields in this revision.",
                "If no_free_regeneration is true, do not replace already-valid meals/attractions/accommodation without a verified failure.",
                "Use only entities consistent with the provided reference information.",
                "If the last day returns to origin, accommodation may be '-'.",
            ],
            "output_schema": {
                "plan": [
                    {
                        "days": 1,
                        "current_city": "string",
                        "transportation": "string",
                        "breakfast": "string",
                        "attraction": "string",
                        "lunch": "string",
                        "dinner": "string",
                        "accommodation": "string",
                    }
                ],
                "change_summary": ["list of changes"],
            },
            "reference_information_compact": task["reference_information_compact"],
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def selector_prompt(task, candidates):
    system_prompt = (
        "You are the selector agent in a multi-agent TravelPlanner system. "
        "Use deterministic audit as the primary signal and break close ties only. "
        "Return one JSON object."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "seed_source": c["seed_source"],
                    "stage": c["stage"],
                    "audit": c["audit"],
                    "summary": c["summary"],
                    "text": c["text"],
                }
                for c in candidates
            ],
            "instructions": [
                "Only break close ties; do not pick a candidate that is clearly worse under deterministic audit.",
                "Prefer complete route consistency and hard-constraint satisfaction over stylistic differences.",
            ],
            "output_schema": {
                "preferred_candidate_id": "candidate id string",
                "summary": "short explanation",
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def save_debug(debug_state, idx):
    atomic_write_text(DEBUG_DIR / f"debug_{idx}.json",
        json.dumps(debug_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_debug(idx):
    path = DEBUG_DIR / f"debug_{idx}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def build_fallback_scratch_candidate(task, seed_candidates):
    base = rank_candidates(seed_candidates)[0]
    return build_candidate(
        "scratch_planner_r1",
        "scratch",
        "round1",
        1,
        task,
        list(base["plan"]),
        base["text"],
        provenance={"fallback_from": base["candidate_id"], "planner_status": "fallback_seed_copy"},
    )


def generate_scratch_candidate(task, idx, seed_candidates):
    if not ALLOW_LLM:
        return build_fallback_scratch_candidate(task, seed_candidates)
    sp, up = planner_prompt(task)
    try:
        result = call_deepseek_json(
            sp,
            up,
            max_tokens=4096,
            temperature=SELF_GEN_TEMPERATURE,
            cache_tag=f"planner_scratch_idx{idx}",
        )
    except ModelRequestError:
        raise
    except Exception as exc:
        candidate = build_fallback_scratch_candidate(task, seed_candidates)
        candidate["provenance"]["planner_error"] = str(exc)
        return candidate

    plan = normalize_plan_list(result, int(task["days"]))
    if not plan:
        candidate = build_fallback_scratch_candidate(task, seed_candidates)
        candidate["provenance"]["planner_error"] = "empty_or_invalid_plan"
        return candidate

    return build_candidate(
        "scratch_planner_r1",
        "scratch",
        "round1",
        1,
        task,
        plan,
        render_plan_text(plan),
        provenance={"planner_output": result},
    )


def select_candidates_for_llm_verification(candidates):
    ranked = rank_candidates(candidates)
    selected = []
    for candidate in ranked:
        if len(selected) >= max(1, min(REPAIR_TARGETS, len(ranked))):
            break
        if repair_priority(candidate) is None:
            continue
        if candidate["seed_source"] == "scratch" or not candidate["audit"].get("likely_pass") or candidate["audit"].get("score", 0) >= 90:
            selected.append(candidate["candidate_id"])

    if not selected and ranked:
        selected.append(ranked[0]["candidate_id"])
    return set(selected)


def verify_candidates(task, idx, candidates, cache_prefix):
    selected_ids = select_candidates_for_llm_verification(candidates)
    verified = []
    for candidate in candidates:
        candidate = json.loads(json.dumps(candidate, ensure_ascii=False))
        if ALLOW_LLM and candidate["candidate_id"] in selected_ids:
            sp, up = verifier_prompt(task, candidate)
            try:
                llm_verifier = call_deepseek_json(
                    sp,
                    up,
                    max_tokens=2048,
                    temperature=0,
                    cache_tag=f"{cache_prefix}_verify_{candidate['candidate_id']}_idx{idx}",
                )
            except ModelRequestError:
                raise
            except Exception as exc:
                llm_verifier = {"summary": "llm_verifier_failed", "error": str(exc), "priority_issues": []}
        else:
            llm_verifier = {
                "summary": "deterministic_audit_only",
                "priority_issues": [],
            }
        candidate["llm_verifier"] = llm_verifier if isinstance(llm_verifier, dict) else {}
        verified.append(candidate)
    return verified


def choose_repair_targets(candidates):
    ranked = rank_candidates(candidates)
    scored = []
    for candidate in ranked:
        priority = repair_priority(candidate)
        if priority is None:
            continue
        if candidate["audit"].get("likely_pass") and candidate["seed_source"] in {"direct", "program"}:
            continue
        scored.append((priority, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    targets = [candidate for _, candidate in scored[: max(1, min(REPAIR_TARGETS, len(scored) or len(ranked)))]]
    if not targets and ranked:
        targets = [candidate for candidate in ranked if repair_priority(candidate) is not None][:1]
    return targets


def build_candidate_memory(reflection_payload, candidate_id):
    if not isinstance(reflection_payload, dict):
        return []
    instructions = []
    for item in reflection_payload.get("candidate_memories", []):
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
            if isinstance(item.get("instructions"), list):
                instructions.extend(str(x) for x in item["instructions"])
    if isinstance(reflection_payload.get("global_lessons"), list):
        instructions.extend(str(x) for x in reflection_payload["global_lessons"])
    return instructions


def reflect_on_candidates(task, idx, candidates):
    deterministic_memories = [deterministic_reflection(candidate) for candidate in candidates]
    if deterministic_memories and all(memory["instructions"] for memory in deterministic_memories):
        return {
            "global_lessons": [
                "Prefer verifier-identified fixes over unconstrained rewrites.",
                "Preserve route structure unless the verifier reports a route violation.",
            ],
            "candidate_memories": deterministic_memories,
            "reflection_mode": "deterministic",
        }

    if not ALLOW_LLM:
        return {"global_lessons": [], "candidate_memories": []}
    judged_payload = [
        {
            "candidate_id": c["candidate_id"],
            "seed_source": c["seed_source"],
            "stage": c["stage"],
            "audit": c["audit"],
            "llm_verifier": c.get("llm_verifier", {}),
            "summary": c["summary"],
        }
        for c in candidates
    ]
    sp, up = reflector_prompt(task, judged_payload)
    try:
        reflection = call_deepseek_json(
            sp,
            up,
            max_tokens=2048,
            temperature=0,
            cache_tag=f"reflect_idx{idx}_r2",
        )
    except ModelRequestError:
        raise
    except Exception as exc:
        reflection = {"global_lessons": [f"reflection_failed: {exc}"], "candidate_memories": []}
    return reflection if isinstance(reflection, dict) else {"global_lessons": [], "candidate_memories": []}


def revise_candidate(task, idx, candidate, reflection_payload):
    memory = build_candidate_memory(reflection_payload, candidate["candidate_id"])
    deterministic = deterministic_repair(task, candidate)

    if deterministic["improved"] and deterministic["audit"].get("likely_pass"):
        return build_candidate(
            candidate["candidate_id"] + "_revised_r2",
            candidate["seed_source"],
            "revised",
            2,
            task,
            deterministic["plan"],
            render_plan_text(deterministic["plan"]),
            provenance={
                "parent_candidate_id": candidate["candidate_id"],
                "repair_mode": "deterministic_short_circuit",
                "change_log": deterministic["change_log"],
                "reflection_memory": memory,
            },
        )

    if not ALLOW_LLM:
        plan = deterministic["plan"] if deterministic["improved"] else clone_plan(candidate["plan"])
        revised = build_candidate(
            candidate["candidate_id"] + "_revised_r2",
            candidate["seed_source"],
            "revised",
            2,
            task,
            plan,
            render_plan_text(plan),
            provenance={
                "parent_candidate_id": candidate["candidate_id"],
                "reviser_status": "deterministic_only",
                "change_log": deterministic["change_log"],
                "reflection_memory": memory,
            },
        )
        return revised

    reviser_candidate = candidate
    if deterministic["improved"]:
        reviser_candidate = build_candidate(
            candidate["candidate_id"],
            candidate["seed_source"],
            candidate["stage"],
            candidate["round"],
            task,
            deterministic["plan"],
            render_plan_text(deterministic["plan"]),
            provenance={
                **candidate.get("provenance", {}),
                "deterministic_repair_seed": deterministic["change_log"],
            },
        )

    sp, up = reviser_prompt(task, reviser_candidate, memory)
    try:
        result = call_deepseek_json(
            sp,
            up,
            max_tokens=4096,
            temperature=REVISE_TEMPERATURE,
            cache_tag=f"revise_idx{idx}_{candidate['candidate_id']}_r2",
        )
        plan = normalize_plan_list(result, int(task["days"]))
    except ModelRequestError:
        raise
    except Exception as exc:
        result = {"change_summary": [f"reviser_failed: {exc}"]}
        plan = []

    if not plan:
        fallback_plan = deterministic["plan"] if deterministic["improved"] else clone_plan(candidate["plan"])
        return build_candidate(
            candidate["candidate_id"] + "_revised_r2",
            candidate["seed_source"],
            "revised",
            2,
            task,
            fallback_plan,
            render_plan_text(fallback_plan),
            provenance={
                **candidate.get("provenance", {}),
                "reviser_status": "fallback_original_candidate",
                "reviser_output": result,
                "deterministic_changes": deterministic["change_log"],
                "reflection_memory": memory,
            },
        )

    return build_candidate(
        candidate["candidate_id"] + "_revised_r2",
        candidate["seed_source"],
        "revised",
        2,
        task,
        plan,
        render_plan_text(plan),
        provenance={
            "parent_candidate_id": candidate["candidate_id"],
            "reviser_output": result,
            "deterministic_changes": deterministic["change_log"],
            "reflection_memory": memory,
        },
    )


def choose_final_candidate(task, idx, candidates):
    ranked = rank_candidates(candidates)
    base_best = get_base_seed_best(candidates)
    promotable = [candidate for candidate in ranked if should_promote_candidate(candidate, base_best)]
    chosen = promotable[0] if promotable else base_best

    if ALLOW_LLM and len(promotable) >= 2:
        first = promotable[0]["audit"]
        second = promotable[1]["audit"]
        if (
            abs(int(first.get("score", 0)) - int(second.get("score", 0))) <= TIE_BREAK_GAP
            and int(first.get("fatal_count", 0)) == int(second.get("fatal_count", 0))
        ):
            sp, up = selector_prompt(task, promotable[:3])
            try:
                selector = call_deepseek_json(
                    sp,
                    up,
                    max_tokens=1536,
                    temperature=0,
                    cache_tag=f"select_idx{idx}",
                )
            except ModelRequestError:
                raise
            except Exception as exc:
                selector = {"preferred_candidate_id": chosen["candidate_id"], "summary": f"selector_failed: {exc}"}
            if isinstance(selector, dict):
                preferred_id = str(selector.get("preferred_candidate_id", "")).strip()
                for candidate in promotable[:3]:
                    if candidate["candidate_id"] == preferred_id and should_promote_candidate(candidate, base_best):
                        candidate["selector_summary"] = selector.get("summary", "")
                        return candidate, selector
            return chosen, selector

    return chosen, {"preferred_candidate_id": chosen["candidate_id"], "summary": "deterministic_verifier_choice"}


def run_one(row, idx, direct_submission_rows, program_submission_rows):
    task = build_task_context(row)
    existing_debug = load_debug(idx)
    output_path = OUTPUT_DIR / f"generated_plan_{idx}.json"
    if RESUME and existing_debug and existing_debug.get("status") == "completed" and output_path.exists():
        try:
            saved = json.loads(output_path.read_text())[0]
            if valid_plan_container(saved.get(f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results")):
                return None, None, existing_debug, True
        except (ValueError, IndexError, KeyError, TypeError):
            pass

    debug_state = {
        "idx": idx,
        "task": task,
        "strategy": STRATEGY,
        "config": {
            "rounds": ROUNDS,
            "candidates": CANDIDATES,
            "allow_llm": ALLOW_LLM,
            "strict_route_patch_only": STRICT_ROUTE_PATCH_ONLY,
            "no_free_regeneration": NO_FREE_REGEN,
        },
        "stages": [],
        "status": "running",
    }

    seed_candidates = load_seed_candidates(idx, task, direct_submission_rows, program_submission_rows)
    debug_state["stages"].append(
        {
            "stage": "seeder",
            "candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "seed_source": c["seed_source"],
                    "audit": c["audit"],
                    "summary": c["summary"],
                }
                for c in seed_candidates
            ],
        }
    )
    save_debug(debug_state, idx)

    round1_candidates = list(seed_candidates)
    if CANDIDATES >= 3 and should_generate_scratch(task, seed_candidates):
        scratch_candidate = generate_scratch_candidate(task, idx, seed_candidates)
        round1_candidates.append(scratch_candidate)
        debug_state["stages"].append(
            {
                "stage": "planner_round1",
                "candidate": {
                    "candidate_id": scratch_candidate["candidate_id"],
                    "seed_source": scratch_candidate["seed_source"],
                    "audit": scratch_candidate["audit"],
                    "summary": scratch_candidate["summary"],
                    "provenance": scratch_candidate["provenance"],
                },
            }
        )
        save_debug(debug_state, idx)
    elif CANDIDATES >= 3:
        debug_state["stages"].append(
            {
                "stage": "planner_round1",
                "candidate": {
                    "candidate_id": "scratch_planner_r1",
                    "seed_source": "scratch",
                    "audit": None,
                    "summary": None,
                    "provenance": {"planner_status": "skipped_high_confidence_seed_pair"},
                },
            }
        )
        save_debug(debug_state, idx)

    round1_verified = verify_candidates(task, idx, round1_candidates, "r1")
    debug_state["stages"].append(
        {
            "stage": "verifier_round1",
            "candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "stage": c["stage"],
                    "audit": c["audit"],
                    "llm_verifier": c.get("llm_verifier", {}),
                }
                for c in round1_verified
            ],
        }
    )
    save_debug(debug_state, idx)

    all_candidates = list(round1_verified)
    reflection_payload = {"global_lessons": [], "candidate_memories": []}

    if ROUNDS >= 2:
        repair_targets = choose_repair_targets(round1_verified)
        reflection_payload = reflect_on_candidates(task, idx, repair_targets)
        debug_state["stages"].append(
            {
                "stage": "reflector_round2",
                "repair_targets": [c["candidate_id"] for c in repair_targets],
                "reflection": reflection_payload,
            }
        )
        save_debug(debug_state, idx)

        revised_candidates = []
        for candidate in repair_targets:
            revised = revise_candidate(task, idx, candidate, reflection_payload)
            revised_candidates.append(revised)
            debug_state["stages"].append(
                {
                    "stage": "reviser_round2",
                    "candidate_id": revised["candidate_id"],
                    "parent_candidate_id": candidate["candidate_id"],
                    "audit": revised["audit"],
                    "provenance": revised["provenance"],
                }
            )
            save_debug(debug_state, idx)

        revised_verified = verify_candidates(task, idx, revised_candidates, "r2")
        debug_state["stages"].append(
            {
                "stage": "verifier_round2",
                "candidates": [
                    {
                        "candidate_id": c["candidate_id"],
                        "stage": c["stage"],
                        "audit": c["audit"],
                        "llm_verifier": c.get("llm_verifier", {}),
                    }
                    for c in revised_verified
                ],
            }
        )
        save_debug(debug_state, idx)
        all_candidates.extend(revised_verified)

    chosen_candidate, selector_payload = choose_final_candidate(task, idx, all_candidates)
    debug_state["final_choice"] = chosen_candidate["candidate_id"]
    debug_state["final_audit"] = chosen_candidate["audit"]
    debug_state["final_summary"] = chosen_candidate["summary"]
    debug_state["selector"] = selector_payload
    debug_state["candidate_pool"] = [
        {
            "candidate_id": c["candidate_id"],
            "seed_source": c["seed_source"],
            "stage": c["stage"],
            "round": c["round"],
            "audit": c["audit"],
            "summary": c["summary"],
            "provenance": c["provenance"],
        }
        for c in all_candidates
    ]
    debug_state["status"] = "completed"
    save_debug(debug_state, idx)
    return chosen_candidate["text"], chosen_candidate["plan"], debug_state, False


def main():
    df = pd.read_csv(DATA_PATH)
    globals()["DATASET_FINGERPRINT"] = hashlib.sha256(df.to_json().encode()).hexdigest()
    select_rows(df)  # Validate the request before claiming output paths.
    direct_submission_rows = load_jsonl(DIRECT_SUBMISSION_FILE)
    program_submission_rows = load_jsonl(PROGRAM_SUBMISSION_FILE)
    if len(direct_submission_rows) != len(df) or len(program_submission_rows) != len(df):
        raise ValueError("Direct/program seed counts must match the validation dataset.")
    configure_run(globals())
    with exclusive_run(OUT_ROOT):
        _run_dataframe(df, direct_submission_rows, program_submission_rows)


def _run_dataframe(df, direct_submission_rows, program_submission_rows):
    ensure_dirs()

    df_run = select_rows(df)

    key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_results"
    parsed_key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results"
    selection_key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_selection"
    submission_rows = []

    for idx, row in df_run.iterrows():
        number = idx + 1
        try:
            text, plan, payload, skipped = run_one(row, number, direct_submission_rows, program_submission_rows)
        except Exception as exc:
            text, plan, skipped = "", [], False
            payload = {"idx":number,"status":"failed","error_type":type(exc).__name__,"error":str(exc),"final_choice":None}
            save_debug(payload, number)
        debug_payload = payload if payload else load_debug(number)
        if skipped:
            existing = json.loads((OUTPUT_DIR / f"generated_plan_{number}.json").read_text(encoding="utf-8"))[0]
            submission_rows.append(
                {
                    "idx": number,
                    "query": str(row.get("query", "")),
                    "plan": existing.get(parsed_key, []),
                }
            )
            print(f"skipped {number}", flush=True)
            continue

        out_path = OUTPUT_DIR / f"generated_plan_{number}.json"
        atomic_write_text(out_path,
            json.dumps(
                [
                    {
                        key: text,
                        parsed_key: plan,
                        selection_key: debug_payload["final_choice"],
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        submission_rows.append(
            {
                "idx": number,
                "query": str(row.get("query", "")),
                "plan": plan,
            }
        )
        print(f"generated {number}", flush=True)

    atomic_write_text(SUBMISSION_FILE,
        "\n".join(json.dumps(item, ensure_ascii=False) for item in submission_rows) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(submission_rows)} plans into {OUTPUT_DIR}", flush=True)
    print(f"submission saved to {SUBMISSION_FILE}", flush=True)
    failed = []
    for row in submission_rows:
        checkpoint = DEBUG_DIR / f"debug_{row['idx']}.json"
        if checkpoint.exists() and json.loads(checkpoint.read_text()).get("status") == "failed":
            failed.append(row['idx'])
    atomic_write_text(OUT_ROOT / "status.json", json.dumps({"status":"incomplete" if failed else "completed","failed_ids":failed}))
    if failed:
        raise SystemExit(f"Failed cases remain in submission: {failed}; inspect debug records and resume")


if __name__ == "__main__":
    main()
