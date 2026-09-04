import ast
import difflib
import hashlib
import json
import os
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

import pandas as pd
from utils.llm_client import complete, ModelRequestError
from utils.run_state import configure_run, atomic_write_text, exclusive_run
from utils.runner_inputs import select_rows, valid_plan_container, load_seed_submission as load_jsonl
from utils.paths import DATABASE, seed_path

from utils.plan_audit import (
    _load_accommodations,
    _load_attractions,
    _load_restaurants,
    audit_plan,
    get_valid_name_city,
)


MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
STRATEGY = os.getenv("STRATEGY", "cc_mar_r3")
MODE = "sole-planning"

DATA_PATH = DATABASE / "validation.csv"
DIRECT_SUBMISSION_FILE = seed_path("direct")
PROGRAM_SUBMISSION_FILE = seed_path("program")

OUT_ROOT = Path(os.getenv("OUT_ROOT", f"outputs_{STRATEGY}"))
OUTPUT_DIR = OUT_ROOT / "validation"
DEBUG_DIR = OUT_ROOT / "debug"
CACHE_DIR = OUT_ROOT / "llm_cache"
SUBMISSION_FILE = Path("evaluation") / f"validation_{MODEL_NAME}_{STRATEGY}_{MODE}_submission.jsonl"

ROUNDS = int(os.getenv("ROUNDS", "3"))
RESUME = os.getenv("RESUME", "1") == "1"
DISABLE_CRITIC = os.getenv("DISABLE_CRITIC", "0") == "1"
DISABLE_BOARD = os.getenv("DISABLE_BOARD", "0") == "1"
GENERIC_AGENTS = os.getenv("GENERIC_AGENTS", "0") == "1"
SINGLE_AGENT = os.getenv("SINGLE_AGENT", "0") == "1"
MAX_REFERENCE_CHARS = int(os.getenv("MAX_REFERENCE_CHARS", "14000"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

REQUIRED_PLAN_KEYS = [
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
]

ROLE_SPECS = {
    "route_agent": {
        "contracts": ["route", "day_count", "city_count", "transportation"],
        "responsibility": "route closure, city sequence, visiting city count, and transportation continuity",
    },
    "entity_agent": {
        "contracts": ["entity", "sandbox", "current_city", "diversity"],
        "responsibility": "restaurant, attraction, accommodation, sandbox, current-city alignment, and duplicate entities",
    },
    "lodging_agent": {
        "contracts": ["accommodation", "minimum_nights", "room_rule", "room_type"],
        "responsibility": "accommodation legality, minimum nights, room type, house rule, and final-day lodging",
    },
    "budget_agent": {
        "contracts": ["budget", "cost", "cuisine"],
        "responsibility": "budget reduction, meal cost, lodging cost, and required cuisine coverage",
    },
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FILE.parent.mkdir(parents=True, exist_ok=True)


def safe_literal_eval(value, default):
    try:
        if pd.isna(value):
            return default
        return ast.literal_eval(str(value))
    except Exception:
        return default


def compact_reference_information(ref, max_chars=MAX_REFERENCE_CHARS):
    sections = safe_literal_eval(ref, [])
    if not isinstance(sections, list):
        return str(ref)[:max_chars]
    chunks = []
    for sec in sections:
        desc = str(sec.get("Description", ""))
        content = str(sec.get("Content", ""))
        lines = content.splitlines()
        if len(lines) > 20:
            content = "\n".join(lines[:20])
        chunks.append(f"[{desc}]\n{content}")
    return "\n\n".join(chunks)[:max_chars]


def extract_candidate_cities(ref):
    sections = safe_literal_eval(ref, [])
    cities = []
    if not isinstance(sections, list):
        return cities
    for sec in sections:
        desc = str(sec.get("Description", ""))
        for prefix in ["Attractions in ", "Restaurants in ", "Accommodations in "]:
            if desc.startswith(prefix):
                city = desc[len(prefix) :].strip()
                if city and city not in cities:
                    cities.append(city)
    return cities[:40]


def build_task(row):
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
        "local_constraint": local_constraint,
        "query": str(row["query"]),
        "level": str(row["level"]),
        "candidate_cities_from_reference": extract_candidate_cities(row["reference_information"]),
        "reference_information_compact": compact_reference_information(row["reference_information"]),
    }


def parse_json_payload(text):
    text = str(text or "").strip().replace("```json", "").replace("```", "").strip()
    candidates = []
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for raw in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return None


def extract_json_array_after_key(text, key):
    marker = f'"{key}"'
    start_key = str(text or "").find(marker)
    if start_key == -1:
        return None
    start = str(text).find("[", start_key)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : pos + 1])
                except Exception:
                    return None
    return None


def salvage_patch_payload(raw, role_id):
    patches = extract_json_array_after_key(raw, "patches")
    if isinstance(patches, list) and patches:
        return {
            "agent_id": role_id,
            "repairable": True,
            "patches": patches,
            "revised_plan": [],
            "expected_effect": {"decrease": [], "risk": ["json_salvage"]},
            "rationale": "Recovered field-level patches from truncated or malformed JSON.",
            "parse_recovery": "patches_salvaged_from_raw_text",
        }
    return {"repairable": False, "raw_parse_failed": raw}


def normalize_plan_list(payload, expected_days):
    if isinstance(payload, dict):
        payload = payload.get("plan") or payload.get("revised_plan") or payload.get("itinerary") or payload.get("days")
    if not isinstance(payload, list):
        return []
    normalized = []
    for idx, raw_day in enumerate(payload[:expected_days], start=1):
        if not isinstance(raw_day, dict):
            return []
        day = {}
        for key in REQUIRED_PLAN_KEYS:
            if key == "days":
                try:
                    day[key] = int(raw_day.get("days", raw_day.get("day", idx)))
                except Exception:
                    day[key] = idx
            else:
                value = raw_day.get(key, "-")
                day[key] = "-" if value in [None, ""] else str(value).strip()
        normalized.append(day)
    if len(normalized) != expected_days:
        return []
    return normalized


def render_plan_text(plan):
    lines = ["Travel Plan:"]
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


def chat_completions_url(api_base):
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def call_llm(system_prompt, user_prompt, max_tokens=4096, temperature=0.0, cache_tag="default"):
    return complete(system_prompt, user_prompt, cache_dir=CACHE_DIR, model=MODEL_NAME,
                    max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag, json_mode=False)


def violation_vector(audit):
    checks = audit.get("checks", {})
    budget_bad = 0 if checks.get("budget_ok", True) else 1
    return [
        int(audit.get("fatal_count", 0)),
        int(audit.get("invalid_entity_count", 0)),
        int(audit.get("invalid_transportation_count", 0)),
        len(audit.get("min_nights_issues", []) or []),
        len(audit.get("local_constraint_misses", []) or []),
        budget_bad,
        int(audit.get("warning_count", 0)),
        -int(audit.get("score", 0)),
    ]


def vector_improves(after, before):
    return tuple(after) < tuple(before)


def protected_regression(before_audit, after_audit):
    before_checks = before_audit.get("checks", {})
    after_checks = after_audit.get("checks", {})
    protected = [
        "day_count_ok",
        "first_city_ok",
        "closed_loop_ok",
        "city_sequence_ok",
        "visiting_city_number_ok",
        "budget_ok",
    ]
    for key in protected:
        if before_checks.get(key, True) and not after_checks.get(key, True):
            return f"protected_check_regressed:{key}"
    if not before_checks.get("transportation_conflict", False) and after_checks.get("transportation_conflict", False):
        return "protected_check_regressed:transportation_conflict"
    if int(after_audit.get("fatal_count", 0)) > int(before_audit.get("fatal_count", 0)):
        return "fatal_count_increased"
    for key, passed in before_audit.get("strict_checks", {}).items():
        if passed and not after_audit.get("strict_checks", {}).get(key, False):
            return f"protected_check_regressed:{key}"
    for key in ["invalid_entity_count", "invalid_transportation_count"]:
        if after_audit.get(key, 0) > before_audit.get(key, 0):
            return f"{key}_increased"
    return ""


def contract_board(audit):
    checks = audit.get("checks", {})
    violations = []
    min_night_issue_set = set(audit.get("min_nights_issues", []) or [])
    local_issue_set = set(audit.get("local_constraint_misses", []) or [])
    for issue in audit.get("min_nights_issues", []) or []:
        violations.append({"contract": "accommodation", "issue": issue})
    for issue in audit.get("local_constraint_misses", []) or []:
        text = str(issue).lower()
        contract = "budget" if "budget" in text or "cuisine" in text else "entity"
        violations.append({"contract": contract, "issue": issue})
    for key, ok in checks.items():
        if key == "transportation_conflict":
            if ok:
                violations.append({"contract": "transportation", "issue": key})
        elif ok is False:
            violations.append({"contract": "route" if "city" in key or "loop" in key or "day" in key else key, "issue": key})
    for issue in audit.get("fatal_issues", []) or []:
        if issue in min_night_issue_set or issue in local_issue_set:
            continue
        text = str(issue).lower()
        if "minimum" in text or "accommodation" in text or "room" in text or ("requires" in text and "night" in text):
            contract = "accommodation"
        elif "transport" in text or "flight" in text or "self-driving" in text:
            contract = "transportation"
        elif "cost" in text or "budget" in text or "cuisine" in text:
            contract = "budget"
        elif "invalid" in text or "sandbox" in text:
            contract = "entity"
        else:
            contract = "route"
        violations.append({"contract": contract, "issue": issue})
    for issue in audit.get("warnings", []) or []:
        text = str(issue).lower()
        contract = "entity" if "repeated" in text or "mismatch" in text else "warning"
        violations.append({"contract": contract, "issue": issue})
    violations.extend(audit.get("strict_issues", []))
    return violations[:40]


def active_roles(audit):
    if SINGLE_AGENT:
        return ["generic_repair_agent"]
    if GENERIC_AGENTS:
        return list(ROLE_SPECS.keys())
    board = contract_board(audit)
    contracts = {item["contract"] for item in board}
    if contracts.intersection({"complete_information", "destination_region", "local_transportation"}):
        contracts.add("route")
    if contracts.intersection({"current_city_alignment", "restaurant_diversity", "attraction_diversity", "complete_information", "entity_eligibility"}):
        contracts.add("entity")
    roles = []
    if {"route", "transportation"}.intersection(contracts):
        roles.append("route_agent")
    if {"entity", "warning"}.intersection(contracts):
        roles.append("entity_agent")
    if {"accommodation"}.intersection(contracts):
        roles.append("lodging_agent")
        # A lodging repair can require coordinated city/entity edits; keep one
        # cross-contract agent in the discussion rather than relying on a
        # single specialist proposal.
        if "route_agent" not in roles:
            roles.append("route_agent")
    if {"budget", "budget_ok"}.intersection(contracts):
        roles.append("budget_agent")
    if not roles and not audit.get("likely_pass"):
        roles = ["route_agent", "entity_agent"]
    return roles[:4]


def agent_prompt(role_id, task, current_plan, audit, round_idx, board_state):
    if role_id == "generic_repair_agent" or GENERIC_AGENTS:
        responsibility = "all TravelPlanner contracts, with no role specialization"
        contracts = ["route", "entity", "accommodation", "budget", "transportation"]
    else:
        responsibility = ROLE_SPECS[role_id]["responsibility"]
        contracts = ROLE_SPECS[role_id]["contracts"]
    system_prompt = (
        f"You are {role_id} in CC-MAR, a verifier-mediated multi-agent repair system. "
        "You independently propose a minimal repair patch. Return JSON only."
    )
    visible_board = {} if DISABLE_BOARD else board_state
    user_prompt = json.dumps(
        {
            "role_id": role_id,
            "responsibility": responsibility,
            "target_contracts": contracts,
            "task": task,
            "round": round_idx,
            "current_plan": current_plan,
            "deterministic_verifier_audit": audit,
            "shared_board": visible_board,
            "instructions": [
                "Use only entities and transportation from reference_information_compact.",
                "Prefer minimal field-level patches over full rewrites.",
                "Return patches as the primary output; include revised_plan only if the patch cannot be expressed as day-field-value edits.",
                "Keep rationale under 30 words to avoid truncation.",
                "Do not change fields outside your target contracts unless necessary to avoid breaking the plan.",
                "If you cannot safely repair your target contracts, return repairable=false.",
                "Last-day Accommodation may be '-' only when returning to origin.",
            ],
            "output_schema": {
                "agent_id": role_id,
                "repairable": True,
                "target_contracts": contracts,
                "patches": [
                    {"day": 1, "field": "accommodation", "value": "new field value"}
                ],
                "revised_plan": [],
                "expected_effect": {"decrease": ["contract"], "risk": ["contract"]},
                "rationale": "short grounded explanation",
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def critic_prompt(task, current_plan, audit, proposals):
    system_prompt = (
        "You are the critic agent in CC-MAR. Review other agents' patches for cross-contract risk. "
        "Do not choose the final plan. Return JSON only."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "current_plan": current_plan,
            "deterministic_verifier_audit": audit,
            "proposals": proposals,
            "instructions": [
                "Mark invalid if a proposal invents entities, breaks route closure, changes unrelated fields without reason, or ignores the target contract.",
                "Mark risky if it may fix one contract while harming budget, route, or entity validity.",
                "Mark safe only when the patch is local and grounded.",
            ],
            "output_schema": {
                "judgments": [
                    {
                        "proposal_id": "string",
                        "status": "safe | risky | invalid",
                        "reason": "short explanation",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt



ROLE_FIELDS = {
    "route_agent": {"current_city", "transportation"},
    "entity_agent": {"breakfast", "lunch", "dinner", "attraction", "accommodation"},
    "lodging_agent": {"accommodation"},
    "budget_agent": {"breakfast", "lunch", "dinner", "accommodation"},
}


def validate_role_changes(role_id, before, after):
    if not after:
        return
    allowed = set(REQUIRED_PLAN_KEYS)-{"days"} if GENERIC_AGENTS or role_id=="generic_repair_agent" else ROLE_FIELDS.get(role_id, set())
    if len(before) != len(after):
        raise ValueError("patch may not change plan length")
    for old, new in zip(before, after):
        changed = {key for key in set(old)|set(new) if old.get(key)!=new.get(key)}
        if not changed <= allowed:
            raise ValueError(f"{role_id} changed fields outside its contract: {sorted(changed-allowed)}")


def apply_patches(plan, patches):
    if not isinstance(patches, list):
        raise ValueError("patches must be a list")
    for patch in patches:
        if not isinstance(patch, dict) or type(patch.get("day")) is not int:
            raise ValueError("patch day must be an integer")
        if not 1 <= patch["day"] <= len(plan) or patch.get("field") not in set(REQUIRED_PLAN_KEYS)-{"days"}:
            raise ValueError("patch day or field outside allowed schema")
        if not isinstance(patch.get("value"), str):
            raise ValueError("patch value must be a string")
    revised = deepcopy(plan)
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        try:
            day_idx = int(patch.get("day")) - 1
        except Exception:
            continue
        field = str(patch.get("field", "")).strip()
        if field not in REQUIRED_PLAN_KEYS or field == "days":
            continue
        if 0 <= day_idx < len(revised):
            value = patch.get("value", "-")
            revised[day_idx][field] = canonicalize_patch_value(
                "-" if value in [None, ""] else str(value).strip(),
                field,
                revised[day_idx],
            )
    return revised


def canonicalize_patch_value(value, field, day):
    if value == "-" or field not in {"breakfast", "lunch", "dinner", "attraction", "accommodation"}:
        return value
    if field == "attraction":
        parts = [part.strip() for part in str(value).split(";") if part.strip()]
        fixed = [canonicalize_entity(part, "attraction", day) for part in parts]
        return ";".join(fixed) + (";" if fixed else "")
    if field in {"breakfast", "lunch", "dinner"}:
        return canonicalize_entity(value, "restaurant", day)
    return canonicalize_entity(value, "accommodation", day)


def canonicalize_entity(value, entity_type, day):
    loaders = {
        "restaurant": (_load_restaurants, "Name", "City"),
        "attraction": (_load_attractions, "Name", "City"),
        "accommodation": (_load_accommodations, "NAME", "city"),
    }
    loader, name_col, city_col = loaders[entity_type]
    name, city = get_valid_name_city(value)
    if not city or city == "-":
        _, inferred_city = get_valid_name_city(day.get("current_city", ""))
        city = inferred_city if inferred_city and inferred_city != "-" else city
    if not name or not city or city == "-":
        return value

    df = loader()
    city_rows = df[df[city_col].astype(str).str.casefold() == str(city).casefold()]
    if city_rows.empty:
        return value
    names = city_rows[name_col].astype(str).tolist()
    exact = [candidate for candidate in names if candidate.casefold() == str(name).casefold()]
    if exact:
        return f"{exact[0]}, {city}"
    contains = [candidate for candidate in names if str(name).casefold() in candidate.casefold() or candidate.casefold() in str(name).casefold()]
    if contains:
        return f"{contains[0]}, {city}"
    matches = difflib.get_close_matches(str(name), names, n=1, cutoff=0.82)
    if matches:
        return f"{matches[0]}, {city}"
    return value


def proposal_to_plan(proposal, current_plan, expected_days):
    plan = normalize_plan_list(proposal, expected_days)
    if plan:
        return plan
    if isinstance(proposal, dict):
        plan = normalize_plan_list(proposal.get("revised_plan"), expected_days)
        if plan:
            return plan
        patched = apply_patches(current_plan, proposal.get("patches", []))
        if patched != current_plan:
            return patched
    return []


def propose_from_agent(role_id, task, current_plan, audit, round_idx, board_state, idx):
    sp, up = agent_prompt(role_id, task, current_plan, audit, round_idx, board_state)
    try:
        raw = call_llm(sp, up, max_tokens=4096, temperature=0.05, cache_tag=f"agent_{role_id}_idx{idx}_r{round_idx}")
        parsed = parse_json_payload(raw)
    except ModelRequestError:
        raise
    except Exception as exc:
        return {
            "proposal_id": f"{role_id}_r{round_idx}",
            "agent_id": role_id,
            "repairable": False,
            "raw": "",
            "parsed": {"error": str(exc)},
            "plan": [],
        }
    if not isinstance(parsed, dict):
        parsed = salvage_patch_payload(raw, role_id)
    try:
        plan = proposal_to_plan(parsed, current_plan, int(task["days"]))
        validate_role_changes(role_id, current_plan, plan)
    except ValueError as exc:
        parsed = {"repairable": False, "error": str(exc)}
        plan = []
    return {
        "proposal_id": f"{role_id}_r{round_idx}",
        "agent_id": role_id,
        "repairable": bool(parsed.get("repairable", bool(plan))),
        "raw": raw,
        "parsed": parsed,
        "plan": plan,
    }


def critic_judgments(task, current_plan, audit, proposals, idx, round_idx):
    if DISABLE_CRITIC:
        return {
            proposal["proposal_id"]: {"status": "safe", "reason": "critic disabled"}
            for proposal in proposals
        }
    compact = [
        {
            "proposal_id": p["proposal_id"],
            "agent_id": p["agent_id"],
            "repairable": p["repairable"],
            "parsed": p["parsed"],
        }
        for p in proposals
    ]
    sp, up = critic_prompt(task, current_plan, audit, compact)
    try:
        raw = call_llm(sp, up, max_tokens=2048, temperature=0, cache_tag=f"critic_idx{idx}_r{round_idx}")
        parsed = parse_json_payload(raw)
    except ModelRequestError:
        raise
    except Exception as exc:
        parsed = {"judgments": [], "error": str(exc)}
    judgments = {}
    for item in (parsed or {}).get("judgments", []) if isinstance(parsed, dict) else []:
        if isinstance(item, dict) and item.get("proposal_id"):
            judgments[str(item["proposal_id"])] = {
                "status": str(item.get("status", "risky")),
                "reason": str(item.get("reason", "")),
            }
    for proposal in proposals:
        judgments.setdefault(proposal["proposal_id"], {"status": "risky", "reason": "critic omitted proposal"})
    return judgments


def candidate_sort_key(item):
    after_audit = item["after_audit"]
    vec = item["after_vector"]
    status = item["critic"].get("status", "risky")
    status_bonus = 0 if status == "safe" else 1
    return (status_bonus, tuple(vec), -int(after_audit.get("score", 0)))


def mediate(task, current_plan, before_audit, proposals, judgments):
    before_vector = violation_vector(before_audit)
    evaluated = []
    decisions = []
    for proposal in proposals:
        proposal_id = proposal["proposal_id"]
        critic = judgments.get(proposal_id, {"status": "risky", "reason": "missing critic judgment"})
        if not proposal.get("repairable") or not proposal.get("plan"):
            decisions.append(
                {
                    "proposal_id": proposal_id,
                    "agent_id": proposal["agent_id"],
                    "decision": "rejected",
                    "reason": "not repairable or no valid plan",
                    "critic": critic,
                }
            )
            continue
        if critic.get("status") == "invalid":
            decisions.append({"proposal_id":proposal_id,"agent_id":proposal["agent_id"],"decision":"rejected","reason":"critic_invalid","critic":critic})
            continue
        after_audit = audit_plan(task, proposal["plan"])
        after_vector = violation_vector(after_audit)
        regression = protected_regression(before_audit, after_audit)
        if regression:
            decisions.append(
                {
                    "proposal_id": proposal_id,
                    "agent_id": proposal["agent_id"],
                    "decision": "rejected",
                    "reason": regression,
                    "before_vector": before_vector,
                    "after_vector": after_vector,
                    "critic": critic,
                }
            )
            continue
        if not vector_improves(after_vector, before_vector):
            decisions.append(
                {
                    "proposal_id": proposal_id,
                    "agent_id": proposal["agent_id"],
                    "decision": "rejected",
                    "reason": "violation vector did not improve",
                    "before_vector": before_vector,
                    "after_vector": after_vector,
                    "critic": critic,
                }
            )
            continue
        evaluated.append(
            {
                "proposal": proposal,
                "critic": critic,
                "after_audit": after_audit,
                "after_vector": after_vector,
            }
        )

    if not evaluated:
        return current_plan, before_audit, None, decisions

    selected = sorted(evaluated, key=candidate_sort_key)[0]
    proposal = selected["proposal"]
    decisions.append(
        {
            "proposal_id": proposal["proposal_id"],
            "agent_id": proposal["agent_id"],
            "decision": "accepted",
            "reason": "contract-dominating patch",
            "before_vector": before_vector,
            "after_vector": selected["after_vector"],
            "critic": selected["critic"],
        }
    )
    return proposal["plan"], selected["after_audit"], proposal, decisions


def rank_seed_plans(task, direct_rows, program_rows, idx):
    expected_days = int(task["days"])
    candidates = []
    if idx <= len(direct_rows):
        plan = normalize_plan_list(direct_rows[idx - 1].get("plan"), expected_days)
        if plan:
            audit = audit_plan(task, plan)
            candidates.append(("direct_seed", plan, audit))
    if idx <= len(program_rows):
        plan = normalize_plan_list(program_rows[idx - 1].get("plan"), expected_days)
        if plan:
            audit = audit_plan(task, plan)
            candidates.append(("program_seed", plan, audit))
    if not candidates:
        empty = []
        return "empty_seed", empty, audit_plan(task, empty)
    return sorted(candidates, key=lambda item: tuple(violation_vector(item[2])))[0]


def load_existing_plan(idx):
    output_path = OUTPUT_DIR / f"generated_plan_{idx}.json"
    debug_path = DEBUG_DIR / f"debug_{idx}.json"
    if not (output_path.exists() and debug_path.exists()):
        return None
    try:
        if json.loads(debug_path.read_text()).get("status") != "completed":
            return None
        payload = json.loads(output_path.read_text(encoding="utf-8"))[0]
        plan = payload.get(f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results")
        return plan if valid_plan_container(plan) else None
    except Exception:
        return None


def run_one(row, idx, direct_rows, program_rows):
    task = build_task(row)
    seed_source, current_plan, current_audit = rank_seed_plans(task, direct_rows, program_rows, idx)
    debug = {
        "idx": idx,
        "strategy": STRATEGY,
        "config": {
            "rounds": ROUNDS,
            "disable_critic": DISABLE_CRITIC,
            "disable_board": DISABLE_BOARD,
            "generic_agents": GENERIC_AGENTS,
            "single_agent": SINGLE_AGENT,
        },
        "task": task,
        "initial_seed_source": seed_source,
        "initial_violation_vector": violation_vector(current_audit),
        "initial_audit": current_audit,
        "rounds": [],
        "accepted_patches": [],
    }

    board_state = {
        "violation_board": contract_board(current_audit),
        "accepted_patches": [],
        "mediator_decisions": [],
    }

    for round_idx in range(1, ROUNDS + 1):
        if current_audit.get("likely_pass"):
            debug["rounds"].append({"round": round_idx, "status": "stopped_likely_pass"})
            break
        roles = active_roles(current_audit)
        proposals = [
            propose_from_agent(role_id, task, current_plan, current_audit, round_idx, board_state, idx)
            for role_id in roles
        ]
        judgments = critic_judgments(task, current_plan, current_audit, proposals, idx, round_idx)
        next_plan, next_audit, accepted, decisions = mediate(task, current_plan, current_audit, proposals, judgments)

        round_record = {
            "round": round_idx,
            "before_vector": violation_vector(current_audit),
            "active_agents": roles,
            "agent_proposals": [
                {
                    "proposal_id": p["proposal_id"],
                    "agent_id": p["agent_id"],
                    "repairable": p["repairable"],
                    "parsed": p["parsed"],
                    "plan_valid_parse": bool(p["plan"]),
                }
                for p in proposals
            ],
            "critic_judgments": judgments,
            "mediator_decisions": decisions,
            "accepted_patch": accepted["proposal_id"] if accepted else None,
            "after_vector": violation_vector(next_audit),
        }
        debug["rounds"].append(round_record)
        board_state["mediator_decisions"].extend(decisions)
        if accepted is None:
            break
        current_plan = next_plan
        current_audit = next_audit
        accepted_record = {
            "round": round_idx,
            "proposal_id": accepted["proposal_id"],
            "agent_id": accepted["agent_id"],
            "patch": accepted["parsed"],
            "after_vector": violation_vector(current_audit),
        }
        debug["accepted_patches"].append(accepted_record)
        board_state["accepted_patches"].append(accepted_record)
        board_state["violation_board"] = contract_board(current_audit)

    debug["final_plan"] = current_plan
    debug["final_audit"] = current_audit
    debug["final_violation_vector"] = violation_vector(current_audit)
    debug["status"] = "completed"
    atomic_write_text(DEBUG_DIR / f"debug_{idx}.json", json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return current_plan, debug


def main():
    df = pd.read_csv(DATA_PATH)
    globals()["DATASET_FINGERPRINT"] = hashlib.sha256(df.to_json().encode()).hexdigest()
    select_rows(df)  # Validate the request before claiming output paths.
    direct_rows = load_jsonl(DIRECT_SUBMISSION_FILE)
    program_rows = load_jsonl(PROGRAM_SUBMISSION_FILE)
    if len(direct_rows) != len(df) or len(program_rows) != len(df):
        raise ValueError("Direct/program seed counts must match the validation dataset.")
    configure_run(globals())
    with exclusive_run(OUT_ROOT):
        _run_dataframe(df, direct_rows, program_rows)


def _run_dataframe(df, direct_rows, program_rows):
    ensure_dirs()
    df_run = select_rows(df)
    key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_results"
    parsed_key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results"
    submission_rows = []
    for idx, row in df_run.iterrows():
        number = idx + 1
        plan = load_existing_plan(number) if RESUME else None
        if plan is not None:
            print(f"skipped {number}", flush=True)
        else:
            try:
                plan, _ = run_one(row, number, direct_rows, program_rows)
            except Exception as exc:
                plan = []
                atomic_write_text(DEBUG_DIR / f"debug_{number}.json", json.dumps({"idx":number,"status":"failed","error_type":type(exc).__name__,"error":str(exc)}))
            atomic_write_text(OUTPUT_DIR / f"generated_plan_{number}.json",
                json.dumps([{key: render_plan_text(plan), parsed_key: plan}], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"generated {number}", flush=True)
        submission_rows.append({"idx": number, "query": str(row.get("query", "")), "plan": plan})

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
