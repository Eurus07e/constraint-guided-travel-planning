import ast
import hashlib
import json
import os
import re
import time
from itertools import permutations
from pathlib import Path

import pandas as pd
from utils.paths import DATABASE

from baselines import program_planner_v23_llm as base

ORIGINAL_CALL_DEEPSEEK_JSON = base.call_deepseek_json


MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
STRATEGY = "program_v23_best"
MODE = "sole-planning"

DATA_PATH = DATABASE / "validation.csv"
OUT_ROOT = Path("outputs_program_v23_best")
OUTPUT_DIR = OUT_ROOT / "validation"
DEBUG_DIR = OUT_ROOT / "debug"
CACHE_DIR = OUT_ROOT / "llm_cache"
SUBMISSION_DIR = Path("evaluation")
SUBMISSION_FILE = SUBMISSION_DIR / "validation_deepseek-v4-flash_program_v23_best_sole-planning_submission.jsonl"

STATE_NAMES = {
    "Florida", "Texas", "California", "Illinois", "Michigan", "New York",
    "South Carolina", "North Carolina", "Georgia", "Washington", "Virginia",
    "Ohio", "Pennsylvania", "Missouri", "Arizona", "Nevada", "Colorado",
    "Tennessee", "Louisiana", "Alabama", "Oregon", "Maryland", "Indiana"
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def safe_literal_eval(x, default):
    try:
        if pd.isna(x):
            return default
        return ast.literal_eval(str(x))
    except Exception:
        return default


def cached_call_deepseek_json(system_prompt, user_prompt, max_tokens=2048, temperature=0):
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com").rstrip("/")
    model_name = os.getenv("MODEL_NAME", MODEL_NAME)

    if not api_key or not api_key.startswith("sk-"):
        return {}

    payload_for_hash = {
        "model": model_name,
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    key = hashlib.sha256(json.dumps(payload_for_hash, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        try:
            return json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            pass

    result = ORIGINAL_CALL_DEEPSEEK_JSON(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)

    try:
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return result


base.call_deepseek_json = cached_call_deepseek_json


ORIGINAL_CHOOSE_TRANSPORT = base.choose_transport

def hard_choose_transport(sections, origin, dest, date=None, required_method=None):
    options = base.get_transport_options(sections, origin, dest, date, None)

    if required_method:
        req = str(required_method).lower().strip()
        hard_filtered = []
        for x in options:
            method = str(x.get("method", "")).lower()
            if req in method:
                hard_filtered.append(x)

        if hard_filtered:
            return sorted(hard_filtered, key=lambda x: x.get("cost", 999999))[0]

        return None

    if options:
        return sorted(options, key=lambda x: x.get("cost", 999999))[0]

    return None

base.choose_transport = hard_choose_transport


def compute_segment_nights_by_day(state):
    plan = state["plan"]
    result = {}

    i = 0
    last_night_day = len(plan) - 1

    while i < last_night_day:
        city = plan[i]["stay_city"]
        j = i

        while j < last_night_day and plan[j]["stay_city"] == city:
            j += 1

        segment_nights = j - i

        for k in range(i, j):
            result[plan[k]["day"]] = segment_nights

        i = j

    return result


def recalc_cost_strict(state):
    people = int(state["constraints"].get("people", 1))

    cost = {
        "transportation": 0.0,
        "restaurants": 0.0,
        "accommodations": 0.0,
        "budget": state["constraints"]["budget"],
    }

    for day in state["plan"]:
        t = day.get("transportation_obj")
        if t:
            cost["transportation"] += float(t.get("cost", 0.0)) * people

        for key in ["breakfast_obj", "lunch_obj", "dinner_obj"]:
            item = day.get(key)
            if item:
                cost["restaurants"] += float(item.get("cost", 0.0)) * people

        if day["day"] != len(state["plan"]):
            acc = day.get("accommodation_obj")
            if acc:
                cost["accommodations"] += float(acc.get("price", 0.0))

    cost["total"] = cost["transportation"] + cost["restaurants"] + cost["accommodations"]
    state["cost"] = cost
    return state


def fix_accommodations_by_segments(state, candidates):
    constraints = state["constraints"]
    people = constraints["people"]
    plan = state["plan"]

    i = 0
    last_night_day = len(plan) - 1

    while i < last_night_day:
        city = plan[i]["stay_city"]
        j = i

        while j < last_night_day and plan[j]["stay_city"] == city:
            j += 1

        segment_nights = j - i
        items = candidates["accommodations"].get(city, [])

        valid_items = []
        for item in items:
            if base.accommodation_matches(item, constraints, people, segment_nights):
                valid_items.append(item)

        if not valid_items:
            for item in items:
                if item.get("maximum_occupancy", 0) >= people and item.get("minimum_nights", 1) <= segment_nights:
                    valid_items.append(item)

        if valid_items:
            chosen = sorted(valid_items, key=lambda x: x.get("price", 999999))[0]
        elif items:
            chosen = sorted(items, key=lambda x: x.get("price", 999999))[0]
        else:
            chosen = {
                "name": "-",
                "city": city,
                "price": 0,
                "minimum_nights": 1,
                "maximum_occupancy": 999,
                "room_type": "",
                "house_rules": ""
            }

        for k in range(i, j):
            plan[k]["accommodation_obj"] = chosen
            plan[k]["accommodation"] = chosen["name"]

        i = j

    recalc_cost_strict(state)
    return state


def semantic_contains_requirement(required, value):
    terms = normalize_terms(required)
    if not terms:
        return True

    text = str(value).lower()

    synonym_groups = {
        "no smoking": ["no smoking", "smoking is not allowed", "smoke-free", "non smoking", "no smoke"],
        "pets allowed": ["pets allowed", "pet friendly", "allows pets"],
        "no parties": ["no parties", "no parties or events", "parties are not allowed"],
        "private room": ["private room"],
        "entire home": ["entire home", "entire home/apt", "entire apartment", "entire place"],
        "shared room": ["shared room"]
    }

    expanded = []
    for term in terms:
        expanded.append(term)
        for key, vals in synonym_groups.items():
            if term in key or key in term:
                expanded.extend(vals)

    return any(t in text for t in expanded)


def candidate_has_valid_min_nights(state, candidates):
    nights_by_day = compute_segment_nights_by_day(state)
    failures = []

    for day in state["plan"]:
        if day["day"] == len(state["plan"]):
            continue

        acc = day.get("accommodation_obj", {})
        min_nights = float(acc.get("minimum_nights", 1))
        segment_nights = nights_by_day.get(day["day"], 1)

        if min_nights > segment_nights:
            failures.append({
                "type": "minimum_nights",
                "day": day["day"],
                "message": f"Accommodation requires minimum {min_nights} nights, but segment has {segment_nights} nights."
            })

    return failures



def normalize_terms(x):
    if not x:
        return []
    if isinstance(x, list):
        return [str(v).lower() for v in x if str(v).strip()]
    return [str(x).lower()]


def entity_sets(candidates):
    data = {
        "restaurants": {},
        "attractions": {},
        "accommodations": {}
    }

    for city, items in candidates.get("restaurants", {}).items():
        data["restaurants"][city] = {x["name"] for x in items}

    for city, items in candidates.get("attractions", {}).items():
        data["attractions"][city] = set(items)

    for city, items in candidates.get("accommodations", {}).items():
        data["accommodations"][city] = {x["name"] for x in items}

    return data


def generate_route_candidates(sections, constraints, candidates, max_routes=80):
    org = constraints["org"]
    dest = constraints["dest"]
    days = constraints["days"]
    visiting_city_number = int(constraints.get("visiting_city_number", 1))

    usable = [c for c in candidates["usable_cities"] if c != org and c not in STATE_NAMES]

    if not usable:
        usable = [c for c in candidates["cities"] if c != org and c not in STATE_NAMES]

    if not usable:
        usable = [dest]

    k = max(1, min(visiting_city_number, len(usable), days - 1))

    route_candidates = []

    if dest in usable and dest not in STATE_NAMES:
        optional = [c for c in usable if c != dest]
        if k == 1:
            route_candidates.append([dest])
        else:
            for extra in permutations(optional, k - 1):
                route_candidates.append(base.unique_keep_order([dest] + list(extra)))
                route_candidates.append(base.unique_keep_order(list(extra) + [dest]))
    else:
        for route in permutations(usable, k):
            route_candidates.append(list(route))

    if not route_candidates:
        route_candidates.append(usable[:k])

    scored = []
    for route in route_candidates[:max_routes * 4]:
        if len(route) != k:
            continue

        score, legs = base.route_score(
            sections,
            org,
            route,
            constraints["dates"],
            constraints.get("transportation")
        )

        for c in route:
            if c in STATE_NAMES:
                score -= 100000
            score += 300 if candidates["restaurants"].get(c) else -500
            score += 300 if candidates["attractions"].get(c) else -500
            score += 300 if candidates["accommodations"].get(c) else -500

        if legs:
            missing = sum(1 for x in legs if x is None)
            score -= 3000 * missing

        scored.append((score, route, legs))

    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    seen = set()
    for score, route, legs in scored:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        out.append((route, legs, score))
        if len(out) >= max_routes:
            break

    return out


def strict_validate(state, candidates):
    failures = []
    constraints = state["constraints"]
    plan = state["plan"]
    sets = entity_sets(candidates)

    for city in state.get("route", []):
        if city in STATE_NAMES:
            failures.append({
                "type": "state_name_in_route",
                "message": f"Route contains non-city state name: {city}"
            })

    required_fields = ["current_city", "transportation", "breakfast", "lunch", "dinner", "attractions", "accommodation"]

    for day in plan:
        for f in required_fields:
            value = day.get(f)
            if value is None or value == "" or value == []:
                failures.append({
                    "type": "complete_information",
                    "day": day["day"],
                    "message": f"Missing {f} on Day {day['day']}"
                })

    for day in plan:
        text = day.get("current_city", "")
        transportation = day.get("transportation", "")

        if text.startswith("from "):
            m = re.match(r"from (.+) to (.+)", text)
            if not m:
                failures.append({
                    "type": "bad_current_city_format",
                    "day": day["day"],
                    "message": f"Bad current city format: {text}"
                })
            else:
                origin, dest = m.group(1), m.group(2)
                if transportation == "-":
                    failures.append({
                        "type": "missing_transportation",
                        "day": day["day"],
                        "message": f"City changes from {origin} to {dest} but transportation is missing."
                    })
                elif f"from {origin} to {dest}" not in transportation:
                    failures.append({
                        "type": "transportation_direction",
                        "day": day["day"],
                        "message": f"Transportation direction does not match current city change: {text} vs {transportation}"
                    })

    restaurant_seen = set()
    attraction_seen = set()

    for day in plan:
        city = day["stay_city"]

        for meal_key, obj_key in [
            ("breakfast", "breakfast_obj"),
            ("lunch", "lunch_obj"),
            ("dinner", "dinner_obj")
        ]:
            name = day.get(meal_key)
            obj = day.get(obj_key, {})

            if not name or name == "-":
                failures.append({
                    "type": "missing_restaurant",
                    "day": day["day"],
                    "message": f"{meal_key} missing on Day {day['day']}"
                })
                continue

            if obj.get("city") != city:
                failures.append({
                    "type": "wrong_city_restaurant",
                    "day": day["day"],
                    "message": f"{name} is not assigned to current stay city {city}"
                })

            if name not in sets["restaurants"].get(city, set()):
                failures.append({
                    "type": "out_of_sandbox_restaurant",
                    "day": day["day"],
                    "message": f"{name} is not in restaurant candidates of {city}"
                })

            if name in restaurant_seen:
                failures.append({
                    "type": "duplicate_restaurant",
                    "day": day["day"],
                    "message": f"Restaurant repeated: {name}"
                })
            restaurant_seen.add(name)

        for a in day.get("attractions", []):
            if not a or a == "-":
                failures.append({
                    "type": "missing_attraction",
                    "day": day["day"],
                    "message": f"Attraction missing on Day {day['day']}"
                })
                continue

            if a not in sets["attractions"].get(city, set()):
                failures.append({
                    "type": "out_of_sandbox_attraction",
                    "day": day["day"],
                    "message": f"{a} is not in attraction candidates of {city}"
                })

            if a in attraction_seen:
                failures.append({
                    "type": "duplicate_attraction",
                    "day": day["day"],
                    "message": f"Attraction repeated: {a}"
                })
            attraction_seen.add(a)

        if day["day"] != len(plan):
            acc = day.get("accommodation_obj", {})
            acc_name = day.get("accommodation")
            if not acc_name or acc_name == "-":
                failures.append({
                    "type": "missing_accommodation",
                    "day": day["day"],
                    "message": f"Accommodation missing on Day {day['day']}"
                })
            else:
                if acc.get("city") != city:
                    failures.append({
                        "type": "wrong_city_accommodation",
                        "day": day["day"],
                        "message": f"Accommodation city mismatch on Day {day['day']}"
                    })

                if acc_name not in sets["accommodations"].get(city, set()):
                    failures.append({
                        "type": "out_of_sandbox_accommodation",
                        "day": day["day"],
                        "message": f"{acc_name} is not in accommodation candidates of {city}"
                    })

    if state["cost"]["total"] > constraints["budget"]:
        failures.append({
            "type": "budget",
            "message": f"Total cost {state['cost']['total']} exceeds budget {constraints['budget']}"
        })

    required_transport = constraints.get("transportation")
    if required_transport:
        req = str(required_transport).lower()
        for day in plan:
            if day["transportation"] != "-" and req not in day["transportation"].lower():
                failures.append({
                    "type": "transportation_constraint",
                    "day": day["day"],
                    "message": f"Required transportation {required_transport}, got {day['transportation']}"
                })

    required_cuisine = constraints.get("cuisine")
    cuisine_terms = normalize_terms(required_cuisine)
    if cuisine_terms:
        ok = False
        for day in plan:
            for obj_key in ["breakfast_obj", "lunch_obj", "dinner_obj"]:
                cuisines = str(day.get(obj_key, {}).get("cuisines", "")).lower()
                if any(t in cuisines for t in cuisine_terms):
                    ok = True
        if not ok:
            failures.append({
                "type": "cuisine_constraint",
                "message": f"No restaurant satisfies cuisine constraint: {required_cuisine}"
            })

    room_type_terms = normalize_terms(constraints.get("room_type") or constraints.get("room type"))
    room_rule_terms = normalize_terms(constraints.get("room_rule") or constraints.get("house_rule") or constraints.get("house rule"))

    for day in plan:
        if day["day"] == len(plan):
            continue

        acc = day.get("accommodation_obj", {})

        if room_type_terms:
            value = str(acc.get("room_type", "")).lower()
            if not any(t in value for t in room_type_terms):
                failures.append({
                    "type": "room_type",
                    "day": day["day"],
                    "message": f"Room type constraint not satisfied on Day {day['day']}"
                })

        if room_rule_terms:
            value = str(acc.get("house_rules", "")).lower()
            if not any(t in value for t in room_rule_terms):
                failures.append({
                    "type": "room_rule",
                    "day": day["day"],
                    "message": f"Room rule constraint not satisfied on Day {day['day']}"
                })

    return failures


def failure_penalty(failure):
    t = failure.get("type")
    weights = {
        "state_name_in_route": 5000,
        "complete_information": 4000,
        "missing_transportation": 4000,
        "transportation_direction": 4000,
        "bad_current_city_format": 4000,
        "out_of_sandbox_restaurant": 3500,
        "out_of_sandbox_attraction": 3500,
        "out_of_sandbox_accommodation": 3500,
        "wrong_city_restaurant": 3000,
        "wrong_city_accommodation": 3000,
        "budget": 2500,
        "transportation_constraint": 2500,
        "room_type": 2200,
        "room_rule": 2200,
        "cuisine_constraint": 1800,
        "duplicate_restaurant": 800,
        "duplicate_attraction": 800,
        "missing_restaurant": 3500,
        "missing_attraction": 3500,
        "missing_accommodation": 3500
    }
    return weights.get(t, 1000)


def score_state(state, candidates):
    recalc_cost_strict(state)
    fix_accommodations_by_segments(state, candidates)
    failures = strict_validate(state, candidates)
    failures.extend(candidate_has_valid_min_nights(state, candidates))
    score = 100000

    for f in failures:
        score -= failure_penalty(f)

    score -= state["cost"]["total"] * 0.1

    budget = state["constraints"]["budget"]
    if state["cost"]["total"] <= budget:
        score += 1000

    unique_restaurants = set()
    unique_attractions = set()

    for day in state["plan"]:
        for k in ["breakfast", "lunch", "dinner"]:
            if day.get(k) and day.get(k) != "-":
                unique_restaurants.add(day[k])
        for a in day.get("attractions", []):
            if a and a != "-":
                unique_attractions.add(a)

    score += len(unique_restaurants) * 30
    score += len(unique_attractions) * 30

    return score, failures


def strict_llm_repair_advisor(row, state, failures):
    if os.getenv("USE_LLM", "1") != "1":
        return []

    system_prompt = (
        "You are an expert TravelPlanner constraint auditor and repair controller. "
        "You must carefully compare the Python-generated plan against TravelPlanner-style constraints. "
        "Do not invent restaurants, hotels, attractions, cities, or transportation. "
        "The candidate database and Python city fields are authoritative. "
        "Your role is not to freely rewrite the itinerary. "
        "Your role is to give structured feedback to Python. "
        "You may choose one of three strategies: "
        "1. discard_or_reject_bad_candidate: if the current route or candidate plan has unrecoverable bugs; "
        "2. patch_template_or_module: if the Python template or module has a systematic bug, such as minimum nights, transportation direction, or room rule handling; "
        "3. direct_override_fields: only when you are highly confident and the corrected fields use existing candidate entities from the plan or database. "
        "Return only one valid JSON object. No markdown."
    )

    plan_summary = []
    for day in state["plan"]:
        plan_summary.append({
            "day": day["day"],
            "current_city": day["current_city"],
            "stay_city": day["stay_city"],
            "transportation": day["transportation"],
            "breakfast": day["breakfast"],
            "lunch": day["lunch"],
            "dinner": day["dinner"],
            "attractions": day["attractions"],
            "accommodation": day["accommodation"]
        })

    user_prompt = json.dumps({
        "task": "Audit the plan and decide how Python should repair it.",
        "important_travelplanner_constraints": [
            "Every day must have complete fields.",
            "Every city transition must have matching transportation direction.",
            "Restaurants, attractions, and accommodations must be in the current city and in the sandbox candidates.",
            "Restaurants and attractions should be diverse.",
            "Accommodation minimum nights must be satisfied by consecutive nights in the same city.",
            "Room type and room rule are hard constraints when specified.",
            "Transportation type is a hard constraint when specified.",
            "Budget is a hard constraint.",
            "Do not infer city from entity names. Use Python's candidate city fields only."
        ],
        "allowed_actions": [
            "keep",
            "discard_or_reject_bad_candidate",
            "patch_template_or_module",
            "replace_accommodation",
            "replace_transportation",
            "insert_transportation",
            "replace_restaurant",
            "replace_attraction",
            "enforce_cuisine",
            "reduce_cost",
            "direct_override_fields"
        ],
        "validator_failures": failures[:30],
        "cost": state["cost"],
        "route": state["route"],
        "daily_city": state.get("daily_city", []),
        "plan": plan_summary,
        "query": str(row.get("query", "")),
        "output_schema": {
            "confidence": "number between 0 and 1",
            "diagnosis": "short diagnosis",
            "strategy": "keep, discard_or_reject_bad_candidate, patch_template_or_module, or direct_override_fields",
            "actions": [
                {
                    "action": "one allowed action",
                    "day": "integer or null",
                    "target_city": "string or null",
                    "field": "string or null",
                    "value": "string or null",
                    "reason": "short reason"
                }
            ],
            "python_patch_hint": "if strategy is patch_template_or_module, describe the template/module bug"
        }
    }, ensure_ascii=False)

    obj = base.call_deepseek_json(system_prompt, user_prompt, max_tokens=3072, temperature=0)
    actions = obj.get("actions", [])
    if not isinstance(actions, list):
        actions = []

    allowed = {
        "replace_accommodation",
        "replace_transportation",
        "insert_transportation",
        "replace_restaurant",
        "replace_attraction",
        "enforce_cuisine",
        "reduce_cost",
        "keep",
        "discard_or_reject_bad_candidate",
        "patch_template_or_module",
        "direct_override_fields"
    }

    clean = []
    for a in actions:
        if isinstance(a, dict) and a.get("action") in allowed:
            a["_llm_confidence"] = obj.get("confidence")
            a["_llm_strategy"] = obj.get("strategy")
            a["_llm_diagnosis"] = obj.get("diagnosis")
            a["_python_patch_hint"] = obj.get("python_patch_hint")
            clean.append(a)

    if not clean and not failures:
        clean.append({
            "action": "keep",
            "day": None,
            "target_city": None,
            "reason": "No validator failures.",
            "_llm_confidence": obj.get("confidence"),
            "_llm_strategy": obj.get("strategy"),
            "_llm_diagnosis": obj.get("diagnosis")
        })

    return clean[:8]


def pick_new_attractions(city, candidates, used, count=2):
    items = candidates["attractions"].get(city, [])
    chosen = []
    for a in items:
        if a not in used:
            used.add(a)
            chosen.append(a)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for a in items:
            if a not in chosen:
                chosen.append(a)
            if len(chosen) >= count:
                break
    return chosen or ["-"]


def apply_python_repair(state, candidates, sections, failures, actions):
    constraints = state["constraints"]

    requested = {a.get("action") for a in actions if isinstance(a, dict)}

    failure_types = {f.get("type") for f in failures}

    if "discard_or_reject_bad_candidate" in requested:
        state["_discard_candidate"] = True
        return state

    if "patch_template_or_module" in requested:
        state["_patch_hint"] = [
            a.get("_python_patch_hint") or a.get("reason")
            for a in actions
            if isinstance(a, dict) and a.get("action") == "patch_template_or_module"
        ]

    if "direct_override_fields" in requested:
        for a in actions:
            if not isinstance(a, dict):
                continue
            if a.get("action") != "direct_override_fields":
                continue
            try:
                day_num = int(a.get("day"))
            except Exception:
                continue
            if not (1 <= day_num <= len(state["plan"])):
                continue
            field = a.get("field")
            value = a.get("value")
            if field in {"transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"} and value:
                if float(a.get("_llm_confidence") or 0) >= 0.85:
                    state["plan"][day_num - 1][field] = value

    if "missing_transportation" in failure_types or "transportation_direction" in failure_types or "insert_transportation" in requested or "replace_transportation" in requested:
        for day in state["plan"]:
            text = day.get("current_city", "")
            m = re.match(r"from (.+) to (.+)", text)
            if not m:
                continue
            origin, dest = m.group(1), m.group(2)
            transport = base.choose_transport(sections, origin, dest, day.get("date"), constraints.get("transportation"))
            if transport:
                day["transportation_obj"] = transport
                day["transportation"] = base.format_transport(transport, origin, dest)

    if "wrong_city_accommodation" in failure_types or "out_of_sandbox_accommodation" in failure_types or "missing_accommodation" in failure_types or "room_type" in failure_types or "room_rule" in failure_types or "minimum_nights" in failure_types or "replace_accommodation" in requested or "reduce_cost" in requested:
        for day in state["plan"]:
            if day["day"] == len(state["plan"]):
                continue
            city = day["stay_city"]
            items = candidates["accommodations"].get(city, [])
            if not items:
                continue
            chosen = base.choose_accommodation(items, constraints, constraints["people"], 1)
            if chosen:
                day["accommodation_obj"] = chosen
                day["accommodation"] = chosen["name"]

    if "wrong_city_restaurant" in failure_types or "out_of_sandbox_restaurant" in failure_types or "missing_restaurant" in failure_types or "duplicate_restaurant" in failure_types or "cuisine_constraint" in failure_types or "replace_restaurant" in requested or "enforce_cuisine" in requested or "reduce_cost" in requested:
        used = set()
        for day in state["plan"]:
            city = day["stay_city"]
            for meal_name, obj_name in [
                ("breakfast", "breakfast_obj"),
                ("lunch", "lunch_obj"),
                ("dinner", "dinner_obj")
            ]:
                item = base.pick_restaurant(city, candidates, used, constraints.get("cuisine"))
                day[obj_name] = item
                day[meal_name] = item["name"]

    if "out_of_sandbox_attraction" in failure_types or "missing_attraction" in failure_types or "duplicate_attraction" in failure_types or "replace_attraction" in requested:
        used = set()
        for day in state["plan"]:
            city = day["stay_city"]
            chosen = pick_new_attractions(city, candidates, used, 2)
            day["attractions"] = chosen

    fix_accommodations_by_segments(state, candidates)
    recalc_cost_strict(state)
    return state


def build_best_state(row, number):
    sections = base.get_reference_sections(row["reference_information"])

    constraints = base.llm_interpret_constraints(row)
    constraints["org"] = str(row["org"])
    constraints["dest"] = str(row["dest"])
    constraints["days"] = int(row["days"])
    constraints["dates"] = safe_literal_eval(row["date"], [])
    constraints["people"] = int(row["people_number"])
    constraints["budget"] = float(row["budget"])
    constraints["visiting_city_number"] = int(row["visiting_city_number"])

    candidates = base.extract_candidates(sections)

    route_candidates = generate_route_candidates(sections, constraints, candidates, max_routes=5)

    best_state = None
    best_score = -10**18
    best_info = None

    for route_idx, (route, route_legs, route_score_value) in enumerate(route_candidates):
        state = base.build_plan(row, constraints, sections, candidates, route, route_legs)

        first_score, first_failures = score_state(state, candidates)
        actions_total = []

        for repair_round in range(1):
            score_now, failures_now = score_state(state, candidates)

            if not failures_now:
                if route_idx == 0:
                    actions = strict_llm_repair_advisor(row, state, failures_now)
                    actions_total.extend(actions)
                break

            if route_idx == 0:
                actions = strict_llm_repair_advisor(row, state, failures_now)
            else:
                actions = []

            actions_total.extend(actions)
            state = apply_python_repair(state, candidates, sections, failures_now, actions)

        final_score, final_failures = score_state(state, candidates)

        if state.get("_discard_candidate"):
            final_score -= 50000

        if final_score > best_score:
            best_score = final_score
            best_state = state
            best_info = {
                "route_score": route_score_value,
                "first_score": first_score,
                "first_failures": first_failures,
                "final_score": final_score,
                "final_failures": final_failures,
                "llm_actions": actions_total
            }

    if best_state is None:
        route, route_legs, route_score_value = route_candidates[0]
        best_state = base.build_plan(row, constraints, sections, candidates, route, route_legs)
        final_score, final_failures = score_state(best_state, candidates)
        best_info = {
            "route_score": route_score_value,
            "first_score": final_score,
            "first_failures": final_failures,
            "final_score": final_score,
            "final_failures": final_failures,
            "llm_actions": []
        }

    save_debug(number, row, best_state, best_info)
    return best_state


def state_to_submission_plan(state):
    plan = []
    for day in state["plan"]:
        attraction_text = ";".join([f"{a}, {day['stay_city']}" for a in day["attractions"] if a != "-"]) + ";"
        if attraction_text == ";":
            attraction_text = "-"

        plan.append({
            "days": day["day"],
            "current_city": day["current_city"],
            "transportation": day["transportation"],
            "breakfast": f"{day['breakfast']}, {day['stay_city']}" if day["breakfast"] != "-" else "-",
            "attraction": attraction_text,
            "lunch": f"{day['lunch']}, {day['stay_city']}" if day["lunch"] != "-" else "-",
            "dinner": f"{day['dinner']}, {day['stay_city']}" if day["dinner"] != "-" else "-",
            "accommodation": f"{day['accommodation']}, {day['stay_city']}" if day["accommodation"] != "-" else "-"
        })
    return plan


def save_debug(number, row, state, info):
    payload = {
        "idx": number,
        "query": str(row.get("query", "")),
        "constraints": state.get("constraints", {}),
        "route": state.get("route", []),
        "daily_city": state.get("daily_city", []),
        "cost": state.get("cost", {}),
        "route_score": info.get("route_score"),
        "first_score": info.get("first_score"),
        "first_failures": info.get("first_failures"),
        "final_score": info.get("final_score"),
        "final_failures": info.get("final_failures"),
        "llm_actions": info.get("llm_actions")
    }

    (DEBUG_DIR / f"debug_{number}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)

    ids_env = os.getenv("IDS")
    if ids_env:
        ids = [int(x.strip()) for x in ids_env.split(",") if x.strip()]
        df_run = df.iloc[[i - 1 for i in ids]]
    else:
        limit = os.getenv("LIMIT")
        if limit:
            df_run = df.head(int(limit))
        else:
            df_run = df

    generated = 0
    submission_rows = []

    for idx, row in df_run.iterrows():
        number = idx + 1
        state = build_best_state(row, number)
        text = base.render_plan_text(state)

        key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_results"
        out_path = OUTPUT_DIR / f"generated_plan_{number}.json"
        out_path.write_text(
            json.dumps([{key: text}], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        submission_rows.append({
            "idx": number,
            "query": str(row.get("query", "")),
            "plan": state_to_submission_plan(state)
        })

        generated += 1
        print(f"generated {number}", flush=True)

    SUBMISSION_FILE.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in submission_rows) + "\n",
        encoding="utf-8"
    )

    print(f"generated {generated} plans into {OUTPUT_DIR}")
    print(f"submission saved to {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
