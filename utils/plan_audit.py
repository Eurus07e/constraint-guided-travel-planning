import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
from utils.paths import DATABASE
DB_DIR = DATABASE


def extract_before_parenthesis(value):
    value = str(value or "").strip()
    match = re.search(r"^(.*?)\([^)]*\)", value)
    return match.group(1).strip() if match else value


def extract_from_to(text):
    match = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", str(text or ""), re.IGNORECASE)
    if not match:
        return "", ""
    return extract_before_parenthesis(match.group(1)), extract_before_parenthesis(match.group(2))


def get_valid_name_city(text):
    match = re.search(r"(.*?),\s*([^,]+)(\([^)]*\))?$", str(text or "").strip().rstrip(";"))
    if not match:
        return "-", "-"
    return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip())


def transportation_mode(text):
    text = str(text or "").lower()
    if "flight" in text:
        return "flight"
    if "self-driving" in text:
        return "self-driving"
    if "taxi" in text:
        return "taxi"
    return ""


def split_attractions(text):
    values = [x.strip() for x in str(text or "").split(";")]
    return [x for x in values if x and x != "-"]


def count_consecutive_values(values):
    if not values:
        return []
    groups = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            groups.append((current, count))
            current = value
            count = 1
    groups.append((current, count))
    return groups


def is_valid_city_sequence(city_list):
    if len(city_list) < 3:
        return False

    visited_cities = set()
    idx = 0
    while idx < len(city_list):
        city = city_list[idx]
        if city in visited_cities and idx not in {0, len(city_list) - 1}:
            return False

        count = 0
        while idx < len(city_list) and city_list[idx] == city:
            count += 1
            idx += 1

        if count == 1 and 0 < idx - 1 < len(city_list) - 1:
            return False

        visited_cities.add(city)

    return True


def _normalize_budget(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _distance_to_cost(distance_text, mode):
    text = str(distance_text or "").replace("km", "").replace(",", "").strip()
    if not text:
        return None
    try:
        distance = float(text)
    except ValueError:
        return None
    if mode == "self-driving":
        return int(distance * 0.05)
    if mode == "taxi":
        return int(distance)
    return None


@lru_cache(maxsize=1)
def _load_city_state_map():
    mapping = {}
    path = DB_DIR / "background/citySet_with_states.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        city, state = line.split("\t", 1)
        mapping[city.strip()] = state.strip()
    return mapping


@lru_cache(maxsize=1)
def _load_flights():
    return pd.read_csv(DB_DIR / "flights/clean_Flights_2022.csv").dropna(
        subset=["Flight Number", "OriginCityName", "DestCityName", "Price"]
    )


@lru_cache(maxsize=1)
def _load_accommodations():
    return pd.read_csv(DB_DIR / "accommodations/clean_accommodations_2022.csv").dropna(
        subset=["NAME", "city", "price", "minimum nights", "maximum occupancy", "room type", "house_rules"]
    )


@lru_cache(maxsize=1)
def _load_restaurants():
    return pd.read_csv(DB_DIR / "restaurants/clean_restaurant_2022.csv").dropna(
        subset=["Name", "City", "Average Cost", "Cuisines"]
    )


@lru_cache(maxsize=1)
def _load_attractions():
    return pd.read_csv(DB_DIR / "attractions/attractions.csv").dropna(subset=["Name", "City"])


@lru_cache(maxsize=1)
def _load_distance_matrix():
    return pd.read_csv(DB_DIR / "googleDistanceMatrix/distance.csv")


def _match_rows(df, name_col, city_col, name, city):
    if not name or not city or name == "-" or city == "-":
        return df.iloc[0:0]
    city_mask = df[city_col].astype(str).str.casefold() == str(city).casefold()
    name_series = df[name_col].astype(str)
    exact = df[city_mask & (name_series.str.casefold() == str(name).casefold())]
    if len(exact) > 0:
        return exact
    return df[city_mask & name_series.str.contains(str(name), case=False, regex=False)]


@lru_cache(maxsize=20000)
def _lookup_restaurant(item):
    name, city = get_valid_name_city(item)
    return _match_rows(_load_restaurants(), "Name", "City", name, city)


@lru_cache(maxsize=20000)
def _lookup_accommodation(item):
    name, city = get_valid_name_city(item)
    return _match_rows(_load_accommodations(), "NAME", "city", name, city)


@lru_cache(maxsize=20000)
def _lookup_attraction(item):
    name, city = get_valid_name_city(item)
    return _match_rows(_load_attractions(), "Name", "City", name, city)


@lru_cache(maxsize=20000)
def _lookup_transport(transportation, current_city):
    value = str(transportation or "").strip()
    if not value or value == "-":
        return None

    origin, destination = extract_from_to(value)
    if not origin or not destination:
        origin, destination = extract_from_to(current_city)

    if not origin or not destination:
        return None

    mode = transportation_mode(value)
    if mode == "flight":
        match = re.search(r"Flight Number:\s*([^,]+)", value, re.IGNORECASE)
        if not match:
            return None
        flight_number = match.group(1).strip()
        rows = _load_flights()
        rows = rows[
            (rows["Flight Number"].astype(str) == flight_number)
            & (rows["OriginCityName"].astype(str) == origin)
            & (rows["DestCityName"].astype(str) == destination)
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        return {
            "mode": mode,
            "origin": origin,
            "destination": destination,
            "price": float(row["Price"]),
        }

    if mode in {"self-driving", "taxi"}:
        rows = _load_distance_matrix()
        rows = rows[
            (rows["origin"].astype(str) == origin)
            & (rows["destination"].astype(str) == destination)
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        duration = row.get("duration")
        distance = row.get("distance")
        if pd.isna(duration) or pd.isna(distance):
            return None
        if "day" in str(duration).lower():
            return None
        return {
            "mode": mode,
            "origin": origin,
            "destination": destination,
            "price": _distance_to_cost(distance, mode),
        }

    return None


def _check_local_constraints(task, plan):
    local_constraint = task.get("local_constraint", {}) or {}
    misses = []
    cuisines_seen = set()

    for day in plan:
        transport = str(day.get("transportation", "") or "")
        if local_constraint.get("transportation") == "no flight" and "Flight" in transport:
            misses.append("transportation:no flight")
        if local_constraint.get("transportation") == "no self-driving" and "Self-driving" in transport:
            misses.append("transportation:no self-driving")

        accommodation = str(day.get("accommodation", "") or "")
        if accommodation and accommodation != "-":
            rows = _lookup_accommodation(accommodation)
            if len(rows) > 0:
                row = rows.iloc[0]
                room_type = str(local_constraint.get("room type") or "")
                if room_type == "shared room" and row["room type"] != "Shared room":
                    misses.append("room type:shared room")
                elif room_type == "private room" and row["room type"] != "Private room":
                    misses.append("room type:private room")
                elif room_type == "entire room" and row["room type"] != "Entire home/apt":
                    misses.append("room type:entire room")
                elif room_type == "not shared room" and row["room type"] == "Shared room":
                    misses.append("room type:not shared room")

                house_rule = str(local_constraint.get("house rule") or "")
                rules = str(row["house_rules"])
                if house_rule == "smoking" and "No smoking" in rules:
                    misses.append("house rule:smoking")
                elif house_rule == "parties" and "No parties" in rules:
                    misses.append("house rule:parties")
                elif house_rule == "children under 10" and "No children under 10" in rules:
                    misses.append("house rule:children under 10")
                elif house_rule == "visitors" and "No visitors" in rules:
                    misses.append("house rule:visitors")
                elif house_rule == "pets" and "No pets" in rules:
                    misses.append("house rule:pets")

        required_cuisines = local_constraint.get("cuisine") or []
        for meal_key in ["breakfast", "lunch", "dinner"]:
            meal = str(day.get(meal_key, "") or "")
            if not meal or meal == "-":
                continue
            _, city = get_valid_name_city(meal)
            if city == task.get("org"):
                continue
            rows = _lookup_restaurant(meal)
            if len(rows) == 0:
                continue
            cuisines = str(rows.iloc[0]["Cuisines"])
            for cuisine in required_cuisines:
                if cuisine in cuisines:
                    cuisines_seen.add(cuisine)

    for cuisine in local_constraint.get("cuisine") or []:
        if cuisine not in cuisines_seen:
            misses.append(f"cuisine:{cuisine}")

    unique_misses = []
    seen = set()
    for item in misses:
        if item not in seen:
            seen.add(item)
            unique_misses.append(item)
    return unique_misses


def _estimate_total_cost(task, plan):
    people = int(task.get("people_number", 1) or 1)
    total_cost = 0.0

    for day in plan:
        transport = _lookup_transport(day.get("transportation", "-"), day.get("current_city", ""))
        if transport:
            if transport["mode"] == "flight" and transport["price"] is not None:
                total_cost += float(transport["price"]) * people
            elif transport["mode"] == "self-driving" and transport["price"] is not None:
                total_cost += float(transport["price"]) * math.ceil(people / 5.0)
            elif transport["mode"] == "taxi" and transport["price"] is not None:
                total_cost += float(transport["price"]) * math.ceil(people / 4.0)

        for meal_key in ["breakfast", "lunch", "dinner"]:
            meal = str(day.get(meal_key, "") or "")
            if not meal or meal == "-":
                continue
            rows = _lookup_restaurant(meal)
            if len(rows) > 0:
                total_cost += float(rows.iloc[0]["Average Cost"]) * people

        accommodation = str(day.get("accommodation", "") or "")
        if accommodation and accommodation != "-":
            rows = _lookup_accommodation(accommodation)
            if len(rows) > 0:
                row = rows.iloc[0]
                total_cost += float(row["price"]) * math.ceil(people / float(row["maximum occupancy"]))

    return round(total_cost, 2)


def _city_trace(plan):
    trace = []
    for day in plan:
        current_city = str(day.get("current_city", "") or "").strip()
        if "from " in current_city and " to " in current_city:
            city1, city2 = extract_from_to(current_city)
            if city1:
                trace.append(city1)
            if city2:
                trace.append(city2)
        elif current_city:
            trace.append(extract_before_parenthesis(current_city))
    return trace


def _current_city_targets(current_city):
    current_city = str(current_city or "").strip()
    if "from " in current_city and " to " in current_city:
        city1, city2 = extract_from_to(current_city)
        return [city for city in [city1, city2] if city]
    city = extract_before_parenthesis(current_city)
    return [city] if city else []


def audit_plan(task, plan):
    city_state_map = _load_city_state_map()
    expected_days = int(task.get("days", 0) or 0)
    score = 100
    fatal_issues = []
    warnings = []
    invalid_entities = []
    invalid_transportations = []
    alignment_issues = []
    repeated_restaurants = []
    repeated_attractions = []
    min_nights_issues = []

    checks = {
        "day_count_ok": len(plan) == expected_days,
        "first_city_ok": True,
        "closed_loop_ok": True,
        "city_sequence_ok": True,
        "visiting_city_number_ok": True,
        "transportation_conflict": False,
        "budget_ok": True,
    }

    city_trace = _city_trace(plan)
    visited_non_origin = sorted({city for city in city_trace if city and city != task.get("org")})

    if len(plan) != expected_days:
        checks["day_count_ok"] = False
        fatal_issues.append(f"expected {expected_days} days but got {len(plan)}")
        score -= 25

    if city_trace:
        if city_trace[0] != task.get("org"):
            checks["first_city_ok"] = False
            fatal_issues.append(f"trip should start from {task.get('org')}")
            score -= 18
        if city_trace[0] != city_trace[-1]:
            checks["closed_loop_ok"] = False
            fatal_issues.append("trip is not a closed loop")
            score -= 15
        if not is_valid_city_sequence(city_trace):
            checks["city_sequence_ok"] = False
            fatal_issues.append("city sequence is not contiguous")
            score -= 12
        for city in city_trace:
            if city and city not in city_state_map:
                fatal_issues.append(f"unknown city: {city}")
                score -= 12
                break

    if len(visited_non_origin) != int(task.get("visiting_city_number", 1) or 1):
        checks["visiting_city_number_ok"] = False
        fatal_issues.append(
            f"expected visiting_city_number={task.get('visiting_city_number')} but got {len(visited_non_origin)}"
        )
        score -= 18

    transport_modes = set()
    seen_restaurants = set()
    seen_attractions = set()

    for day_idx, day in enumerate(plan, start=1):
        current_targets = _current_city_targets(day.get("current_city", ""))

        transport = str(day.get("transportation", "") or "").strip()
        if transport and transport != "-":
            mode = transportation_mode(transport)
            if mode:
                transport_modes.add(mode)
            if any(target not in transport for target in current_targets):
                alignment_issues.append(f"day {day_idx} transportation city mismatch")
            if _lookup_transport(transport, day.get("current_city", "")) is None:
                invalid_transportations.append(f"day {day_idx} invalid transportation")

        for meal_key in ["breakfast", "lunch", "dinner"]:
            meal = str(day.get(meal_key, "") or "").strip()
            if not meal or meal == "-":
                continue
            if current_targets and not any(target in meal for target in current_targets):
                alignment_issues.append(f"day {day_idx} {meal_key} city mismatch")
            if meal in seen_restaurants:
                repeated_restaurants.append(f"day {day_idx} repeated {meal_key}")
            else:
                seen_restaurants.add(meal)
            if len(_lookup_restaurant(meal)) == 0:
                invalid_entities.append(f"day {day_idx} invalid {meal_key}")

        attraction = str(day.get("attraction", "") or "").strip()
        for item in split_attractions(attraction):
            if current_targets and not any(target in item for target in current_targets):
                alignment_issues.append(f"day {day_idx} attraction city mismatch")
            if item in seen_attractions:
                repeated_attractions.append(f"day {day_idx} repeated attraction")
            else:
                seen_attractions.add(item)
            if len(_lookup_attraction(item)) == 0:
                invalid_entities.append(f"day {day_idx} invalid attraction")

        accommodation = str(day.get("accommodation", "") or "").strip()
        if accommodation and accommodation != "-":
            if current_targets and current_targets[-1] not in accommodation:
                alignment_issues.append(f"day {day_idx} accommodation city mismatch")
            if len(_lookup_accommodation(accommodation)) == 0:
                invalid_entities.append(f"day {day_idx} invalid accommodation")

    if ("self-driving" in transport_modes and "flight" in transport_modes) or (
        "self-driving" in transport_modes and "taxi" in transport_modes
    ):
        checks["transportation_conflict"] = True
        fatal_issues.append("mixed transportation modes conflict")
        score -= 18

    if invalid_transportations:
        fatal_issues.extend(invalid_transportations)
        score -= min(30, 12 * len(invalid_transportations))

    if invalid_entities:
        fatal_issues.extend(invalid_entities[:3])
        score -= min(30, 6 * len(invalid_entities))

    if alignment_issues:
        warnings.extend(alignment_issues[:5])
        score -= min(18, 4 * len(alignment_issues))

    if repeated_restaurants:
        warnings.extend(repeated_restaurants[:4])
        score -= min(16, 8 * len(repeated_restaurants))

    if repeated_attractions:
        warnings.extend(repeated_attractions[:4])
        score -= min(12, 6 * len(repeated_attractions))

    accommodations = [str(day.get("accommodation", "") or "").strip() for day in plan]
    for accommodation, nights in count_consecutive_values(accommodations):
        if not accommodation or accommodation == "-":
            continue
        rows = _lookup_accommodation(accommodation)
        if len(rows) == 0:
            continue
        minimum_nights = float(rows.iloc[0]["minimum nights"])
        if nights < minimum_nights:
            min_nights_issues.append(f"{accommodation} requires {int(minimum_nights)} nights")

    if min_nights_issues:
        fatal_issues.extend(min_nights_issues[:2])
        score -= min(24, 12 * len(min_nights_issues))

    local_constraint_misses = _check_local_constraints(task, plan)
    if local_constraint_misses:
        fatal_issues.extend(local_constraint_misses[:3])
        score -= min(24, 10 * len(local_constraint_misses))

    estimated_total_cost = _estimate_total_cost(task, plan)
    budget = _normalize_budget(task.get("budget"))
    if budget is not None and estimated_total_cost > budget:
        checks["budget_ok"] = False
        fatal_issues.append(f"estimated cost {estimated_total_cost} exceeds budget {budget}")
        score -= 14

    score = max(0, min(100, int(round(score))))
    likely_pass = (
        len(fatal_issues) == 0
        and len(invalid_entities) == 0
        and len(invalid_transportations) == 0
        and len(min_nights_issues) == 0
        and len(local_constraint_misses) == 0
        and checks["day_count_ok"]
        and checks["visiting_city_number_ok"]
        and not checks["transportation_conflict"]
        and checks["budget_ok"]
    )

    return {
        "score": score,
        "likely_pass": likely_pass,
        "fatal_count": len(fatal_issues),
        "warning_count": len(warnings),
        "checks": checks,
        "fatal_issues": fatal_issues[:10],
        "warnings": warnings[:10],
        "invalid_entity_count": len(invalid_entities),
        "invalid_transportation_count": len(invalid_transportations),
        "min_nights_issues": min_nights_issues,
        "local_constraint_misses": local_constraint_misses,
        "estimated_total_cost": estimated_total_cost,
        "visited_non_origin_cities": visited_non_origin,
        "transport_modes": sorted(transport_modes),
    }


# Preserve the published selector/repair behavior for exact historical replay.
audit_plan_legacy = audit_plan


def strict_issues(task, plan):
    """Independent preflight checks; this function never calls the final scorer."""
    issues = []
    def add(contract, message):
        issues.append({'contract': contract, 'issue': message})
    expected = int(task.get('days', 0))
    fields = {'current_city','transportation','breakfast','lunch','dinner','attraction','accommodation'}
    if len(plan) != expected:
        add('complete_information', 'plan day count differs from task')
    seen_meals = set(); seen_attractions = set()
    for i, day in enumerate(plan, 1):
        if not isinstance(day, dict) or not fields.issubset(day):
            add('complete_information', f'day {i}: missing required fields')
            continue
        if day.get('days', day.get('day')) != i:
            add('complete_information', f'day {i}: invalid day index')
        current = str(day.get('current_city') or '').strip()
        targets = _current_city_targets(current)
        travel = 'from ' in current.lower() or ' to ' in current.lower()
        if not current or current == '-':
            add('complete_information', f'day {i}: missing current city')
        missing = lambda key: not str(day.get(key) or '').strip() or str(day.get(key)).strip() == '-'
        if travel and missing('transportation'):
            add('complete_information', f'day {i}: missing transportation')
        if not travel:
            for key in ['breakfast','lunch','dinner','attraction']:
                if missing(key): add('complete_information', f'day {i}: missing {key}')
        if i < expected and missing('accommodation'):
            add('complete_information', f'day {i}: missing accommodation')
        transport = str(day.get('transportation') or '')
        restriction = (task.get('local_constraint') or {}).get('transportation')
        if restriction == 'no flight' and transportation_mode(transport) == 'flight':
            add('local_transportation', f'day {i}: flight prohibited')
        if restriction == 'no self-driving' and transportation_mode(transport) == 'self-driving':
            add('local_transportation', f'day {i}: self-driving prohibited')
        if not missing('transportation'):
            origin,dest = extract_from_to(transport)
            if not origin or not dest or any(t not in (origin,dest) for t in targets):
                add('current_city_alignment', f'day {i}: transportation city mismatch')
        for key in ['breakfast','lunch','dinner','accommodation']:
            if missing(key): continue
            value = str(day[key]); _,city = get_valid_name_city(value)
            allowed = targets[-1:] if key=='accommodation' else targets
            if not allowed or city.casefold() not in {c.casefold() for c in allowed}:
                add('current_city_alignment', f'day {i}: {key} city mismatch')
            if key!='accommodation':
                identity=value.casefold().strip()
                if identity in seen_meals:add('restaurant_diversity', f'day {i}: repeated {key}')
                seen_meals.add(identity)
        for value in split_attractions(day.get('attraction')):
            _,city=get_valid_name_city(value)
            if city.casefold() not in {c.casefold() for c in targets}:
                add('current_city_alignment', f'day {i}: attraction city mismatch')
            identity=value.casefold().strip()
            if identity in seen_attractions:add('attraction_diversity', f'day {i}: repeated attraction')
            seen_attractions.add(identity)
    return issues



@lru_cache(maxsize=20000)
def strict_entity_eligible(value, kind):
    lookup, name_col, city_col = {
        'restaurant': (_lookup_restaurant, 'Name', 'City'),
        'accommodation': (_lookup_accommodation, 'NAME', 'city'),
        'attraction': (_lookup_attraction, 'Name', 'City'),
    }[kind]
    name, city = get_valid_name_city(value)
    rows = lookup(value).dropna()
    if rows.empty:
        return False
    return bool(((rows[city_col] == city) & rows[name_col].astype(str).str.contains(re.escape(name))).any())


def audit_plan(task, plan, version=None):
    import os
    version = version or task.get('_audit_version') or os.getenv('TP_AUDIT_VERSION', 'strict-v2')
    if version not in {'legacy-v1','strict-v2'}:
        raise ValueError(f'Unknown audit version: {version}')
    result = audit_plan_legacy(task, plan)
    result['audit_version'] = version
    if version == 'legacy-v1':
        return result
    findings = strict_issues(task, plan)
    if int(task.get('days', 0)) > 3:
        mapping = _load_city_state_map()
        for city in _city_trace(plan)[1:-1]:
            if mapping.get(city) != task.get('dest'):
                findings.append({'contract':'destination_region','issue':f'{city} outside requested region'})
    # The released tools discard rows missing any database column. Keep that
    # eligibility boundary while retaining legacy-v1's historical lookups.
    for i, day in enumerate(plan, 1):
        for field, kind in [('breakfast','restaurant'),('lunch','restaurant'),('dinner','restaurant'),('accommodation','accommodation')]:
            value = str(day.get(field) or '').strip()
            if value and value != '-' and not strict_entity_eligible(value, kind):
                findings.append({'contract':'entity_eligibility','issue':f'day {i}: {field} not a case-sensitive eligible database match'})
        for value in split_attractions(day.get('attraction')):
            if not strict_entity_eligible(value, 'attraction'):
                findings.append({'contract':'entity_eligibility','issue':f'day {i}: attraction not a case-sensitive eligible database match'})
    contracts = ('complete_information','current_city_alignment','restaurant_diversity','attraction_diversity','local_transportation','destination_region','entity_eligibility')
    result['strict_checks'] = {name:not any(i['contract']==name for i in findings) for name in contracts}
    result['strict_issues'] = findings
    result['fatal_count'] += len(findings)
    result['fatal_issues'] = result['fatal_issues'] + [f"{i['contract']}: {i['issue']}" for i in findings]
    result['score'] = max(0, result['score'] - 8 * len(findings))
    result['likely_pass'] = result['likely_pass'] and not findings
    return result
