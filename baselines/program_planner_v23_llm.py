import ast
import json
import os
import time
import urllib.request
import urllib.error
import re
from io import StringIO
from itertools import permutations
from pathlib import Path

import pandas as pd
from utils.paths import DATABASE


MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
STRATEGY = "program_v23_llm"
MODE = "sole-planning"

DATA_PATH = DATABASE / "validation.csv"
OUTPUT_DIR = Path("outputs_program_v23_llm/validation")


def safe_literal_eval(x, default):
    try:
        if pd.isna(x):
            return default
        return ast.literal_eval(str(x))
    except Exception:
        return default


def unique_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_table_from_content(content):
    try:
        df = pd.read_fwf(StringIO(content))
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()


def get_reference_sections(ref):
    sections = safe_literal_eval(ref, [])
    if not isinstance(sections, list):
        return []
    return sections


def section_desc(section):
    return str(section.get("Description", ""))


def section_content(section):
    return str(section.get("Content", ""))


def extract_city_names(sections):
    cities = []

    patterns = [
        r"Attractions in (.+)",
        r"Restaurants in (.+)",
        r"Accommodations in (.+)",
        r"Flight from (.+?) to (.+?) on",
        r"Self-driving from (.+?) to (.+)",
        r"Taxi from (.+?) to (.+)",
    ]

    for s in sections:
        desc = section_desc(s)
        for pat in patterns:
            m = re.match(pat, desc)
            if not m:
                continue
            for g in m.groups():
                city = g.strip()
                city = re.sub(r"\s+on\s+.*$", "", city).strip()
                if city:
                    cities.append(city)

    return unique_keep_order(cities)


def find_section_exact_prefix(sections, prefix):
    for s in sections:
        desc = section_desc(s)
        if desc.startswith(prefix):
            return section_content(s)
    return ""


def find_transport_section(sections, method, origin, dest, date=None):
    if method == "flight":
        target = f"Flight from {origin} to {dest}"
    elif method == "self-driving":
        target = f"Self-driving from {origin} to {dest}"
    elif method == "taxi":
        target = f"Taxi from {origin} to {dest}"
    else:
        return ""

    candidates = []
    for s in sections:
        desc = section_desc(s)
        if target in desc:
            if date is None or str(date) in desc or method != "flight":
                candidates.append(section_content(s))

    if candidates:
        return candidates[0]
    return ""


def parse_flights(content):
    if not content or "There is no flight" in content:
        return []

    df = read_table_from_content(content)
    flights = []

    if "Flight Number" in df.columns:
        for _, row in df.iterrows():
            fn = str(row.get("Flight Number", "")).strip()
            if not fn or fn.lower() == "flight number":
                continue

            try:
                price = float(row.get("Price", 999999))
            except Exception:
                price = 999999.0

            flights.append({
                "method": "flight",
                "flight_number": fn,
                "cost": price,
                "dep": str(row.get("DepTime", "")).strip(),
                "arr": str(row.get("ArrTime", "")).strip(),
            })

    return sorted(flights, key=lambda x: x["cost"])


def parse_ground_transport(content, method):
    if not content:
        return None

    if "There is no" in content:
        return None

    cost_match = re.search(r"cost:\s*([0-9.]+)", content, re.I)
    duration_match = re.search(r"duration:\s*([^,]+)", content, re.I)

    try:
        cost = float(cost_match.group(1)) if cost_match else 999999.0
    except Exception:
        cost = 999999.0

    duration = duration_match.group(1).strip() if duration_match else ""

    return {
        "method": method,
        "cost": cost,
        "duration": duration,
        "raw": content,
    }


def format_money(x):
    try:
        x = float(x)
        return str(int(x)) if x.is_integer() else str(round(x, 2))
    except Exception:
        return str(x)


def format_transport(option, origin, dest):
    if option is None:
        return "-"

    method = option.get("method")

    if method == "flight":
        return (
            f"Flight Number: {option.get('flight_number')}, "
            f"from {origin} to {dest}, "
            f"Departure Time: {option.get('dep')}, "
            f"Arrival Time: {option.get('arr')}"
        )

    if method == "self-driving":
        return (
            f"Self-driving, from {origin} to {dest}, "
            f"Duration: {option.get('duration')}, "
            f"Cost: {format_money(option.get('cost', 0))}"
        )

    if method == "taxi":
        return (
            f"Taxi, from {origin} to {dest}, "
            f"Duration: {option.get('duration')}, "
            f"Cost: {format_money(option.get('cost', 0))}"
        )

    return "-"


def get_transport_options(sections, origin, dest, date=None, required_method=None):
    options = []

    flight_content = find_transport_section(sections, "flight", origin, dest, date)
    for f in parse_flights(flight_content):
        options.append(f)

    sd_content = find_transport_section(sections, "self-driving", origin, dest, date)
    sd = parse_ground_transport(sd_content, "self-driving")
    if sd:
        options.append(sd)

    taxi_content = find_transport_section(sections, "taxi", origin, dest, date)
    taxi = parse_ground_transport(taxi_content, "taxi")
    if taxi:
        options.append(taxi)

    if required_method:
        req = str(required_method).lower()
        filtered = [x for x in options if req in x["method"].lower()]
        if filtered:
            options = filtered

    return sorted(options, key=lambda x: x.get("cost", 999999))


def choose_transport(sections, origin, dest, date=None, required_method=None):
    options = get_transport_options(sections, origin, dest, date, required_method)
    if options:
        return options[0]
    return None


def parse_attractions(sections, city):
    content = find_section_exact_prefix(sections, f"Attractions in {city}")
    df = read_table_from_content(content)

    names = []
    if "Name" in df.columns:
        for x in df["Name"].dropna().tolist():
            name = str(x).strip()
            if name and name.lower() != "name":
                names.append(name)

    if not names:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines[1:]:
            parts = re.split(r"\s{2,}", line)
            if parts:
                names.append(parts[0].strip())

    return unique_keep_order(names)


def parse_restaurants(sections, city):
    content = find_section_exact_prefix(sections, f"Restaurants in {city}")
    df = read_table_from_content(content)

    items = []

    if "Name" in df.columns:
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            cuisines = str(row.get("Cuisines", "")).strip()
            try:
                cost = float(row.get("Average Cost", 50))
            except Exception:
                cost = 50.0

            if not name or name.lower() == "name":
                continue

            items.append({
                "name": name,
                "city": city,
                "cuisines": cuisines,
                "cost": cost,
            })

    if not items:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines[1:]:
            parts = re.split(r"\s{2,}", line)
            if parts:
                items.append({
                    "name": parts[0].strip(),
                    "city": city,
                    "cuisines": "",
                    "cost": 50.0,
                })

    items = sorted(items, key=lambda x: x["cost"])
    out = []
    seen = set()
    for item in items:
        if item["name"] not in seen:
            seen.add(item["name"])
            out.append(item)

    return out


def normalize_terms(x):
    if not x:
        return []
    if isinstance(x, list):
        return [str(v).lower() for v in x]
    return [str(x).lower()]


def restaurant_matches_cuisine(item, required_cuisine):
    terms = normalize_terms(required_cuisine)
    if not terms:
        return True
    cuisines = item.get("cuisines", "").lower()
    return any(t in cuisines for t in terms)


def parse_accommodations(sections, city):
    content = find_section_exact_prefix(sections, f"Accommodations in {city}")
    df = read_table_from_content(content)

    items = []

    if "NAME" in df.columns:
        for _, row in df.iterrows():
            name = str(row.get("NAME", "")).strip()
            if not name or name.lower() == "name":
                continue

            try:
                price = float(row.get("price", 999999))
            except Exception:
                price = 999999.0

            try:
                min_nights = float(row.get("minimum nights", 1))
            except Exception:
                min_nights = 1.0

            try:
                max_occ = float(row.get("maximum occupancy", 1))
            except Exception:
                max_occ = 1.0

            items.append({
                "name": name,
                "city": city,
                "room_type": str(row.get("room type", "")).strip(),
                "price": price,
                "minimum_nights": min_nights,
                "house_rules": str(row.get("house_rules", "")).strip(),
                "maximum_occupancy": max_occ,
            })

    return sorted(items, key=lambda x: x["price"])


def accommodation_matches(item, constraints, people, nights):
    if item is None:
        return False

    if item.get("maximum_occupancy", 0) < people:
        return False

    if item.get("minimum_nights", 1) > max(1, nights):
        return False

    room_type = constraints.get("room_type") or constraints.get("room type")
    if room_type:
        terms = normalize_terms(room_type)
        value = item.get("room_type", "").lower()
        if not any(t in value for t in terms):
            return False

    room_rule = constraints.get("room_rule") or constraints.get("house_rule") or constraints.get("house rule")
    if room_rule:
        terms = normalize_terms(room_rule)
        value = item.get("house_rules", "").lower()
        if not any(t in value for t in terms):
            return False

    return True


def choose_accommodation(items, constraints, people, nights):
    good = [x for x in items if accommodation_matches(x, constraints, people, nights)]
    if good:
        return good[0]

    occupancy_good = [x for x in items if x.get("maximum_occupancy", 0) >= people]
    if occupancy_good:
        return occupancy_good[0]

    if items:
        return items[0]

    return {
        "name": "-",
        "city": "",
        "price": 0,
        "minimum_nights": 1,
        "maximum_occupancy": 999,
        "room_type": "",
        "house_rules": "",
    }


def extract_candidates(sections):
    cities = extract_city_names(sections)

    restaurants = {}
    attractions = {}
    accommodations = {}

    for city in cities:
        rs = parse_restaurants(sections, city)
        ats = parse_attractions(sections, city)
        acc = parse_accommodations(sections, city)

        if rs:
            restaurants[city] = rs
        if ats:
            attractions[city] = ats
        if acc:
            accommodations[city] = acc

    usable_cities = []
    for c in cities:
        if restaurants.get(c) and attractions.get(c) and accommodations.get(c):
            usable_cities.append(c)

    return {
        "cities": unique_keep_order(cities),
        "usable_cities": unique_keep_order(usable_cities),
        "restaurants": restaurants,
        "attractions": attractions,
        "accommodations": accommodations,
    }


def python_normalize_constraints(row):
    base = safe_literal_eval(row.get("local_constraint", "{}"), {})
    if not isinstance(base, dict):
        base = {}

    normalized = {}
    for k, v in base.items():
        key = str(k).strip().lower().replace(" ", "_")
        normalized[key] = v
    return normalized


def extract_json_object(text):
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def call_deepseek_json(system_prompt, user_prompt, max_tokens=2048, temperature=0):
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com").rstrip("/")
    model_name = os.getenv("MODEL_NAME", MODEL_NAME)

    if not api_key or not api_key.startswith("sk-"):
        return {}

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_base + "/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        },
        method="POST"
    )

    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                content = obj["choices"][0]["message"]["content"]
                return extract_json_object(content)
        except Exception:
            time.sleep(2)

    return {}


def llm_interpret_constraints(row):
    normalized = python_normalize_constraints(row)

    if os.getenv("USE_LLM", "1") != "1":
        return normalized

    system_prompt = (
        "You are a constraint interpreter for a travel planning benchmark. "
        "Return only one valid JSON object. Do not write markdown. "
        "Do not invent entities. Only normalize constraints."
    )

    user_prompt = json.dumps({
        "task": "Extract executable travel constraints from the query and local_constraint.",
        "query": str(row.get("query", "")),
        "local_constraint": str(row.get("local_constraint", "")),
        "output_schema": {
            "cuisine": "string or list or null",
            "transportation": "flight, self-driving, taxi, or null",
            "room_type": "string or null",
            "room_rule": "string or null",
            "priority": "list of important constraints",
            "notes": "short note"
        }
    }, ensure_ascii=False)

    llm_obj = call_deepseek_json(system_prompt, user_prompt, max_tokens=1024, temperature=0)

    for key in ["cuisine", "transportation", "room_type", "room_rule"]:
        value = llm_obj.get(key)
        if value not in [None, "", [], {}]:
            normalized[key] = value

    if "priority" in llm_obj:
        normalized["priority"] = llm_obj.get("priority")

    if "notes" in llm_obj:
        normalized["llm_notes"] = llm_obj.get("notes")

    return normalized


def route_score(sections, org, route, dates, required_transport):
    if not route:
        return -10**9, []

    full = [org] + route + [org]
    legs = []
    score = 0

    for i in range(len(full) - 1):
        origin = full[i]
        dest = full[i + 1]
        date = dates[min(i, len(dates) - 1)] if dates else None
        option = choose_transport(sections, origin, dest, date, required_transport)
        if option is None:
            score -= 10000
            legs.append(None)
        else:
            score += 1000
            score -= option.get("cost", 0)
            legs.append(option)

    return score, legs


def plan_route(sections, constraints, candidates):
    org = constraints["org"]
    dest = constraints["dest"]
    dates = constraints["dates"]
    visiting_city_number = int(constraints.get("visiting_city_number", 1))
    required_transport = constraints.get("transportation")

    usable = unique_keep_order([c for c in candidates["usable_cities"] if c != org])

    if not usable:
        usable = unique_keep_order([c for c in candidates["cities"] if c != org])

    k = max(1, min(visiting_city_number, len(usable), constraints["days"] - 1))

    route_candidates = []

    if dest in usable:
        must = [dest]
        optional = [c for c in usable if c != dest]

        if k == 1:
            route_candidates = [[dest]]
        else:
            for extra in permutations(optional, k - 1):
                route_candidates.append(unique_keep_order(must + list(extra)))
            for extra in permutations(optional, k - 1):
                route_candidates.append(unique_keep_order(list(extra) + must))
    else:
        for route in permutations(usable, k):
            route_candidates.append(list(route))

    if not route_candidates and usable:
        route_candidates = [usable[:k]]

    best_route = None
    best_legs = None
    best_score = -10**18

    for route in route_candidates[:200]:
        if len(route) != k:
            continue

        score, legs = route_score(sections, org, route, dates, required_transport)

        city_quality = 0
        for c in route:
            city_quality += 100 if candidates["restaurants"].get(c) else -100
            city_quality += 100 if candidates["attractions"].get(c) else -100
            city_quality += 100 if candidates["accommodations"].get(c) else -100

        score += city_quality

        if score > best_score:
            best_score = score
            best_route = route
            best_legs = legs

    if best_route is None:
        fallback_city = usable[0] if usable else dest
        best_route = [fallback_city]
        _, best_legs = route_score(sections, org, best_route, dates, required_transport)

    return best_route, best_legs


def allocate_days(route, days):
    k = len(route)

    if k <= 1:
        return [route[0]] * days

    daily = [None] * days

    transition_days = [0]
    if k > 1:
        for j in range(1, k):
            pos = round(j * (days - 1) / k)
            pos = max(1, min(days - 2, pos))
            transition_days.append(pos)

    for i, start_day in enumerate(transition_days):
        end_day = transition_days[i + 1] if i + 1 < len(transition_days) else days
        for d in range(start_day, end_day):
            daily[d] = route[i]

    for i in range(days):
        if daily[i] is None:
            daily[i] = route[-1]

    return daily


def count_nights_by_city(daily_city):
    nights = {}
    for city in daily_city[:-1]:
        nights[city] = nights.get(city, 0) + 1
    return nights


def pick_restaurant(city, candidates, used, required_cuisine=None):
    items = candidates["restaurants"].get(city, [])

    preferred = [x for x in items if x["name"] not in used and restaurant_matches_cuisine(x, required_cuisine)]
    if preferred:
        item = preferred[0]
        used.add(item["name"])
        return item

    fallback = [x for x in items if x["name"] not in used]
    if fallback:
        item = fallback[0]
        used.add(item["name"])
        return item

    if items:
        return items[0]

    return {"name": "-", "city": city, "cost": 0, "cuisines": ""}


def pick_attractions(city, candidates, used, count=2):
    items = candidates["attractions"].get(city, [])
    chosen = []

    for name in items:
        if name not in used:
            used.add(name)
            chosen.append(name)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        for name in items:
            if name not in chosen:
                chosen.append(name)
            if len(chosen) >= count:
                break

    if not chosen:
        chosen = ["-"]

    return chosen


def build_plan(row, constraints, sections, candidates, route, route_legs):
    org = constraints["org"]
    days = constraints["days"]
    dates = constraints["dates"]
    people = constraints["people"]
    required_cuisine = constraints.get("cuisine")
    budget = constraints["budget"]

    daily_city = allocate_days(route, days)
    nights_by_city = count_nights_by_city(daily_city)

    accommodations = {}
    for city, nights in nights_by_city.items():
        items = candidates["accommodations"].get(city, [])
        accommodations[city] = choose_accommodation(items, constraints, people, nights)

    used_restaurants = set()
    used_attractions = set()

    plan = []
    cost = {
        "transportation": 0.0,
        "restaurants": 0.0,
        "accommodations": 0.0,
        "total": 0.0,
        "budget": budget,
    }

    current_origin = org
    leg_index = 0

    for d in range(days):
        day_num = d + 1
        date = dates[d] if d < len(dates) else ""

        stay_city = daily_city[d]
        previous_city = org if d == 0 else daily_city[d - 1]

        if d == 0:
            from_city = org
            to_city = stay_city
            transport = route_legs[0] if route_legs and len(route_legs) > 0 else choose_transport(sections, from_city, to_city, date, constraints.get("transportation"))
            current_city_text = f"from {from_city} to {to_city}"
        elif d == days - 1:
            from_city = daily_city[d - 1]
            to_city = org
            leg_pos = len(route)
            transport = route_legs[leg_pos] if route_legs and len(route_legs) > leg_pos else choose_transport(sections, from_city, to_city, date, constraints.get("transportation"))
            current_city_text = f"from {from_city} to {to_city}"
            stay_city = from_city
        elif daily_city[d] != daily_city[d - 1]:
            from_city = daily_city[d - 1]
            to_city = daily_city[d]
            route_pos = route.index(to_city) if to_city in route else 0
            transport = route_legs[route_pos] if route_legs and len(route_legs) > route_pos else choose_transport(sections, from_city, to_city, date, constraints.get("transportation"))
            current_city_text = f"from {from_city} to {to_city}"
        else:
            from_city = stay_city
            to_city = stay_city
            transport = None
            current_city_text = stay_city

        if transport:
            cost["transportation"] += transport.get("cost", 0.0)

        breakfast = pick_restaurant(stay_city, candidates, used_restaurants, required_cuisine)
        lunch = pick_restaurant(stay_city, candidates, used_restaurants, required_cuisine)
        dinner = pick_restaurant(stay_city, candidates, used_restaurants, required_cuisine)

        cost["restaurants"] += breakfast.get("cost", 0.0) + lunch.get("cost", 0.0) + dinner.get("cost", 0.0)

        attractions = pick_attractions(stay_city, candidates, used_attractions, 2)

        if d == days - 1:
            accommodation = {"name": "-", "city": stay_city, "price": 0}
        else:
            accommodation = accommodations.get(stay_city, {"name": "-", "city": stay_city, "price": 0})
            cost["accommodations"] += accommodation.get("price", 0.0)

        plan.append({
            "day": day_num,
            "date": date,
            "stay_city": stay_city,
            "current_city": current_city_text,
            "transportation_obj": transport,
            "transportation": format_transport(transport, from_city, to_city) if transport else "-",
            "breakfast_obj": breakfast,
            "lunch_obj": lunch,
            "dinner_obj": dinner,
            "breakfast": breakfast["name"],
            "lunch": lunch["name"],
            "dinner": dinner["name"],
            "attractions": attractions,
            "accommodation_obj": accommodation,
            "accommodation": accommodation["name"],
        })

    cost["total"] = cost["transportation"] + cost["restaurants"] + cost["accommodations"]

    return {
        "constraints": constraints,
        "route": route,
        "daily_city": daily_city,
        "plan": plan,
        "cost": cost,
    }


def validate_plan(state, candidates):
    failures = []
    constraints = state["constraints"]
    plan = state["plan"]
    budget = constraints["budget"]

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
        if "from " in day["current_city"] and day["transportation"] == "-":
            failures.append({
                "type": "missing_transportation",
                "day": day["day"],
                "message": f"City changes on Day {day['day']} but transportation is missing."
            })

    for day in plan:
        city = day["stay_city"]

        for meal_key in ["breakfast_obj", "lunch_obj", "dinner_obj"]:
            meal = day.get(meal_key, {})
            if meal.get("name") != "-" and meal.get("city") != city:
                failures.append({
                    "type": "wrong_city_restaurant",
                    "day": day["day"],
                    "message": f"Restaurant city mismatch on Day {day['day']}"
                })

        acc = day.get("accommodation_obj", {})
        if day["day"] != len(plan) and acc.get("name") != "-" and acc.get("city") != city:
            failures.append({
                "type": "wrong_city_accommodation",
                "day": day["day"],
                "message": f"Accommodation city mismatch on Day {day['day']}"
            })

    if state["cost"]["total"] > budget:
        failures.append({
            "type": "budget",
            "message": f"Total cost {state['cost']['total']} exceeds budget {budget}"
        })

    return failures


def deterministic_repair(state, candidates, sections):
    failures = validate_plan(state, candidates)
    if not failures:
        return state

    constraints = state["constraints"]

    for fail in failures:
        if fail["type"] == "budget":
            for day in state["plan"]:
                if day["day"] == len(state["plan"]):
                    continue
                city = day["stay_city"]
                items = candidates["accommodations"].get(city, [])
                cheaper = choose_accommodation(items, {}, constraints["people"], 1)
                if cheaper and cheaper.get("price", 999999) < day["accommodation_obj"].get("price", 999999):
                    day["accommodation_obj"] = cheaper
                    day["accommodation"] = cheaper["name"]

        if fail["type"] == "missing_transportation":
            day_idx = fail["day"] - 1
            day = state["plan"][day_idx]
            text = day["current_city"]
            m = re.match(r"from (.+) to (.+)", text)
            if m:
                origin, dest = m.group(1), m.group(2)
                date = day.get("date")
                transport = choose_transport(sections, origin, dest, date, constraints.get("transportation"))
                if transport:
                    day["transportation_obj"] = transport
                    day["transportation"] = format_transport(transport, origin, dest)

    recalc_cost(state)
    return state


def recalc_cost(state):
    cost = {
        "transportation": 0.0,
        "restaurants": 0.0,
        "accommodations": 0.0,
        "budget": state["constraints"]["budget"],
    }

    for day in state["plan"]:
        t = day.get("transportation_obj")
        if t:
            cost["transportation"] += t.get("cost", 0.0)

        for key in ["breakfast_obj", "lunch_obj", "dinner_obj"]:
            item = day.get(key)
            if item:
                cost["restaurants"] += item.get("cost", 0.0)

        if day["day"] != len(state["plan"]):
            acc = day.get("accommodation_obj")
            if acc:
                cost["accommodations"] += acc.get("price", 0.0)

    cost["total"] = cost["transportation"] + cost["restaurants"] + cost["accommodations"]
    state["cost"] = cost



def summarize_state_for_llm(state, failures):
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

    return {
        "constraints": state["constraints"],
        "route": state["route"],
        "daily_city": state["daily_city"],
        "cost": state["cost"],
        "failures": failures,
        "plan": plan_summary
    }


def llm_repair_advisor(row, state, failures):
    if os.getenv("USE_LLM", "1") != "1":
        return []

    system_prompt = (
        "You are a repair advisor for a Python travel planner. "
        "You must not invent restaurants, hotels, attractions, or flights. "
        "Return only one valid JSON object. "
        "Your job is to choose repair actions, not to rewrite the itinerary."
    )

    user_prompt = json.dumps({
        "task": "Given the current plan and validator failures, choose executable repair actions.",
        "allowed_actions": [
            "replace_accommodation",
            "replace_transportation",
            "insert_transportation",
            "replace_restaurant",
            "enforce_cuisine",
            "reduce_cost",
            "keep"
        ],
        "action_schema": {
            "actions": [
                {
                    "action": "one allowed action",
                    "day": "integer or null",
                    "target_city": "string or null",
                    "reason": "short reason"
                }
            ]
        },
        "query": str(row.get("query", "")),
        "state": summarize_state_for_llm(state, failures)
    }, ensure_ascii=False)

    obj = call_deepseek_json(system_prompt, user_prompt, max_tokens=2048, temperature=0)

    actions = obj.get("actions", [])
    if not isinstance(actions, list):
        return []

    clean = []
    allowed = {
        "replace_accommodation",
        "replace_transportation",
        "insert_transportation",
        "replace_restaurant",
        "enforce_cuisine",
        "reduce_cost",
        "keep"
    }

    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("action") in allowed:
            clean.append(a)

    return clean[:5]


def apply_repair_actions(state, candidates, sections, actions):
    constraints = state["constraints"]

    for action in actions:
        name = action.get("action")
        day_value = action.get("day")

        if name in ["keep", None]:
            continue

        target_days = []
        if isinstance(day_value, int) and 1 <= day_value <= len(state["plan"]):
            target_days = [state["plan"][day_value - 1]]
        else:
            target_days = state["plan"]

        if name in ["replace_accommodation", "reduce_cost"]:
            for day in target_days:
                if day["day"] == len(state["plan"]):
                    continue
                city = day["stay_city"]
                items = candidates["accommodations"].get(city, [])
                if not items:
                    continue
                current_price = day.get("accommodation_obj", {}).get("price", 999999)
                cheaper = [x for x in items if x.get("price", 999999) <= current_price]
                if not cheaper:
                    cheaper = items
                chosen = choose_accommodation(cheaper, constraints, constraints["people"], 1)
                if chosen:
                    day["accommodation_obj"] = chosen
                    day["accommodation"] = chosen["name"]

        if name in ["replace_transportation", "insert_transportation"]:
            for day in target_days:
                text = day.get("current_city", "")
                m = re.match(r"from (.+) to (.+)", text)
                if not m:
                    continue
                origin, dest = m.group(1), m.group(2)
                transport = choose_transport(sections, origin, dest, day.get("date"), constraints.get("transportation"))
                if transport:
                    day["transportation_obj"] = transport
                    day["transportation"] = format_transport(transport, origin, dest)

        if name in ["replace_restaurant", "enforce_cuisine", "reduce_cost"]:
            used = set()
            for d in state["plan"]:
                for k in ["breakfast", "lunch", "dinner"]:
                    if d.get(k) and d.get(k) != "-":
                        used.add(d[k])

            for day in target_days:
                city = day["stay_city"]
                for meal_name, obj_name in [
                    ("breakfast", "breakfast_obj"),
                    ("lunch", "lunch_obj"),
                    ("dinner", "dinner_obj")
                ]:
                    item = pick_restaurant(city, candidates, used, constraints.get("cuisine"))
                    day[obj_name] = item
                    day[meal_name] = item["name"]

    recalc_cost(state)
    return state


def save_debug_log(number, row, state, failures_before, actions, failures_after):
    debug_dir = Path("outputs_program_v23_llm/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "idx": number,
        "query": str(row.get("query", "")),
        "constraints": state.get("constraints", {}),
        "route": state.get("route", []),
        "cost": state.get("cost", {}),
        "failures_before": failures_before,
        "llm_actions": actions,
        "failures_after": failures_after
    }

    (debug_dir / f"debug_{number}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )



def render_plan_text(state):
    lines = []
    lines.append("Travel Plan:")

    for day in state["plan"]:
        lines.append("")
        lines.append(f"Day {day['day']}:")
        lines.append(f"Current City: {day['current_city']}")
        lines.append(f"Transportation: {day['transportation']}")
        lines.append(f"Breakfast: {day['breakfast']}, {day['stay_city']}" if day["breakfast"] != "-" else "Breakfast: -")
        attraction_text = ";".join([f"{a}, {day['stay_city']}" for a in day["attractions"] if a != "-"]) + ";"
        lines.append(f"Attraction: {attraction_text if attraction_text != ';' else '-'}")
        lines.append(f"Lunch: {day['lunch']}, {day['stay_city']}" if day["lunch"] != "-" else "Lunch: -")
        lines.append(f"Dinner: {day['dinner']}, {day['stay_city']}" if day["dinner"] != "-" else "Dinner: -")
        lines.append(f"Accommodation: {day['accommodation']}, {day['stay_city']}" if day["accommodation"] != "-" else "Accommodation: -")

    return "\n".join(lines)


def generate_for_row(row, number=None):
    sections = get_reference_sections(row["reference_information"])

    constraints = llm_interpret_constraints(row)
    constraints["org"] = str(row["org"])
    constraints["dest"] = str(row["dest"])
    constraints["days"] = int(row["days"])
    constraints["dates"] = safe_literal_eval(row["date"], [])
    constraints["people"] = int(row["people_number"])
    constraints["budget"] = float(row["budget"])
    constraints["visiting_city_number"] = int(row["visiting_city_number"])

    candidates = extract_candidates(sections)
    route, route_legs = plan_route(sections, constraints, candidates)

    state = build_plan(row, constraints, sections, candidates, route, route_legs)

    all_actions = []
    first_failures = validate_plan(state, candidates)

    for _ in range(2):
        failures = validate_plan(state, candidates)
        actions = llm_repair_advisor(row, state, failures)
        all_actions.extend(actions)
        if actions:
            state = apply_repair_actions(state, candidates, sections, actions)
        state = deterministic_repair(state, candidates, sections)

        if not validate_plan(state, candidates):
            break

    final_failures = validate_plan(state, candidates)

    if number is not None:
        save_debug_log(number, row, state, first_failures, all_actions, final_failures)

    return render_plan_text(state)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    ids_env = os.getenv("IDS")
    if ids_env:
        ids = [int(x.strip()) for x in ids_env.split(",") if x.strip()]
        df = df.iloc[[i - 1 for i in ids]]
    else:
        limit = os.getenv("LIMIT")
        if limit:
            df = df.head(int(limit))

    key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_results"

    count = 0
    for idx, row in df.iterrows():
        number = idx + 1
        text = generate_for_row(row, number)
        out = [{key: text}]
        out_path = OUTPUT_DIR / f"generated_plan_{number}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    print(f"generated {count} plans into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
