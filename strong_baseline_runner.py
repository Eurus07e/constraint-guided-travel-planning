import ast
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
STRATEGY = os.getenv("STRATEGY", "constraint_direct_json")
SET_TYPE = os.getenv("SET_TYPE", "validation").strip().lower()
MODE = "sole-planning"


def submission_file_path(set_type, model_name, strategy, mode):
    default = f"evaluation/{set_type}_{model_name}_{strategy}_{mode}_submission.jsonl"
    return Path(os.getenv("SUBMISSION_FILE", default))

OUT_ROOT = Path(os.getenv("OUT_ROOT", "outputs_strong_baselines")) / STRATEGY
OUTPUT_DIR = OUT_ROOT / SET_TYPE
DEBUG_DIR = OUT_ROOT / f"debug_{SET_TYPE}"
CACHE_DIR = OUT_ROOT / f"llm_cache_{SET_TYPE}"
SUBMISSION_FILE = submission_file_path(SET_TYPE, MODEL_NAME, STRATEGY, MODE)
CONSTRAINT_DIRECT_OUTPUT_DIR = Path("outputs_strong_baselines") / "constraint_direct_json" / SET_TYPE
DIRECT_SUBMISSION_FILE = Path("evaluation") / f"{SET_TYPE}_{MODEL_NAME}_constraint_direct_json_{MODE}_submission.jsonl"
PROMPT_VERSION = "strong_baseline_v2_json_schema"

RESUME = os.getenv("RESUME", "1") == "1"
MAX_REFERENCE_CHARS = int(os.getenv("MAX_REFERENCE_CHARS", "14000"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
MAX_WORKERS = max(1, int(os.getenv("MAX_WORKERS", "1")))
MAX_PLAN_TOKENS = int(os.getenv("MAX_PLAN_TOKENS", "4096"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

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
        lines = [" ".join(line.split()) for line in content.splitlines() if line.strip()]
        if len(lines) > 22:
            lines = lines[:22]
        content = "\n".join(lines)
        chunks.append(f"[{desc}]\n{content}")
    compact = "\n\n".join(chunks)
    if len(compact) <= max_chars:
        return compact
    # Preserve each section boundary before applying a final character cap.
    sections_out = []
    remaining = max_chars
    for chunk in chunks:
        if remaining <= 0:
            break
        piece = chunk[:remaining]
        sections_out.append(piece)
        remaining -= len(piece) + 2
    return "\n\n".join(sections_out)


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


def normalize_plan_list(payload, expected_days):
    if isinstance(payload, dict):
        payload = payload.get("plan") or payload.get("itinerary") or payload.get("days")
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


def extract_candidate_cities(ref):
    sections = safe_literal_eval(ref, [])
    cities = []
    if not isinstance(sections, list):
        return cities
    patterns = [
        ("Attractions in ", 1),
        ("Restaurants in ", 1),
        ("Accommodations in ", 1),
    ]
    for sec in sections:
        desc = str(sec.get("Description", ""))
        for prefix, _ in patterns:
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


def chat_completions_url(api_base):
    base = str(api_base).rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def call_llm(system_prompt, user_prompt, max_tokens=4096, temperature=0.0, cache_tag="default"):
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com").rstrip("/")
    model_name = os.getenv("MODEL_NAME", MODEL_NAME)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if "dashscope.aliyuncs.com" in api_base:
        payload["enable_thinking"] = os.getenv("ENABLE_THINKING", "0") == "1"
    cache_key = hashlib.sha256(
        json.dumps({"version": PROMPT_VERSION, "strategy": STRATEGY, "tag": cache_tag, "payload": payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = chat_completions_url(api_base)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
        "Connection": "close",
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            started_at = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            obj = response.json()
            message = obj["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            cache_path.write_text(content, encoding="utf-8")
            cache_path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "model": obj.get("model", model_name),
                        "usage": obj.get("usage"),
                        "latency_seconds": time.time() - started_at,
                        "attempt": attempt + 1,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return content
        except Exception as exc:
            last_err = exc
            time.sleep(min(45, 4 + attempt * 6))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def direct_prompt(task):
    system_prompt = (
        "You are a careful TravelPlanner baseline. Produce one complete day-level itinerary as JSON only. "
        "Use only entities and transportation present in the provided reference information. "
        "Do not write analysis, markdown, or prose outside the JSON object."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "constraint_checklist": [
                "The plan must have exactly task.days days.",
                "The route must start at org and return to org by the end of the trip.",
                "Use exactly visiting_city_number concrete destination cities; do not count a state name as a city.",
                "Every city transition needs matching transportation in the same direction.",
                "Restaurants, attractions, accommodations, and transportation must be from reference_information.",
                "Copy entity names exactly and append the city, e.g. 'Restaurant Name, City'.",
                "For flights, use the format 'Flight Number: F1234567, from A to B, Departure Time: HH:MM, Arrival Time: HH:MM'.",
                "For self-driving/taxi, include 'Self-driving, from A to B, Duration: ..., Cost: ...' or the exact transportation string from reference.",
                "For same-city days, use '-' for transportation.",
                "For travel days, current_city must be 'from ORIGIN to DESTINATION'.",
                "Meals, attractions, and accommodations must be in the current city.",
                "Avoid repeated restaurants and repeated attractions.",
                "Respect accommodation minimum nights, budget, room type, room rule, cuisine, and transportation constraints.",
                "Last-day Accommodation may be '-' only when returning to the origin city.",
                "Return a single JSON object with key 'plan'. No extra keys are needed.",
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
                ]
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def critique_prompt(task, plan):
    system_prompt = (
        "You are a generic self-refinement critic for TravelPlanner. "
        "Do not use external tools. Identify likely constraint violations and concise repair instructions."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "candidate_plan": plan,
            "instructions": [
                "Check route closure, exact day count, visiting city count, transportation direction, valid entities, city alignment, repeats, budget, local constraints, and minimum nights.",
                "Return JSON only.",
            ],
            "output_schema": {
                "issues": [
                    {
                        "constraint": "string",
                        "day": "integer or null",
                        "field": "string or null",
                        "repair_instruction": "string",
                    }
                ],
                "summary": "string",
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def revise_prompt(task, plan, critique):
    system_prompt = (
        "You are a generic self-refinement reviser for TravelPlanner. "
        "Revise the itinerary using the critique and reference information. Return JSON only."
    )
    user_prompt = json.dumps(
        {
            "task": task,
            "previous_plan": plan,
            "critique": critique,
            "instructions": [
                "Preserve correct parts of the previous plan.",
                "Repair only fields or route segments that are likely invalid.",
                "Use only entities and transportation from reference_information.",
                "Return a complete plan with exactly task.days days.",
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
                "change_summary": ["string"],
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def run_direct(task, idx):
    sp, up = direct_prompt(task)
    raw = call_llm(sp, up, max_tokens=MAX_PLAN_TOKENS, temperature=0.1, cache_tag=f"direct_{idx}")
    parsed = parse_json_payload(raw)
    plan = normalize_plan_list(parsed, int(task["days"]))
    fallback = None
    if schema_quality_bad(plan):
        fallback_plan = load_direct_baseline_plan(idx)
        if fallback_plan:
            plan = fallback_plan
            fallback = "direct_prompt_schema_fallback"
    return {
        "plan": plan,
        "raw": raw,
        "parsed": parsed,
        "fallback": fallback,
    }


def schema_quality_bad(plan):
    if not plan:
        return True
    for day in plan:
        for field in ["breakfast", "lunch", "dinner", "accommodation"]:
            value = str(day.get(field, "") or "")
            if value != "-" and "," not in value:
                return True
        for value in str(day.get("attraction", "") or "").split(";"):
            value = value.strip()
            if value and value != "-" and "," not in value:
                return True
        transportation = str(day.get("transportation", "") or "")
        if transportation not in {"", "-"} and "from" not in transportation:
            return True
    return False


def load_direct_baseline_plan(idx):
    if not hasattr(load_direct_baseline_plan, "_rows"):
        if DIRECT_SUBMISSION_FILE.exists():
            load_direct_baseline_plan._rows = [
                json.loads(line) for line in DIRECT_SUBMISSION_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        else:
            load_direct_baseline_plan._rows = []
    rows = load_direct_baseline_plan._rows
    if 0 < idx <= len(rows):
        plan = rows[idx - 1].get("plan")
        return plan if isinstance(plan, list) else None
    return None


def run_self_refine(task, idx):
    draft = load_constraint_direct_draft(idx) or run_direct(task, idx)
    draft_plan = draft["plan"]
    if not draft_plan:
        return {**draft, "critique": {}, "revised_raw": "", "fallback": "draft_parse_failed"}

    sp, up = critique_prompt(task, draft_plan)
    critique_raw = call_llm(sp, up, max_tokens=2048, temperature=0, cache_tag=f"critique_{idx}")
    critique = parse_json_payload(critique_raw) or {"summary": critique_raw, "issues": []}

    sp, up = revise_prompt(task, draft_plan, critique)
    revised_raw = call_llm(sp, up, max_tokens=MAX_PLAN_TOKENS, temperature=0.05, cache_tag=f"revise_{idx}")
    revised = parse_json_payload(revised_raw)
    revised_plan = normalize_plan_list(revised, int(task["days"]))
    if revised_plan:
        return {
            "plan": revised_plan,
            "raw": draft["raw"],
            "parsed": draft["parsed"],
            "critique": critique,
            "critique_raw": critique_raw,
            "revised_raw": revised_raw,
            "revised_parsed": revised,
            "fallback": None,
        }
    return {
        "plan": draft_plan,
        "raw": draft["raw"],
        "parsed": draft["parsed"],
        "critique": critique,
        "critique_raw": critique_raw,
        "revised_raw": revised_raw,
        "revised_parsed": revised,
        "fallback": "revised_parse_failed_used_draft",
    }


def load_constraint_direct_draft(idx):
    output_path = CONSTRAINT_DIRECT_OUTPUT_DIR / f"generated_plan_{idx}.json"
    if not output_path.exists():
        return None
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))[0]
        plan = payload.get(f"{MODEL_NAME}_constraint_direct_json_{MODE}_parsed_results")
    except Exception:
        return None
    if not isinstance(plan, list):
        return None
    return {
        "plan": plan,
        "raw": "",
        "parsed": {"plan": plan},
        "fallback": "reused_constraint_direct_json_draft",
    }


def run_one(row, idx):
    task = build_task(row)
    if STRATEGY == "constraint_direct_json":
        result = run_direct(task, idx)
    elif STRATEGY == "generic_self_refine_r1":
        result = run_self_refine(task, idx)
    else:
        raise ValueError(f"Unsupported STRATEGY for strong_baseline_runner: {STRATEGY}")

    debug = {
        "idx": idx,
        "strategy": STRATEGY,
        "task": task,
        "result": result,
        "status": "completed",
    }
    (DEBUG_DIR / f"debug_{idx}.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return result["plan"], debug


def select_rows(df):
    ids_env = os.getenv("IDS")
    if ids_env:
        ids = [int(x.strip()) for x in ids_env.split(",") if x.strip()]
        return df.iloc[[i - 1 for i in ids]]
    limit = os.getenv("LIMIT")
    if limit:
        df = df.head(int(limit))
    excluded_env = os.getenv("EXCLUDE_IDS", "")
    if excluded_env:
        excluded = {int(x.strip()) - 1 for x in excluded_env.split(",") if x.strip()}
        df = df.loc[[idx for idx in df.index if idx not in excluded]]
    return df


def load_records_dataframe():
    if SET_TYPE == "validation":
        return pd.read_csv(Path("database/validation.csv"))
    if SET_TYPE not in {"train", "test"}:
        raise ValueError(f"Unsupported SET_TYPE: {SET_TYPE}")
    from datasets import load_dataset
    from scripts.heldout_query_adapter import recover_evaluator_fields

    records = load_dataset("osunlp/TravelPlanner", SET_TYPE, download_mode="reuse_cache_if_exists")[SET_TYPE]
    rows = [recover_evaluator_fields(record) if SET_TYPE == "test" else dict(record) for record in records]
    return pd.DataFrame(rows)


def load_existing_plan(idx):
    output_path = OUTPUT_DIR / f"generated_plan_{idx}.json"
    debug_path = DEBUG_DIR / f"debug_{idx}.json"
    if not (output_path.exists() and debug_path.exists()):
        return None
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))[0]
        return payload.get(f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results")
    except Exception:
        return None


def main():
    ensure_dirs()
    df = load_records_dataframe()
    df_run = select_rows(df)
    key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_results"
    parsed_key = f"{MODEL_NAME}_{STRATEGY}_{MODE}_parsed_results"
    submission_rows = []

    def process(item):
        idx, row = item
        number = idx + 1
        plan = load_existing_plan(number) if RESUME else None
        if plan is not None:
            print(f"skipped {number}", flush=True)
        else:
            plan, _ = run_one(row, number)
            output_path = OUTPUT_DIR / f"generated_plan_{number}.json"
            output_path.write_text(
                json.dumps([{key: render_plan_text(plan), parsed_key: plan}], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"generated {number}", flush=True)
        return {"idx": number, "query": str(row.get("query", "")), "plan": plan}

    items = list(df_run.iterrows())
    if MAX_WORKERS == 1:
        submission_rows = [process(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(process, item) for item in items]
            submission_rows = [future.result() for future in as_completed(futures)]
        submission_rows.sort(key=lambda item: item["idx"])

    SUBMISSION_FILE.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in submission_rows) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(submission_rows)} plans into {OUTPUT_DIR}", flush=True)
    print(f"submission saved to {SUBMISSION_FILE}", flush=True)


if __name__ == "__main__":
    main()
