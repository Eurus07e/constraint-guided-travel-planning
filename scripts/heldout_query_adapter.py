"""Recover evaluator fields omitted from the public TravelPlanner test split.

The public test configuration exposes the request text but omits fields that
are present in train/validation.  This adapter is intentionally deterministic:
it never asks a model to infer a constraint.  Call ``validate_against_split``
before using it for held-out scoring.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from typing import Any


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
_NUMBER = r"(\d+|one|two|three|four|five|six|seven|eight)"
_CUISINES = ("American", "French", "Chinese", "Indian", "Italian", "Mediterranean", "Mexican")


def _number(value: str) -> int:
    return int(value) if value.isdigit() else _NUMBER_WORDS[value.lower()]


def _first_number(query: str, patterns: Iterable[str], default: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return _number(match.group(1))
    return default


def _parse_people(query: str) -> int:
    patterns = (
        rf"\b{_NUMBER}\s+people\b",
        rf"\b{_NUMBER}\s+persons?\b",
        rf"\b{_NUMBER}\s+travel(?:er|lers|l)?\b",
        rf"\b{_NUMBER}\s+individuals?\b",
        rf"\b{_NUMBER}\s+participants?\b",
        rf"\bgroup\s+of\s+{_NUMBER}\b",
        rf"\bparty\s+of\s+{_NUMBER}\b",
    )
    explicit = _first_number(query, patterns, default=0)
    if explicit:
        return explicit
    lowered = query.lower()
    if re.search(r"\b(two|a)\s+(?:people|travelers?|individuals?)\b", lowered):
        return 2
    if re.search(r"\b(?:for|plan for|trip for|itinerary for)\s+(?:two|2)\b", lowered):
        return 2
    if re.search(r"\b(?:a\s+)?(?:pair|duo|couple)\b", lowered):
        return 2
    if re.search(r"\b(?:solo|single traveler|one person|an individual|travel alone)\b", lowered):
        return 1
    if re.search(r"\b(?:we|our|ourselves)\b", lowered):
        return 2
    return 1


def _parse_city_count(query: str, days: int | None = None) -> int:
    if days in {3, 5, 7}:
        return {3: 1, 5: 2, 7: 3}[days]
    return _first_number(
        query,
        (rf"\b{_NUMBER}\s+(?:(?:different|distinct|unique)\s+)?cit(?:y|ies)\b",),
        default=1,
    )


def _parse_house_rule(query: str) -> str | None:
    lowered = query.lower()
    if re.search(r"\b(?:children?|child-friendly|children-friendly|younger ones)\b", lowered):
        return "children under 10"
    if re.search(r"\bvisitors?\b", lowered):
        return "visitors"
    if re.search(r"\bsmok(?:e|ing|ers?)\w*\b", lowered):
        return "smoking"
    if re.search(r"\bpet-friendly\b|\bpets?\b", lowered):
        return "pets"
    if re.search(r"\bparties\b", lowered):
        return "parties"
    patterns = (
        ("children under 10", ("children under 10", "children under the age of 10", "child-friendly", "children-friendly", "suitable for children", "permit children", "accommodate children", "younger ones")),
        ("parties", ("allow parties", "parties are allowed", "party-friendly", "do not restrict parties", "parties are permitted")),
        ("visitors", ("allow visitors", "visitor-friendly", "visitors-friendly", "welcome visitors", "visitors are allowed", "permit visitors", "visitors restrictions")),
        ("smoking", ("permit smoking", "allow smoking", "smoking is permitted", "smoking areas", "smoking house rules", "smoking is a necessity", "smoking-friendly", "smoking is allowed")),
        ("pets", ("pet-friendly", "allow pets", "pets are allowed")),
    )
    for value, phrases in patterns:
        if any(phrase in lowered for phrase in phrases):
            return value
    return None


def _parse_cuisines(query: str) -> list[str] | None:
    lowered = query.lower()
    found: list[tuple[int, str]] = []
    for cuisine in _CUISINES:
        for match in re.finditer(rf"\b{re.escape(cuisine.lower())}\b", lowered):
            found.append((match.start(), cuisine))
    found.sort()
    values = [value for _, value in found]
    return values or None


def _parse_room_type(query: str) -> str | None:
    lowered = query.lower()
    if re.search(r"non[- ]shared|not shared|should not be shared|shouldn't be shared|neither shared", lowered):
        return "not shared room"
    if re.search(r"private rooms?|private room", lowered):
        return "private room"
    if re.search(r"entire rooms?|entire room", lowered):
        return "entire room"
    if re.search(r"shared rooms?|shared room", lowered):
        return "shared room"
    return None


def _parse_transportation(query: str) -> str | None:
    lowered = query.lower()
    if re.search(r"\bself[- ]driv\w*\b|\bdriv(?:e|ing) ourselves\b", lowered):
        return "no self-driving"
    if re.search(r"\bflights?\b|\bairline\b|\bair travel\b|\b(?:not|n't) (?:be )?fly(?:ing)?\b|\bprefer not to fly\b", lowered):
        return "no flight"
    no_self_drive = re.search(
        r"not self[- ]driv|avoid self[- ]driv|without self[- ]driv|don't drive ourselves|do not drive ourselves|not planning on self[- ]driv|not involve self[- ]driv|prefer not to drive ourselves|rather not drive ourselves",
        lowered,
    )
    no_flight = re.search(
        r"no flights?|not via flight|avoid (?:any )?flights?|without flights?|cannot utilize flights?|not involve any flights?|not involve flights?|avoid airline transportation|avoid air travel|won't require any flight|will not be considering flight|do not require any flight",
        lowered,
    )
    if no_self_drive:
        return "no self-driving"
    if no_flight:
        return "no flight"
    return None


def recover_evaluator_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a record with the fields required by the released evaluator."""
    query = str(record["query"])
    budget_match = re.search(r"\$\s*([\d,]+)", query)
    if not budget_match:
        raise ValueError("Could not recover budget from query")
    local_constraint = {
        "house rule": _parse_house_rule(query),
        "cuisine": _parse_cuisines(query),
        "room type": _parse_room_type(query),
        "transportation": _parse_transportation(query),
    }
    recovered = dict(record)
    recovered.update(
        {
            "visiting_city_number": _parse_city_count(query, int(record["days"])),
            "people_number": _parse_people(query),
            "local_constraint": str(local_constraint),
            "budget": int(budget_match.group(1).replace(",", "")),
        }
    )
    return recovered


def _normalise_constraint(value: Any) -> dict[str, Any]:
    result = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(result, dict):
        raise TypeError(f"Expected constraint mapping, got {type(result)!r}")
    return result


def validate_against_split(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare recovered fields with gold train/validation fields."""
    totals = {"records": 0, "budget": 0, "people_number": 0, "visiting_city_number": 0, "local_constraint": 0}
    mismatches: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        totals["records"] += 1
        recovered = recover_evaluator_fields(record)
        fields = ("budget", "people_number", "visiting_city_number")
        for field in fields:
            if recovered[field] != record[field]:
                totals[field] += 1
                mismatches.append({"index": index, "field": field, "predicted": recovered[field], "gold": record[field]})
        predicted_constraint = _normalise_constraint(recovered["local_constraint"])
        gold_constraint = _normalise_constraint(record["local_constraint"])
        if predicted_constraint != gold_constraint:
            totals["local_constraint"] += 1
            mismatches.append({"index": index, "field": "local_constraint", "predicted": predicted_constraint, "gold": gold_constraint})
    return {"totals": totals, "mismatches": mismatches}
