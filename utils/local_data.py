import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "osunlp/TravelPlanner"
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATABASE_DIR = ROOT_DIR / "database"


def _load_local_validation_records():
    path = LOCAL_DATABASE_DIR / "validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Local validation file not found: {path}")
    df = pd.read_csv(path)
    return df.to_dict("records")


def load_travelplanner_records(set_type, prefer_local_validation=True, download_mode=None):
    set_type = str(set_type).strip().lower()

    if set_type == "validation" and prefer_local_validation:
        return _load_local_validation_records()

    kwargs = {}
    if download_mode:
        kwargs["download_mode"] = download_mode

    try:
        dataset = load_dataset(DATASET_NAME, set_type, **kwargs)[set_type]
    except Exception:
        # Evaluation should remain usable when the already-downloaded official
        # Arrow split is available but the Hub is temporarily unreachable.
        from datasets import Dataset

        cached = sorted(
            Path.home().glob(
                f".cache/huggingface/datasets/osunlp___travel_planner/{set_type}/*/*/travel_planner-{set_type}.arrow"
            )
        )
        if not cached:
            raise
        dataset = Dataset.from_file(str(cached[-1]))
    records = [dict(x) for x in dataset]
    if set_type == "test":
        from scripts.heldout_query_adapter import recover_evaluator_fields

        records = [recover_evaluator_fields(record) for record in records]
    return records


def load_travelplanner_dataframe(set_type, prefer_local_validation=True, download_mode=None):
    set_type = str(set_type).strip().lower()

    if set_type == "validation" and prefer_local_validation:
        path = LOCAL_DATABASE_DIR / "validation.csv"
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame(
        load_travelplanner_records(
            set_type,
            prefer_local_validation=prefer_local_validation,
            download_mode=download_mode,
        )
    )


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
