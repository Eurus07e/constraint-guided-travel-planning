# Reproduction

Run commands from the repository root with Python 3.11.

## 1. Inspect frozen evidence without data or API access

```bash
python -m scripts.verify_frozen
```

The package contains 13 complete validation predictions, sample IDs, per-sample
Final Pass outcomes, expected metrics and SHA-256 digests. Predictions omit task
queries and raw provider traces. This command checks consistency of the stored
artifacts; recomputing benchmark validity requires the official database below.

## 2. Prepare the official database

Download and extract the database from
[TravelPlanner setup](https://github.com/OSU-NLP-Group/TravelPlanner#setup-environment).
Place accommodations, attractions, flights, restaurants, googleDistanceMatrix
and background inside `database/`. Alternatively point `TP_DATABASE_DIR` at an
existing database directory. The environment variable is read at process startup.

```bash
python -m scripts.prepare_data
# Only if validation.csv is missing:
python -m scripts.prepare_data --fetch-validation
# For byte-identical historical input files:
python -m scripts.prepare_data --require-frozen-hashes
```

A freshly exported Hugging Face CSV can differ byte-for-byte from the historical
CSV while containing the same records. A hash mismatch is reported explicitly;
do not claim byte-identical reproduction in that case.

## 3. Reconstruct the deterministic repair control

Both repair runners default to the shipped historical Direct and Program seeds.
Use `DIRECT_SUBMISSION_FILE` and `PROGRAM_SUBMISSION_FILE` to select other full,
indexed validation submissions. `MODEL_NAME` controls new model calls; it does
not relabel the provenance of these frozen seeds.

```bash
python -m scripts.reproduce --limit 3 --output runs/demo.jsonl
python -m scripts.reproduce --output runs/reproduced_repair.jsonl
python -m scripts.evaluate_submission_subset --set-type validation \
  --input runs/reproduced_repair.jsonl --output runs/reproduced_metrics.json
```

The complete historical reconstruction should reproduce **88/180 Final Pass**.
The internal audit used for this reconstruction is `legacy-v1`. Regular runs use the stricter audit; the reconstruction command explicitly
selects the historical audit so existing results remain reproducible.

## 4. Use the existing model runners (model access required)

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_API_BASE='https://your-provider.example/v1'
export MODEL_NAME='your-model'
STRATEGY=constraint_direct_json python strong_baseline_runner.py
python seeded_multi_agent_planner.py
STRATEGY=cc_mar_r3 python contract_multi_agent_repair.py
```

Configure output, request limits and recovery using [the runner guide](running.md).
The seeded runners use the shipped Direct and Program predictions by default.
The checklist Direct runner is distinct from the historical Direct prompt;
its fresh outputs are not an exact regeneration of the original frozen seeds.

## 5. Tests

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s scripts -p 'test_*.py' -v
TP_INTEGRATION=1 python -m unittest scripts.test_heldout_query_adapter scripts.test_runner_integration -v
```

The integration suite requires the official validation split. API tests use
local fake responses; ordinary unit tests never call a model provider.
