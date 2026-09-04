# Repository validation

Run the commands from the repository root with Python 3.11 and dependencies from
`requirements-research.txt` constrained by `requirements-lock.txt`.
`python -m pip check` verifies the installed environment. Verification of frozen artifacts and unit tests does
not need a model account or the benchmark database.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s scripts -p 'test_*.py' -v
python -m scripts.verify_frozen
```

The offline suite covers input identities, metric denominators, strict audit
checks, role-scoped patch rejection, cache separation, request budgets, atomic
writes, explicit draft reuse failed-case recovery, output ownership and concurrent-run exclusion. Five database-dependent
tests are intentionally gated. The CI workflow runs this same offline suite.

With the official database available (set `TP_DATABASE_DIR` if needed):

```bash
python -m scripts.prepare_data --require-frozen-hashes
TP_INTEGRATION=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest scripts.test_heldout_query_adapter scripts.test_runner_integration -v
python -m scripts.replay_frozen
python -m scripts.reproduce --output runs/reproduced_repair.jsonl
python -m scripts.evaluate_submission_subset --set-type validation \
  --input runs/reproduced_repair.jsonl --output runs/reproduced_metrics.json
```

The integration suite checks the cached validation split and all three primary
runner entry points. Direct requests receive in-process fake API responses;
repair runs use the offline path. Each runner writes three cases and resumes
without further requests. The suite also simulates HTTP 401 failures followed by
successful retries in all three runners, and verifies numeric distance conversion
against every populated record in the local distance table. No external model
service is contacted. A small pass/fail sample also exercises the final
evaluator and JSON output with the locked dependencies.

The replay checks all six stored metrics and each per-case Final Pass outcome
for 13 methods on 180 validation cases. Deterministic reconstruction uses the
historical audit and matches the 180 published repair plans, including 88 passing
cases. These are software regression/reproduction checks of existing results.

Paper files and `results/metrics_summary.csv` are unchanged by the maintenance
work. This validation does not introduce new model experiments or paper claims.
