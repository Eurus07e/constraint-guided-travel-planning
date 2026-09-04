# Reliability and reproduction work

Base: public main `5be011f7bbdd49ef673e1d112c79c7c0b86e70c0`.
Updated: 2026-09-04. Work was performed in an isolated copy; the original
working directory and historical predictions remain unchanged.

- [x] 1. CI and offline test entry point
- [x] 2. Frozen predictions, data preparation, seed inputs and reproduction
- [x] 3. Indexed evaluation, consistent metric denominators and path handling
- [x] 4. Versioned audit, conservative stopping and patch regression checks
- [x] 5. Run manifests, cache isolation, recovery and request accounting
- [ ] 6. Independent new model replications: waiting for valid provider credentials

Step 6's paired analysis, archived-run audit, experiment driver, documentation
and updated paper are complete. New model experiments are still incomplete.
Both provider preflights returned HTTP 401 and no batch was dispatched. Update
local credentials, then follow `docs/experiments.md`; results must be reviewed
and the tables regenerated after successful runs. See
`results/replication/new_runs_status.json` for the recorded blocker.

## Verification

- Offline suite: 32 tests, 31 passed and one explicitly gated integration test
  skipped. The three-test integration suite also passed with local benchmark data.
- Frozen artifact verification: all 13 methods, 180 indexed predictions each.
- Fresh evaluation replay: all six reported metrics and all per-case Final Pass
  outcomes match the frozen records for every method.
- Selector plus deterministic repair reconstructed all 180 plans exactly; 88/180
  pass the final evaluator. No model request was needed for reconstruction.
- Strict audit calibration: zero false positives and zero false negatives on
  the 900 stored predictions from five development-set methods. This is
  development calibration, not held-out evidence.
- A three-case offline seeded run and repeat/resume behavior were checked.
- The 16-page paper compiled successfully and rendered pages were visually checked.
- The restored offline CI passed on the first-stage branch commit:
  https://github.com/Eurus07e/constraint-guided-travel-planning/actions/runs/33844385263
  Later changes are subject to the branch/PR's current CI check.

Re-run the key checks from the repository root:

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
python -m scripts.verify_frozen
python -m scripts.prepare_data --require-frozen-hashes
python -m scripts.replay_frozen
python -m scripts.reproduce
TP_INTEGRATION=1 python -m unittest scripts.test_heldout_query_adapter -v
python -m scripts.run_replication --dry-run --model <provider-model>
```

Set `TP_DATABASE_DIR` if the benchmark database is outside this repository.
Historical metrics use the original audit/scorer semantics; new runs use the
versioned strict audit. The release now discloses that historical Checklist-Direct
predictions all used Direct fallback and many Self-Refine cases used parse fallback.
They do not establish clean, independent strong-baseline comparisons.

## Evidence files

- `results/frozen/manifest.json`: hashes, frozen predictions, expected metrics.
- `results/replay_verification.json`: complete fresh replay checks.
- `results/reconstruction_verification.json`: exact deterministic reconstruction.
- `results/audit_comparison.json`: versioned audit calibration.
- `results/baseline_provenance.json`: historical fallback audit.
- `results/paired_analysis.json`: paired intervals and exact tests.
- `results/replication/`: archived Qwen evidence and new-run status.
- `paper/constraint_contract_multi_agent_repair.pdf`: updated report.
