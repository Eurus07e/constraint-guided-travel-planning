# Result artifacts

- `metrics_summary.csv`: historical 180-instance validation table, in percentages.
- `frozen/`: query-free predictions, sample outcomes, expected metrics and hashes.
- `replay_verification.json`: fresh replay of all 13 historical methods.
- `reconstruction_verification.json`: exact 180-plan deterministic reconstruction.
- `audit_comparison.json`: legacy versus strict audit calibration on frozen plans.
- `paired_analysis.json`: conditional paired intervals and exploratory comparisons.
- `replication/`: separately labeled archived Qwen evidence, provenance and the
  status of new model experiments.

Run `python -m scripts.verify_frozen` without data or credentials. Run
`python -m scripts.replay_frozen` with the official database to recompute validity.
Run `python -m scripts.build_public_artifacts` to regenerate public tables and
paper update tables. The legacy `scripts/build_report_artifacts.py` remains for
historical working-directory analyses that need excluded raw debug logs.

Raw provider responses, credentials, local database files and new request caches
are excluded. Historical task text is not redistributed in the frozen predictions.
