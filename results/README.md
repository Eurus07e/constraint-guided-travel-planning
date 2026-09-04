# Result artifacts

- `metrics_summary.csv`: existing validation metrics, unchanged.
- `frozen/`: predictions, sample IDs, expected outcomes and file hashes needed to
  reproduce the existing results without regenerating model outputs.
- `replay_verification.json`: fresh scoring checks for all 13 stored methods.
- `reconstruction_verification.json`: exact 180-plan deterministic reconstruction.
- `audit_comparison.json`: audit regression checks on the stored predictions.

Run `python -m scripts.verify_frozen` without benchmark data or credentials.
Run `python -m scripts.replay_frozen` with the official database to recompute
validity. See `docs/reproduction.md` for setup and commands.

Raw provider responses, credentials, database files and request caches are
excluded from Git. Frozen predictions omit the original task queries.
