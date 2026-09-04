# Runner configuration and recovery

Existing model runners read `OPENAI_API_KEY`, `OPENAI_API_BASE` and `MODEL_NAME`
from the environment. Keep credentials in the environment or an ignored local
file. A model account is unnecessary for offline tests, frozen replay and
reconstruction; see [reproduction](reproduction.md).

## Outputs and configuration

Outputs go to `runs/<strategy>/<configuration fingerprint>/`. `OUT_ROOT` changes
the base directory and still receives a fingerprint subdirectory. The run
manifest records model settings, code hashes, dataset/seed hashes, audit version
and a sanitized endpoint. It never records the API key.

`IDS=1,3,5` selects one-based rows; `LIMIT=3` selects the first three rows.
`EXCLUDE_IDS` removes selected rows. Invalid, duplicate or empty selections fail
before model requests. `DIRECT_SUBMISSION_FILE` and `PROGRAM_SUBMISSION_FILE`
select complete indexed seed submissions for the seeded and CC-MAR runners.
Their defaults are the published frozen seeds.

The audit defaults to `strict-v2`. `scripts.reproduce` explicitly chooses
`legacy-v1` to reconstruct historical results.

## Resume and failure reporting

Re-run the same command to resume completed cases. Corrupt response caches, output files and failed checkpoints are regenerated.
Malformed seed files are rejected with the source path and line number. `status.json` lists failed IDs; submissions
include empty plans for failed requests, and the process exits unsuccessfully
while failures remain. A second process cannot run the same configuration concurrently. Locks are
released when a process exits, including after an exception. An explicit
`SUBMISSION_FILE` is reserved when a run starts, even before its first output
is written, and cannot overwrite output
owned by another configuration or output without an ownership manifest.

Caches are scoped by run, endpoint, request parameters and optional `TP_RUN_ID`.
Changing code, input contents or configuration creates a separate run. Atomic
file replacement protects existing checkpoints from interrupted writes.

## Request accounting

`TP_MAX_REQUESTS` bounds network attempts for a run, including retries. Cache hits
do not consume that bound. HTTP 401/403 errors stop immediately; connection,
timeout, rate-limit and selected server failures receive bounded retries.
`MAX_RETRIES` sets the maximum attempts per request and `REQUEST_TIMEOUT` sets
the timeout in seconds. Both must be positive integers; `TP_MAX_REQUESTS` must
be nonnegative, with zero meaning unlimited. Required request failures propagate
through planner, verifier, specialist and critic stages so affected cases remain
failed and retryable. Increase a depleted request budget in a new run.

```bash
python -m scripts.summarize_requests runs/<strategy>/<fingerprint>/requests.jsonl
```

The log records request IDs, returned model names, usage when supplied by the
provider, latency, failures and cache hits. Missing provider usage is unavailable,
not a zero-token request.

## Explicit Self-Refine draft reuse

Set `CONSTRAINT_DIRECT_OUTPUT_DIR` to a completed Direct run's fingerprint
folder. The runner reads its `<split>/generated_plan_<id>.json` and matching
`debug/debug_<id>.json`, verifies the model and completed status, and records
those files' hashes in the new run manifest. When this option is omitted,
Self-Refine generates its own draft. An invalid explicitly selected draft fails
with an actionable error rather than silently switching inputs.
