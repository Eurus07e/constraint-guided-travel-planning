# Result artifacts

`metrics_summary.csv` contains the compact table used in the project README
and final course paper. Values are percentages from a local reproduction of
the official TravelPlanner evaluation protocol on the 180-instance validation
split.

The repository intentionally omits raw model generations, request caches, and
per-instance debug logs. Those artifacts are large, provider-specific, and not
required to inspect the headline comparison. The final paper documents the
experimental settings, limitations, and artifact paths used in the original
working directory.
