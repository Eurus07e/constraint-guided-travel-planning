<div align="center">

# Constraint-Contract Multi-Agent Repair for Travel Planning

**Verifier-guided local repair for reliable language-agent planning**

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b?style=flat-square)](paper/constraint_contract_multi_agent_repair.pdf)
[![Python](https://img.shields.io/badge/python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://github.com/Eurus07e/constraint-guided-travel-planning/actions/workflows/tests.yml/badge.svg)](https://github.com/Eurus07e/constraint-guided-travel-planning/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-111111?style=flat-square)](LICENSE)

Course research project for *Introduction to Artificial Intelligence*, Nanjing University.

[Read the paper](paper/constraint_contract_multi_agent_repair.pdf) · [Inspect the results](results/metrics_summary.csv) · [Reproduce the evaluation](#reproduction)

</div>

---

Large language models can write convincing itineraries while quietly violating database-grounded constraints: a hotel may require more nights than the schedule allows, a restaurant may be in the wrong city, or the final route may exceed the budget. This project studies a simple question:

> Can explicit verification and conservative local repair make language-agent plans more reliable than prompting or unconstrained self-refinement?

The answer on the 180-instance TravelPlanner validation split is yes. The strongest verifier-directed repair control raises Final Pass Rate from **20.00% to 48.89%**. The proposed **Constraint-Contract Multi-Agent Repair (CC-MAR)** protocol reaches **35.00% Final Pass** and the highest Hard Macro score (**72.22%**), but does not outperform the narrower deterministic repair control. The negative result is important: adding agent roles is less valuable than exposing concrete failures and accepting only verified improvements.

<p align="center">
  <img src="assets/pipeline.png" width="88%" alt="Verifier-guided planning and repair pipeline">
</p>

## Key results

All values below are percentages from a local reproduction of the official TravelPlanner validation protocol. Final Pass is the primary end-to-end metric.

| Method | Final Pass | Commonsense Macro | Hard Macro |
|---|---:|---:|---:|
| Direct Prompt | 20.00 | 20.56 | 57.22 |
| Generic Self-Refine | 12.22 | 13.33 | 51.67 |
| Verifier-Guided Hybrid Selector | 27.22 | 31.11 | 65.00 |
| Verifier-Directed Repair Control | 48.89 | 52.78 | 68.89 |
| Seeded Multi-Agent Planner | 46.67 | 49.44 | 67.22 |
| CC-MAR | 35.00 | 38.33 | 72.22 |

The complete table, including CC-MAR ablations, is available in [`results/metrics_summary.csv`](results/metrics_summary.csv).

## What I built

- A deterministic audit layer for route closure, city count, database membership, transportation consistency, accommodation rules, user constraints, and estimated cost.
- A seeded repair pipeline that ranks heterogeneous drafts, repairs only diagnosed failures, re-audits candidates, and conservatively promotes improvements.
- CC-MAR, a typed patch protocol with route, entity, lodging, and budget agents; a critic for cross-contract risk; and a deterministic mediator.
- Fingerprinted experiment runners with atomic checkpoints and per-request accounting.
- Controlled baselines, component ablations, failure analysis, difficulty breakdowns, and reproducible paper artifacts.

## Method

CC-MAR represents each itinerary with a lexicographic violation vector:

```text
(fatal failures, invalid entities, invalid transport,
 minimum-night misses, local-constraint misses,
 budget failure, warnings, -audit score)
```

Specialist agents propose field-level patches instead of rewriting the whole itinerary. A critic flags possible cross-contract regressions. The mediator materializes each proposal, runs the deterministic audit again, and accepts at most one strictly improving patch per round. This makes every state transition inspectable and prevents a fluent rewrite from silently damaging already valid fields.

## Repository map

```text
.
├── contract_multi_agent_repair.py   # CC-MAR and ablation switches
├── seeded_multi_agent_planner.py    # seeded verifier-repair pipeline
├── strong_baseline_runner.py        # direct and self-refine baselines
├── utils/plan_audit.py              # deterministic constraint audit
├── evaluation/                      # TravelPlanner evaluator integration
├── scripts/                         # analysis, ablations, and tests
├── results/                         # compact, inspectable result artifacts
├── paper/                           # final PDF and LaTeX source
└── upstream/                        # notes on the original benchmark
```

## Quick start

```bash
git clone https://github.com/Eurus07e/constraint-guided-travel-planning.git
cd constraint-guided-travel-planning

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt -c requirements-lock.txt
python -m pip check

python -m unittest discover -s scripts -p 'test_*.py' -v
python -m scripts.verify_frozen
```

Use Python 3.11 on Linux or macOS. The lock file fixes the validated dependency
versions. These tests check parsing, evaluation, cache and checkpoint recovery,
and request failures without contacting a model provider.

## Reproduction

The repository includes 13 frozen validation submissions and per-sample outcomes.
Start with [the reproduction guide](docs/reproduction.md): it separates offline
artifact verification, database preparation, exact historical reconstruction,
and normal runner usage. The rebuilt deterministic control matches all 180
historical plans and reproduces **88/180 Final Pass**.

```bash
# No database or model calls required:
python -m scripts.verify_frozen

# After preparing the official database:
python -m scripts.prepare_data
python -m scripts.reproduce --limit 3 --output runs/demo.jsonl
python -m scripts.reproduce --output runs/reproduced_repair.jsonl
python -m scripts.evaluate_submission_subset --set-type validation \
  --input runs/reproduced_repair.jsonl --output runs/reproduced_metrics.json
```

`TP_DATABASE_DIR` selects an existing database. `DIRECT_SUBMISSION_FILE` and
`PROGRAM_SUBMISSION_FILE` select alternative seeds. The default seeds are the
published historical Direct and Program predictions, regardless of which model
is used for new repair calls.

## Running and maintaining the repository

- Full and subset scoring share indexed sample validation and consistent metric
  denominators. See [replay checks](results/replay_verification.json).
- `legacy-v1` preserves historical reconstruction. Regular runs use `strict-v2`
  to check completeness, city alignment, entity diversity and database eligibility
  before early stopping. [Audit regression checks](results/audit_comparison.json)
  cover the stored validation predictions.
- Output directories are isolated by configuration, code, data and seed hashes.
  Checkpoints use atomic writes, and API attempts and cache hits are recorded.
  See [runner configuration and recovery](docs/running.md).
- [Validation commands](docs/validation.md) cover offline tests, benchmark replay
  and deterministic reconstruction. These checks require no paid model calls.

## Scope and limitations

- The main course-paper comparisons are single stored runs on one benchmark validation split; they should not be read as universal architectural rankings.
- The versioned internal audit is independent of the final scorer; its calibration is measured separately.
- A benchmark-valid itinerary is not real-world travel advice. Live availability, safety, visas, accessibility, and disruptions are outside the benchmark.
- CC-MAR improves hard-constraint coverage, but its current specialist decomposition is not reliably better than simpler repair policies.

## Academic context

This repository presents an independent course research project by **Yuxuan Shu**, School of Intelligence Science and Technology, Nanjing University. The work extends the open-source TravelPlanner benchmark with verifier-guided repair, controlled multi-agent patch arbitration, and additional analysis. It is a course artifact, not an official publication or an official release by the TravelPlanner authors.

## Attribution

This project is built on [OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner), introduced by Xie et al. at ICML 2024. The original benchmark code and assets remain under their MIT license; see [`upstream/NOTICE.md`](upstream/NOTICE.md). Please cite the original TravelPlanner paper when using the benchmark.

```bibtex
@inproceedings{xie2024travelplanner,
  title     = {TravelPlanner: A Benchmark for Real-World Planning with Language Agents},
  author    = {Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

## Contact

**Yuxuan Shu** · [yuxuanshu@smail.nju.edu.cn](mailto:yuxuanshu@smail.nju.edu.cn)
