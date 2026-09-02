<div align="center">

# Constraint-Contract Multi-Agent Repair for Travel Planning

**Verifier-guided local repair for reliable language-agent planning**

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b?style=flat-square)](paper/constraint_contract_multi_agent_repair.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-offline%20unit%20suite-2ea44f?style=flat-square)](#quick-start)
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
| Direct prompt | 20.00 | 20.56 | 57.22 |
| Generic self-refine | 12.22 | 13.33 | 51.67 |
| Verifier-guided seed selection | 27.22 | 31.11 | 65.00 |
| **Verifier-directed repair control** | **48.89** | **52.78** | 68.89 |
| Seeded multi-agent planner | 46.67 | 49.44 | 67.22 |
| **CC-MAR** | 35.00 | 38.33 | **72.22** |

The complete table, including CC-MAR ablations, is available in [`results/metrics_summary.csv`](results/metrics_summary.csv).

## What I built

- A deterministic audit layer for route closure, city count, database membership, transportation consistency, accommodation rules, user constraints, and estimated cost.
- A seeded repair pipeline that ranks heterogeneous drafts, repairs only diagnosed failures, re-audits candidates, and conservatively promotes improvements.
- CC-MAR, a typed patch protocol with route, entity, lodging, and budget agents; a critic for cross-contract risk; and a deterministic mediator.
- Resume-safe experiment runners with per-instance caches and debug records for long API experiments.
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

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt

python -m unittest \
  scripts.test_analyze_verifier_calibration \
  scripts.test_api_endpoint_helpers \
  scripts.test_evaluate_submission_subset \
  scripts.test_repair_operator_ablations \
  scripts.test_strong_baseline_runner
```

These tests exercise parsing, metric summaries, ablation configuration, and API endpoint construction without making model calls.

## Reproduction

### 1. Obtain the benchmark data

Download the database linked by the [official TravelPlanner repository](https://github.com/OSU-NLP-Group/TravelPlanner#setup-environment) and place its extracted folders under `database/`. The research code expects the original accommodations, attractions, flights, restaurants, distance-matrix, and background files. The validation table used by the local evaluator is `database/validation.csv`.

The full database, raw generations, API caches, and debug logs are intentionally excluded from this repository because they are large and may contain provider-specific traces.

### 2. Configure an OpenAI-compatible model endpoint

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-provider.example/v1"
export MODEL_NAME="your-model"
```

Never commit `.env` files or credentials. The published experiments used `deepseek-v4-flash`; model behavior and API availability can change, so exact regeneration may differ from the stored results.

### 3. Run the main methods

```bash
# Strong direct/self-refine baselines
STRATEGY=constraint_direct_json python strong_baseline_runner.py

# Seeded verifier-repair pipeline
python seeded_multi_agent_planner.py

# Constraint-contract multi-agent repair
STRATEGY=cc_mar_r3 python contract_multi_agent_repair.py
```

Both repair methods consume stored seed submissions. See the paper appendix for exact paths, settings, and the evaluation commands used for the reported table.

## Scope and limitations

- The main course-paper comparisons are single stored runs on one benchmark validation split; they should not be read as universal architectural rankings.
- The internal audit is deliberately conservative and is not identical to the official scorer.
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
