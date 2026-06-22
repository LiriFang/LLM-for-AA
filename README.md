# LLM for Advisor-Advisee Prediction

This repository contains experiment code, prompts, datasets, outputs, and analysis scripts for evaluating LLMs on PhD advisor-advisee relationship prediction from publication metadata.

The paper draft directory `JCDL26-LLM-for-AA/` is kept separate from the reproducibility code and is intentionally ignored by the parent repository.

## Repository Layout

```text
scripts/                    Experiment runners and analysis scripts
prompts/                    Prompt templates used for paper/profile tasks
data/raw/                   Original source data
data/processed/             Processed paper-level and profile-level test sets
results/paper_level/        Paper-level JSONL outputs and metrics reports
results/profile_level/      Profile-level JSONL outputs and metrics reports
analysis/disagreement/      Profile-level disagreement case-study tables
analysis/temporal_disciplinary/
                            Temporal and disciplinary analysis outputs
docs/local_notes/           Local planning/review notes ignored by git
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key_here
```

Optional:

```bash
export MODEL=gpt-5.4-nano
export PARALLELISM=5
```

## Run Experiments

Run the full paper-level and profile-level experiments:

```bash
bash run_all_experiments.sh
```

By default this uses:

- `data/processed/llm_test_set_at_paper_level.tsv`
- `data/processed/llm_testing_set_advisee_complete_profile_updated.tsv`
- outputs under `results/paper_level/` and `results/profile_level/`

Run a small dry run:

```bash
python3 scripts/run_mentorship_benchmark.py \
  --mode profile-ranking \
  --csv data/processed/llm_testing_set_advisee_complete_profile_updated.tsv \
  --model gpt-5.4-nano \
  --limit 2 \
  --dry-run
```

Run profile ranking on all rows:

```bash
python3 scripts/run_mentorship_benchmark.py \
  --mode profile-ranking \
  --csv data/processed/llm_testing_set_advisee_complete_profile_updated.tsv \
  --model gpt-5.4-nano \
  --limit 0 \
  --max-papers-per-profile 10 \
  --parallelism 5 \
  --output results/profile_level/results-profile-ranking-updated-gpt-5.4-nano-all.jsonl
```

Run paper-level known-advisee classification on all rows:

```bash
python3 scripts/run_mentorship_benchmark.py \
  --mode paper-known-mentee \
  --csv data/processed/llm_test_set_at_paper_level.tsv \
  --model gpt-5.4-nano \
  --limit 0 \
  --parallelism 5 \
  --output results/paper_level/results-paper-known-mentee-gpt-5.4-nano-all.jsonl
```

Metrics reports are written automatically next to each JSONL output.

## Standalone Metrics

```bash
python3 scripts/compute_ranking_metrics.py \
  results/profile_level/results-profile-ranking-updated-gpt-5.4-nano-all.jsonl

python3 scripts/compute_classification_metrics.py \
  results/paper_level/results-paper-known-mentee-gpt-5.4-nano-all.jsonl
```

## Case Study And Temporal Analysis

Generate profile-level disagreement tables:

```bash
python3 scripts/analyze_profile_disagreements.py
```

Generate temporal and disciplinary analysis tables/figures:

```bash
python3 scripts/analyze_profile_temporal_subject.py
```

The temporal analysis writes outputs to `analysis/temporal_disciplinary/` and also writes the annual Recall@1 figure to `JCDL26-LLM-for-AA/Figures/profile_recall_by_year.png` for use in the paper draft.
