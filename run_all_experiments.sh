#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running experiments.}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
MODEL="${MODEL:-gpt-5.4-nano}"
PARALLELISM="${PARALLELISM:-5}"
MAX_PROFILE_PAPERS="${MAX_PROFILE_PAPERS:-10}"

PAPER_DATA="${PAPER_DATA:-data/processed/llm_test_set_at_paper_level.tsv}"
PROFILE_DATA="${PROFILE_DATA:-data/processed/llm_testing_set_advisee_complete_profile_updated.tsv}"

MODEL_TAG="${MODEL//[^A-Za-z0-9_.-]/_}"

echo "Model: ${MODEL}"
echo "Python: ${PYTHON_BIN}"
echo "Parallelism: ${PARALLELISM}"
echo "Paper data: ${PAPER_DATA}"
echo "Profile data: ${PROFILE_DATA}"

echo
echo "Running paper-level known-advisee classification on all rows..."
"${PYTHON_BIN}" scripts/run_mentorship_benchmark.py \
  --mode paper-known-mentee \
  --csv "${PAPER_DATA}" \
  --model "${MODEL}" \
  --limit 0 \
  --parallelism "${PARALLELISM}" \
  --output "results/paper_level/results-paper-known-mentee-${MODEL_TAG}-all.jsonl"

echo
echo "Running profile-level candidate ranking on all rows..."
"${PYTHON_BIN}" scripts/run_mentorship_benchmark.py \
  --mode profile-ranking \
  --csv "${PROFILE_DATA}" \
  --model "${MODEL}" \
  --limit 0 \
  --max-papers-per-profile "${MAX_PROFILE_PAPERS}" \
  --parallelism "${PARALLELISM}" \
  --output "results/profile_level/results-profile-ranking-${MODEL_TAG}-all.jsonl"

echo
echo "Done."
echo "Metrics files are written automatically next to each JSONL result."

# Optional retrieval-assisted ranking ablation.
# This can be much more expensive because web search is billed separately.
# Uncomment when ready:
#
# "${PYTHON_BIN}" scripts/run_mentorship_benchmark.py \
#   --mode profile-ranking-web-search \
#   --csv "${PROFILE_DATA}" \
#   --model "${MODEL}" \
#   --limit 0 \
#   --max-papers-per-profile "${MAX_PROFILE_PAPERS}" \
#   --parallelism "${PARALLELISM}" \
#   --output "results/profile_level/results-profile-ranking-web-${MODEL_TAG}-all.jsonl"
