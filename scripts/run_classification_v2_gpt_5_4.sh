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

DATA="${DATA:-data/processed/llm_test_set_at_paper_level.tsv}"
TEMPLATE="${TEMPLATE:-prompts/classification_prompt_template_with_prof_v2.txt}"
PARALLELISM="${PARALLELISM:-2}"
MODELS="${MODELS:-gpt-5.4-nano gpt-5.4-mini}"

for MODEL in ${MODELS}; do
  MODEL_TAG="${MODEL//[^A-Za-z0-9_.-]/_}"
  OUTPUT="results/paper_level/results-paper-known-mentee-v2-${MODEL_TAG}-all.jsonl"

  echo
  echo "Running paper-known-mentee-v2"
  echo "Model: ${MODEL}"
  echo "Data: ${DATA}"
  echo "Template: ${TEMPLATE}"
  echo "Output: ${OUTPUT}"

  "${PYTHON_BIN}" scripts/run_mentorship_benchmark.py \
    --mode paper-known-mentee-v2 \
    --csv "${DATA}" \
    --model "${MODEL}" \
    --limit 0 \
    --parallelism "${PARALLELISM}" \
    --classification-template "${TEMPLATE}" \
    --output "${OUTPUT}" \
    --resume
done
