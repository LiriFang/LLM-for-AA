# Advisor-Advisee Prompt Runner

This workspace now includes a small CLI for testing the prompt variants from `BrainstormingDoc.md` against the OpenAI API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key_here
```

Optional:

```bash
export OPENAI_MODEL=gpt-5-mini
```

## Prompt modes

- `paper-no-search`
- `paper-web-search`
- `paper-known-mentee`
- `profile-no-search`
- `profile-web-search`

`paper-*` modes run one CSV row per request. `profile-*` modes group rows by `advisee_name` and send the earliest papers for that advisee.

## Examples

Preview the exact prompt without making an API call:

```bash
python3 run_mentorship_benchmark.py \
  --mode paper-web-search \
  --pmid 22250215 \
  --dry-run
```

Run one paper-level request without web search:

```bash
python3 run_mentorship_benchmark.py \
  --mode paper-no-search \
  --pmid 22250215
```

Run one paper-level request with built-in web search:

```bash
python3 run_mentorship_benchmark.py \
  --mode paper-web-search \
  --pmid 22250215 \
  --output results-paper.jsonl
```

Run one known-mentee request:

```bash
python3 run_mentorship_benchmark.py \
  --mode paper-known-mentee \
  --pmid 14563143
```

Run one known-mentee request with structured decision evidence:

```bash
python3 run_mentorship_benchmark.py \
  --mode paper-known-mentee \
  --pmid 14563143 \
  --collect-evidence \
  --output results-evidence.jsonl
```

Run an author-profile request using the earliest 6 papers for the advisee:

```bash
python3 run_mentorship_benchmark.py \
  --mode profile-web-search \
  --advisee "Beckel JM" \
  --max-papers-per-profile 6 \
  --output results-profile.jsonl
```

## Notes

- Web search uses OpenAI's built-in `web_search` tool, so no separate search API key is needed.
- Results are written as JSON Lines if `--output` is provided.
- The script computes a simple exact-match check against `advisor_name` and `advisee_name` from `testing_set.csv`.
- `--collect-evidence` adds `confidence`, `primary_evidence`, `secondary_evidence`, and `short_justification` fields to the parsed output.
- The CSV `detail` column is malformed in this dataset, so the runner reconstructs the prompt from the normalized fields instead.
