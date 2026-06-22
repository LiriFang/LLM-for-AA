#!/usr/bin/env python3
"""Compute ranking metrics from profile-ranking benchmark JSONL results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON on line {line_number}: {error}") from error
    return rows


def topk_hit(candidates: Sequence[str], ground_truth: str, k: int) -> bool:
    ground_truth_normalized = normalize_name(ground_truth)
    return any(
        normalize_name(candidate) == ground_truth_normalized
        for candidate in candidates[:k]
    )


def reciprocal_rank(candidates: Sequence[str], ground_truth: str, k: int) -> float:
    ground_truth_normalized = normalize_name(ground_truth)
    for index, candidate in enumerate(candidates[:k], start=1):
        if normalize_name(candidate) == ground_truth_normalized:
            return 1.0 / index
    return 0.0


def get_llm_ranked_advisors(row: dict[str, object]) -> list[str]:
    evaluation = row.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise SystemExit("Each row must contain an `evaluation` object.")
    return [str(name) for name in evaluation.get("ranked_advisors") or []]


def get_local_ranked_advisors(row: dict[str, object]) -> list[str]:
    evaluation = row.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise SystemExit("Each row must contain an `evaluation` object.")
    return [str(name) for name in evaluation.get("local_model_top5") or []]


def get_ground_truth(row: dict[str, object]) -> str:
    evaluation = row.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise SystemExit("Each row must contain an `evaluation` object.")
    return str(evaluation.get("ground_truth_mentor") or "")


def compute_one_source(
    rows: Sequence[dict[str, object]],
    source_name: str,
    ranked_advisors_getter,
) -> dict[str, object]:
    n = len(rows)
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_rank_sum = 0.0
    missing_ranked_lists = 0

    for row in rows:
        candidates = ranked_advisors_getter(row)
        ground_truth = get_ground_truth(row)
        if not candidates:
            missing_ranked_lists += 1
        if topk_hit(candidates, ground_truth, 1):
            hits_at_1 += 1
        if topk_hit(candidates, ground_truth, 5):
            hits_at_5 += 1
        reciprocal_rank_sum += reciprocal_rank(candidates, ground_truth, 5)

    return {
        "source": source_name,
        "n": n,
        "missing_ranked_lists": missing_ranked_lists,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "precision_at_1": safe_divide(hits_at_1, n),
        "recall_at_1": safe_divide(hits_at_1, n),
        "precision_at_5": safe_divide(hits_at_5, 5 * n),
        "recall_at_5": safe_divide(hits_at_5, n),
        "mrr_at_5": safe_divide(reciprocal_rank_sum, n),
    }


def compute_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "llm": compute_one_source(rows, "LLM ranking", get_llm_ranked_advisors),
        "local": compute_one_source(rows, "Local trained model ranking", get_local_ranked_advisors),
    }


def format_source_metrics(metrics: dict[str, object]) -> list[str]:
    return [
        f"{metrics['source']}:",
        f"- Missing ranked lists: {metrics['missing_ranked_lists']}",
        f"- Hits@1: {metrics['hits_at_1']} / {metrics['n']}",
        f"- Hits@5: {metrics['hits_at_5']} / {metrics['n']}",
        f"- Precision@1: {metrics['precision_at_1']:.3f}",
        f"- Recall@1: {metrics['recall_at_1']:.3f}",
        f"- Precision@5: {metrics['precision_at_5']:.3f}",
        f"- Recall@5: {metrics['recall_at_5']:.3f}",
        f"- MRR@5: {metrics['mrr_at_5']:.3f}",
    ]


def format_metrics(path: Path, metrics: dict[str, object]) -> str:
    llm = metrics["llm"]
    local = metrics["local"]
    lines = [
        f"Metrics for {path}",
        "",
        f"Dataset size: {llm['n']} profile-ranking requests",
        "Ground truth: one advisor per profile",
        "",
        "Metric definitions:",
        "- Precision@k = number of requests where the ground-truth advisor appears in top k / (k * number of requests)",
        "- Recall@k = number of requests where the ground-truth advisor appears in top k / number of requests",
        "- For k=1, Precision@1 and Recall@1 are equivalent when every request has one top-1 prediction.",
        "- Missing or NONE rankings are counted as misses.",
        "",
        *format_source_metrics(llm),
        "",
        *format_source_metrics(local),
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute precision@k and recall@k for profile-ranking JSONL results."
    )
    parser.add_argument("jsonl", help="Path to profile-ranking benchmark JSONL output.")
    parser.add_argument(
        "--output",
        help="Optional text file to write the metrics report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.jsonl)
    rows = load_jsonl(path)
    metrics = compute_metrics(rows)
    report = format_metrics(path, metrics)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
