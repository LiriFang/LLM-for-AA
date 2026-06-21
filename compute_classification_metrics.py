#!/usr/bin/env python3
"""Compute classification metrics from mentorship benchmark JSONL results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def has_predicted_advisor_name(value: object) -> bool:
    normalized = str(value or "").strip().casefold()
    return normalized not in {"", "none", "no_mentor", "no mentor", "n/a", "na"}


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


def compute_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    labels = Counter()
    tp = fp = tn = fn = 0
    strict_matches = 0
    advisor_matches = 0
    ground_truth_positive_count = 0
    advisor_matches_ground_truth_positive = 0
    advisor_tp = advisor_fp = advisor_fn = 0

    for row in rows:
        evaluation = row.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            raise SystemExit("Each row must contain an `evaluation` object.")
        parsed = row.get("parsed") or {}
        if not isinstance(parsed, dict):
            parsed = {}

        ground_truth_positive = bool(evaluation.get("ground_truth_has_mentor"))
        predicted_label = str(evaluation.get("predicted_label") or "").upper()
        predicted_positive = predicted_label == "YES"
        mentor_match = bool(evaluation.get("mentor_match"))
        predicted_advisor_name = has_predicted_advisor_name(parsed.get("mentor"))
        predicted_advisor_positive = predicted_positive and predicted_advisor_name

        labels[predicted_label or "MISSING"] += 1
        if ground_truth_positive:
            ground_truth_positive_count += 1

        if predicted_positive and ground_truth_positive:
            tp += 1
        elif predicted_positive and not ground_truth_positive:
            fp += 1
        elif not predicted_positive and not ground_truth_positive:
            tn += 1
        else:
            fn += 1

        if evaluation.get("strict_match"):
            strict_matches += 1
        if mentor_match:
            advisor_matches += 1
            if ground_truth_positive:
                advisor_matches_ground_truth_positive += 1

        advisor_correct = ground_truth_positive and predicted_advisor_positive and mentor_match
        if advisor_correct:
            advisor_tp += 1
        elif predicted_advisor_positive:
            advisor_fp += 1
        if ground_truth_positive and not advisor_correct:
            advisor_fn += 1

    n = len(rows)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_divide(tp + tn, n)
    advisor_precision = safe_divide(advisor_tp, advisor_tp + advisor_fp)
    advisor_recall = safe_divide(advisor_tp, advisor_tp + advisor_fn)
    advisor_f1 = safe_divide(2 * advisor_tp, 2 * advisor_tp + advisor_fp + advisor_fn)

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_accuracy_yes_vs_not_yes": accuracy,
        "strict_matches": strict_matches,
        "strict_accuracy": safe_divide(strict_matches, n),
        "advisor_matches": advisor_matches,
        "advisor_match_rate": safe_divide(advisor_matches, n),
        "ground_truth_positive_count": ground_truth_positive_count,
        "advisor_matches_ground_truth_positive": advisor_matches_ground_truth_positive,
        "advisor_match_rate_ground_truth_positive": safe_divide(
            advisor_matches_ground_truth_positive,
            ground_truth_positive_count,
        ),
        "advisor_tp": advisor_tp,
        "advisor_fp": advisor_fp,
        "advisor_fn": advisor_fn,
        "advisor_precision": advisor_precision,
        "advisor_recall": advisor_recall,
        "advisor_f1": advisor_f1,
        "predicted_label_counts": dict(sorted(labels.items())),
        "uncertain_count": labels.get("UNCERTAIN", 0),
        "uncertain_rate": safe_divide(labels.get("UNCERTAIN", 0), n),
    }


def format_metrics(path: Path, metrics: dict[str, object]) -> str:
    lines = [
        f"Metrics for {path}",
        "",
        "Primary binary setup:",
        "- Positive class: ground truth has a PhD advisor",
        "- Predicted positive: model outputs YES",
        "- Predicted negative: model outputs NO, UNCERTAIN, missing, or any non-YES label",
        "",
        "Advisor-identification setup:",
        "- Advisor TP: ground truth has a PhD advisor, model outputs YES with an advisor name, and the advisor name matches",
        "- Advisor FP: model outputs YES with an advisor name, but the ground truth has no advisor or the advisor name is wrong",
        "- Advisor FN: ground truth has a PhD advisor, but the model does not correctly name that advisor",
        "",
        f"N: {metrics['n']}",
        f"TP: {metrics['tp']}",
        f"FP: {metrics['fp']}",
        f"TN: {metrics['tn']}",
        f"FN: {metrics['fn']}",
        "",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall: {metrics['recall']:.3f}",
        f"F1: {metrics['f1']:.3f}",
        f"Binary accuracy (YES vs not-YES): {metrics['binary_accuracy_yes_vs_not_yes']:.3f}",
        "",
        f"Advisor TP: {metrics['advisor_tp']}",
        f"Advisor FP: {metrics['advisor_fp']}",
        f"Advisor FN: {metrics['advisor_fn']}",
        f"Advisor precision: {metrics['advisor_precision']:.3f}",
        f"Advisor recall: {metrics['advisor_recall']:.3f}",
        f"Advisor F1: {metrics['advisor_f1']:.3f}",
        "",
        f"Strict accuracy: {metrics['strict_accuracy']:.3f} ({metrics['strict_matches']}/{metrics['n']})",
        f"Advisor match rate, all rows: {metrics['advisor_match_rate']:.3f} ({metrics['advisor_matches']}/{metrics['n']})",
        "Advisor match rate, ground-truth-positive rows: "
        f"{metrics['advisor_match_rate_ground_truth_positive']:.3f} "
        f"({metrics['advisor_matches_ground_truth_positive']}/{metrics['ground_truth_positive_count']})",
        f"Uncertain rate: {metrics['uncertain_rate']:.3f} ({metrics['uncertain_count']}/{metrics['n']})",
        f"Predicted label counts: {metrics['predicted_label_counts']}",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, and F1 for classification JSONL results."
    )
    parser.add_argument("jsonl", help="Path to benchmark JSONL output.")
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
