#!/usr/bin/env python3
"""Compute simple paper-level advisor-name baselines."""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence


DEFAULT_DATA = Path("data/processed/llm_test_set_at_paper_level.tsv")


@dataclass
class PaperCase:
    row_number: int
    advisee_name: str
    advisor_name: str
    has_advisor: bool
    authors: list[str]


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_authors(value: str) -> list[str]:
    return [author.strip() for author in (value or "").split(",") if author.strip()]


def parse_bool(value: str) -> bool:
    return (value or "").strip().casefold() in {"true", "1", "yes"}


def load_cases(path: Path) -> list[PaperCase]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    cases: list[PaperCase] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_number, row in enumerate(reader, start=1):
            cases.append(
                PaperCase(
                    row_number=row_number,
                    advisee_name=(row.get("Advisee_name") or row.get("advisee_name") or "").strip(),
                    advisor_name=(row.get("GTruth_Advisor_name") or row.get("advisor_name") or "").strip(),
                    has_advisor=parse_bool(row.get("has_advisor") or row.get("Class") or ""),
                    authors=parse_authors(row.get("authors") or row.get("Authors") or ""),
                )
            )
    return cases


def non_advisee_authors(case: PaperCase) -> list[str]:
    advisee = normalize_name(case.advisee_name)
    return [author for author in case.authors if normalize_name(author) != advisee]


def last_non_advisee_author(case: PaperCase, rng: random.Random) -> str:
    del rng
    candidates = non_advisee_authors(case)
    return candidates[-1] if candidates else ""


def random_non_advisee_author(case: PaperCase, rng: random.Random) -> str:
    candidates = non_advisee_authors(case)
    return rng.choice(candidates) if candidates else ""


def compute_advisor_metrics(cases: Sequence[PaperCase], predictions: Sequence[str]) -> dict[str, float]:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")

    advisor_tp = advisor_fp = advisor_fn = no_prediction = 0
    ground_truth_positive = 0
    for case, predicted in zip(cases, predictions):
        predicted = predicted or ""
        predicted_positive = bool(normalize_name(predicted))
        correct = (
            case.has_advisor
            and predicted_positive
            and normalize_name(predicted) == normalize_name(case.advisor_name)
        )

        if case.has_advisor:
            ground_truth_positive += 1
        if correct:
            advisor_tp += 1
        elif predicted_positive:
            advisor_fp += 1
        else:
            no_prediction += 1
        if case.has_advisor and not correct:
            advisor_fn += 1

    precision = safe_divide(advisor_tp, advisor_tp + advisor_fp)
    recall = safe_divide(advisor_tp, advisor_tp + advisor_fn)
    f1 = safe_divide(2 * advisor_tp, 2 * advisor_tp + advisor_fp + advisor_fn)
    return {
        "n": len(cases),
        "ground_truth_positive": ground_truth_positive,
        "advisor_tp": advisor_tp,
        "advisor_fp": advisor_fp,
        "advisor_fn": advisor_fn,
        "no_prediction": no_prediction,
        "advisor_precision": precision,
        "advisor_recall": recall,
        "advisor_f1": f1,
    }


def run_rule(cases: Sequence[PaperCase], rule, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    predictions = [rule(case, rng) for case in cases]
    return compute_advisor_metrics(cases, predictions)


def summarize_random(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = ["advisor_precision", "advisor_recall", "advisor_f1", "advisor_tp", "advisor_fp", "advisor_fn"]
    summary: dict[str, float] = {}
    for key in keys:
        values = [metric[key] for metric in metrics]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_sd"] = stdev(values) if len(values) > 1 else 0.0
    summary["n"] = metrics[0]["n"] if metrics else 0
    summary["ground_truth_positive"] = metrics[0]["ground_truth_positive"] if metrics else 0
    return summary


def format_fixed(name: str, metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            f"{name}:",
            f"- N: {int(metrics['n'])}",
            f"- Ground-truth positive rows: {int(metrics['ground_truth_positive'])}",
            f"- Advisor TP: {int(metrics['advisor_tp'])}",
            f"- Advisor FP: {int(metrics['advisor_fp'])}",
            f"- Advisor FN: {int(metrics['advisor_fn'])}",
            f"- No prediction: {int(metrics['no_prediction'])}",
            f"- Advisor precision: {metrics['advisor_precision']:.3f}",
            f"- Advisor recall: {metrics['advisor_recall']:.3f}",
            f"- Advisor F1: {metrics['advisor_f1']:.3f}",
        ]
    )


def format_random(name: str, summary: dict[str, float], trials: int) -> str:
    return "\n".join(
        [
            f"{name} ({trials} trials):",
            f"- N: {int(summary['n'])}",
            f"- Ground-truth positive rows: {int(summary['ground_truth_positive'])}",
            f"- Advisor TP: {summary['advisor_tp_mean']:.1f} +/- {summary['advisor_tp_sd']:.1f}",
            f"- Advisor FP: {summary['advisor_fp_mean']:.1f} +/- {summary['advisor_fp_sd']:.1f}",
            f"- Advisor FN: {summary['advisor_fn_mean']:.1f} +/- {summary['advisor_fn_sd']:.1f}",
            f"- Advisor precision: {summary['advisor_precision_mean']:.3f} +/- {summary['advisor_precision_sd']:.3f}",
            f"- Advisor recall: {summary['advisor_recall_mean']:.3f} +/- {summary['advisor_recall_sd']:.3f}",
            f"- Advisor F1: {summary['advisor_f1_mean']:.3f} +/- {summary['advisor_f1_sd']:.3f}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute random-author and last-author advisor baselines.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA, help=f"Input paper-level TSV/CSV. Defaults to {DEFAULT_DATA}.")
    parser.add_argument("--trials", type=int, default=1000, help="Number of random baseline trials. Defaults to 1000.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed. Defaults to 13.")
    parser.add_argument("--output", type=Path, help="Optional path to write the report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    cases = load_cases(args.csv)
    last_metrics = run_rule(cases, last_non_advisee_author, args.seed)
    random_metrics = [
        run_rule(cases, random_non_advisee_author, args.seed + trial)
        for trial in range(args.trials)
    ]
    random_summary = summarize_random(random_metrics)
    report = "\n\n".join(
        [
            f"Advisor-name baselines for {args.csv}",
            "Definitions: each baseline predicts one advisor name from the paper authors excluding the known advisee. Negative rows where the baseline predicts a name count as advisor false positives.",
            format_fixed("Last non-advisee author baseline", last_metrics),
            format_random("Random non-advisee author baseline", random_summary, args.trials),
        ]
    )
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
