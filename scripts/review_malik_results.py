#!/usr/bin/env python3
"""Review Malik-format paper/profile mentorship result JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from compute_classification_metrics import safe_divide
from compute_ranking_metrics import normalize_name, reciprocal_rank, topk_hit


DEFAULT_PROFILE_DIR = Path("/Users/lirifang/Downloads/malik_result_profile_level")
DEFAULT_PAPER_DIR = Path("/Users/lirifang/Downloads/malik_result_paper_level")
DEFAULT_OUTPUT = Path("malik_result_review.md")


def load_json_array(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"{path} is not a JSON array.")
    return [row for row in data if isinstance(row, dict)]


def split_csv_list(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_label(response: str) -> str:
    match = re.search(r"(?:Mentorship|Advisor)\s+Detected:\s*([A-Za-z]+)", response, flags=re.I)
    return match.group(1).upper() if match else ""


def parse_named_field(response: str, field: str) -> str:
    pattern = rf"^\s*(?:If YES or UNCERTAIN:\s*)?{re.escape(field)}:\s*(.+?)\s*$"
    for line in response.splitlines():
        match = re.match(pattern, line, flags=re.I)
        if match:
            value = match.group(1).split("|", maxsplit=1)[0].strip()
            if normalize_name(value) in {"none", "uncertain", "na", "n/a"}:
                return ""
            return value
    return ""


def parse_ranked_advisors(response: str) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for line in response.splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s*([^|]+)", line)
        if not match:
            continue
        advisor = match.group(2).strip()
        if normalize_name(advisor) not in {"none", "uncertain", "na", "n/a"}:
            ranked.append((int(match.group(1)), advisor))
    ranked.sort(key=lambda item: item[0])
    return [advisor for _, advisor in ranked]


def name_match(predicted: str, truth: str) -> bool:
    return bool(predicted) and normalize_name(predicted) == normalize_name(truth)


def is_ranking_profile_file(path: Path) -> bool:
    return "no-ranking" not in path.name


def local_top5(row: dict[str, object]) -> list[str]:
    return split_csv_list(row.get("Model_predicted_ranked_coauthor_name"))


def compute_paper_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    tp = fp = tn = fn = 0
    advisor_tp = advisor_fp = advisor_fn = 0
    mentee_matches = 0
    label_counts: dict[str, int] = {}
    parsed_advisor_count = 0
    parse_missing = 0

    for row in rows:
        response = str(row.get("model_response") or "")
        label = parse_label(response)
        label_counts[label or "MISSING"] = label_counts.get(label or "MISSING", 0) + 1
        if not label:
            parse_missing += 1

        truth_positive = bool(row.get("has_advisor"))
        predicted_positive = label == "YES"
        truth_advisor = str(row.get("GTruth_Advisor_name") or "")
        predicted_advisor = parse_named_field(response, "Advisor")
        predicted_mentee = parse_named_field(response, "Advisee")
        truth_mentee = str(row.get("Advisee_name") or "")

        if predicted_advisor:
            parsed_advisor_count += 1
        if predicted_mentee and name_match(predicted_mentee, truth_mentee):
            mentee_matches += 1

        if predicted_positive and truth_positive:
            tp += 1
        elif predicted_positive and not truth_positive:
            fp += 1
        elif not predicted_positive and not truth_positive:
            tn += 1
        else:
            fn += 1

        advisor_correct = truth_positive and predicted_positive and name_match(predicted_advisor, truth_advisor)
        if advisor_correct:
            advisor_tp += 1
        elif predicted_positive and predicted_advisor:
            advisor_fp += 1
        if truth_positive and not advisor_correct:
            advisor_fn += 1

    n = len(rows)
    return {
        "n": n,
        "labels": label_counts,
        "parse_missing": parse_missing,
        "parsed_advisor_count": parsed_advisor_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "binary_accuracy": safe_divide(tp + tn, n),
        "advisor_tp": advisor_tp,
        "advisor_fp": advisor_fp,
        "advisor_fn": advisor_fn,
        "advisor_precision": safe_divide(advisor_tp, advisor_tp + advisor_fp),
        "advisor_recall": safe_divide(advisor_tp, advisor_tp + advisor_fn),
        "advisor_f1": safe_divide(2 * advisor_tp, 2 * advisor_tp + advisor_fp + advisor_fn),
        "mentee_matches": mentee_matches,
    }


def compute_profile_metrics(rows: Sequence[dict[str, object]], ranking: bool) -> dict[str, object]:
    n = len(rows)
    hits_at_1 = hits_at_5 = 0
    rr_sum = 0.0
    missing_predictions = 0
    local_hits_at_1 = local_hits_at_5 = 0
    local_rr_sum = 0.0
    label_counts: dict[str, int] = {}
    advisor_expanded_name_examples: list[str] = []

    for row in rows:
        response = str(row.get("model_response") or "")
        label = parse_label(response)
        label_counts[label or "MISSING"] = label_counts.get(label or "MISSING", 0) + 1
        truth_advisor = str(row.get("GTruth_Advisor_name") or "")

        predictions = parse_ranked_advisors(response) if ranking else []
        if not predictions:
            advisor = parse_named_field(response, "Advisor")
            predictions = [advisor] if advisor else []
        if not predictions:
            missing_predictions += 1

        if predictions and not any(
            normalize_name(prediction) == normalize_name(candidate)
            for candidate in local_top5(row)
            for prediction in predictions[:1]
        ):
            advisor_expanded_name_examples.append(predictions[0])

        if topk_hit(predictions, truth_advisor, 1):
            hits_at_1 += 1
        if topk_hit(predictions, truth_advisor, 5):
            hits_at_5 += 1
        rr_sum += reciprocal_rank(predictions, truth_advisor, 5)

        local_predictions = local_top5(row)
        if topk_hit(local_predictions, truth_advisor, 1):
            local_hits_at_1 += 1
        if topk_hit(local_predictions, truth_advisor, 5):
            local_hits_at_5 += 1
        local_rr_sum += reciprocal_rank(local_predictions, truth_advisor, 5)

    return {
        "n": n,
        "labels": label_counts,
        "missing_predictions": missing_predictions,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "precision_at_1": safe_divide(hits_at_1, n),
        "recall_at_1": safe_divide(hits_at_1, n),
        "precision_at_5": safe_divide(hits_at_5, 5 * n),
        "recall_at_5": safe_divide(hits_at_5, n),
        "mrr_at_5": safe_divide(rr_sum, n),
        "local_hits_at_1": local_hits_at_1,
        "local_hits_at_5": local_hits_at_5,
        "local_recall_at_1": safe_divide(local_hits_at_1, n),
        "local_recall_at_5": safe_divide(local_hits_at_5, n),
        "local_mrr_at_5": safe_divide(local_rr_sum, n),
        "top1_not_in_local_examples": advisor_expanded_name_examples[:5],
    }


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return lines


def format_review(
    profile_results: list[tuple[Path, dict[str, object], bool]],
    paper_results: list[tuple[Path, dict[str, object]]],
) -> str:
    profile_rows = []
    local_reference_rows = []
    for path, metrics, ranking in profile_results:
        if ranking:
            profile_rows.append(
                (
                    path.name,
                    metrics["n"],
                    "ranking",
                    f"{metrics['hits_at_1']}/{metrics['n']}",
                    f"{metrics['recall_at_1']:.3f}",
                    f"{metrics['hits_at_5']}/{metrics['n']}",
                    f"{metrics['recall_at_5']:.3f}",
                    f"{metrics['mrr_at_5']:.3f}",
                    metrics["missing_predictions"],
                )
            )
        else:
            profile_rows.append(
                (
                    path.name,
                    metrics["n"],
                    "single advisor",
                    f"{metrics['hits_at_1']}/{metrics['n']}",
                    f"{metrics['recall_at_1']:.3f}",
                    "",
                    "",
                    "",
                    metrics["missing_predictions"],
                )
            )
        local_reference_rows.append(
            (
                path.name,
                f"{metrics['local_hits_at_1']}/{metrics['n']}",
                f"{metrics['local_recall_at_1']:.3f}",
                f"{metrics['local_hits_at_5']}/{metrics['n']}",
                f"{metrics['local_recall_at_5']:.3f}",
                f"{metrics['local_mrr_at_5']:.3f}",
            )
        )

    paper_rows = [
        (
            path.name,
            metrics["n"],
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['advisor_precision']:.3f}",
            f"{metrics['advisor_recall']:.3f}",
            f"{metrics['advisor_f1']:.3f}",
            metrics["labels"],
        )
        for path, metrics in paper_results
    ]

    issues = []
    for path, metrics, _ in profile_results:
        if metrics["n"] != 1194:
            issues.append(f"- `{path.name}` has {metrics['n']} rows; expected 1194 profile rows.")
        if metrics["missing_predictions"]:
            issues.append(f"- `{path.name}` has {metrics['missing_predictions']} rows with no parsed advisor prediction.")
        examples = metrics.get("top1_not_in_local_examples") or []
        if examples:
            issues.append(
                f"- `{path.name}` has top-1 outputs not matching local candidate spelling, e.g. {examples}."
            )
    for path, metrics in paper_results:
        if metrics["n"] != 1000:
            issues.append(f"- `{path.name}` has {metrics['n']} rows; expected 1000 paper rows.")
        if metrics["parse_missing"]:
            issues.append(f"- `{path.name}` has {metrics['parse_missing']} rows with missing parsed YES/NO/UNCERTAIN label.")

    lines = [
        "# Malik Result Review",
        "",
        "## Paper-Level Classification",
        "",
        *markdown_table(
            [
                "File",
                "N",
                "Rel. Precision",
                "Rel. Recall",
                "Rel. F1",
                "Advisor Precision",
                "Advisor Recall",
                "Advisor F1",
                "Labels",
            ],
            paper_rows,
        ),
        "",
        "## Profile-Level Outputs",
        "",
        *markdown_table(
            [
                "File",
                "N",
                "Output Type",
                "Hits@1",
                "Recall@1",
                "Hits@5",
                "Recall@5",
                "MRR@5",
                "Missing",
            ],
            profile_rows,
        ),
        "",
        "## Local Candidate Baseline In Same Profile Files",
        "",
        *markdown_table(
            ["File", "Local Hits@1", "Local Recall@1", "Local Hits@5", "Local Recall@5", "Local MRR@5"],
            local_reference_rows,
        ),
        "",
        "## Review Notes",
        "",
    ]
    if issues:
        lines.extend(issues)
    else:
        lines.append("- No structural issues found.")
    lines.extend(
        [
            "",
            "Scoring note: advisor-name matching uses the same normalized exact-name comparison as the main repo metrics.",
            "Expanded web-search names such as full first names may be counted as mismatches unless they normalize to the same string.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Malik-format mentorship output JSON files.")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_paths = sorted(args.profile_dir.glob("*.json"))
    paper_paths = sorted(args.paper_dir.glob("*.json"))

    profile_results = [
        (path, compute_profile_metrics(load_json_array(path), is_ranking_profile_file(path)), is_ranking_profile_file(path))
        for path in profile_paths
    ]
    paper_results = [
        (path, compute_paper_metrics(load_json_array(path)))
        for path in paper_paths
    ]
    report = format_review(profile_results, paper_results)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
