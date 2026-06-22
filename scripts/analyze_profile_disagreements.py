#!/usr/bin/env python3
"""Analyze instance-level disagreements for profile-ranking experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from compute_ranking_metrics import normalize_name, reciprocal_rank, topk_hit
from run_mentorship_benchmark import ProfileRankingRecord, load_profile_ranking_dataset


DEFAULT_NANO = "results/profile_level/results-profile-ranking-updated-gpt-5.4-nano-all.jsonl"
DEFAULT_MINI = "results/profile_level/results-profile-ranking-updated-gpt-5.4-mini-all.jsonl"
DEFAULT_TSV = "data/processed/llm_testing_set_advisee_complete_profile_updated.tsv"
DEFAULT_PREFIX = "analysis/disagreement/case_study_profile_disagreements"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON in {path} line {line_number}: {error}") from error
    return rows


def index_by_advisee_id(rows: Sequence[dict[str, object]], source: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        advisee_id = str(row.get("advisee_id") or "").strip()
        if not advisee_id:
            raise SystemExit(f"{source} contains a row without advisee_id.")
        if advisee_id in indexed:
            duplicate_ids.append(advisee_id)
        indexed[advisee_id] = row
    if duplicate_ids:
        raise SystemExit(f"{source} contains duplicate advisee_id values: {duplicate_ids[:5]}")
    return indexed


def index_records(records: Sequence[ProfileRankingRecord]) -> dict[str, ProfileRankingRecord]:
    indexed: dict[str, ProfileRankingRecord] = {}
    for record in records:
        if record.advisee_id:
            indexed[record.advisee_id] = record
    return indexed


def evaluation(row: dict[str, object]) -> dict[str, object]:
    value = row.get("evaluation") or {}
    if not isinstance(value, dict):
        return {}
    return value


def parsed(row: dict[str, object]) -> dict[str, object]:
    value = row.get("parsed") or {}
    if not isinstance(value, dict):
        return {}
    return value


def ranked_names(row: dict[str, object]) -> list[str]:
    return [str(name) for name in evaluation(row).get("ranked_advisors") or []]


def local_names(row: dict[str, object]) -> list[str]:
    return [str(name) for name in evaluation(row).get("local_model_top5") or []]


def top1(row: dict[str, object]) -> str:
    return str(evaluation(row).get("top1_advisor") or "")


def top1_correct(row: dict[str, object]) -> bool:
    return bool(evaluation(row).get("top1_match"))


def local_top1_correct(row: dict[str, object]) -> bool:
    return bool(evaluation(row).get("local_model_top1_match"))


def model_confidence(row: dict[str, object]) -> str:
    return str(parsed(row).get("top1_confidence") or "")


def model_source(row: dict[str, object]) -> str:
    return str(parsed(row).get("top1_source") or "")


def model_reasoning(row: dict[str, object]) -> str:
    items = parsed(row).get("ranked_advisors") or []
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict) and int(item.get("rank") or 0) == 1:
            return str(item.get("reasoning") or "")
    return ""


def rank_of_name(candidates: Sequence[str], target: str, max_rank: int | None = None) -> int | None:
    limit = max_rank or len(candidates)
    target_normalized = normalize_name(target)
    for rank, candidate in enumerate(candidates[:limit], start=1):
        if normalize_name(candidate) == target_normalized:
            return rank
    return None


def parse_probability(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probability_at_rank(record: ProfileRankingRecord, rank: int | None) -> float | None:
    if not rank or rank < 1 or rank > len(record.model_probabilities):
        return None
    return parse_probability(record.model_probabilities[rank - 1])


def paper_authors(paper_text_authors: Sequence[str], raw_text: str) -> list[str]:
    if paper_text_authors:
        return [str(author) for author in paper_text_authors]

    _, separator, authors_text = raw_text.partition("Authors:")
    if not separator:
        return []

    authors_text = authors_text.strip().strip('"').strip()
    matches = list(re.finditer(r"\(\d+\)\s*", authors_text))
    authors: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(authors_text)
        author = authors_text[start:end].strip().strip(",;")
        if author:
            authors.append(author)
    return authors


def paper_features(record: ProfileRankingRecord) -> dict[str, object]:
    distinct_coauthors: set[str] = set()
    years: list[int] = []
    gt_shared_count = 0
    gt_last_author_count = 0
    gt_first_author_count = 0
    gt_earliest_year: int | None = None
    advisee_first_author_count = 0

    gt_normalized = normalize_name(record.advisor_name)
    advisee_normalized = normalize_name(record.advisee_name)

    for paper in record.papers:
        authors = paper_authors(paper.authors, paper.raw_text)
        try:
            years.append(int(paper.year))
        except (TypeError, ValueError):
            pass

        for author in authors:
            author_normalized = normalize_name(author)
            if author_normalized and author_normalized != advisee_normalized:
                distinct_coauthors.add(author_normalized)

        if authors and normalize_name(authors[0]) == advisee_normalized:
            advisee_first_author_count += 1

        gt_in_paper = any(normalize_name(author) == gt_normalized for author in authors)
        if gt_in_paper:
            gt_shared_count += 1
            try:
                year = int(paper.year)
                gt_earliest_year = year if gt_earliest_year is None else min(gt_earliest_year, year)
            except (TypeError, ValueError):
                pass
            if authors and normalize_name(authors[0]) == gt_normalized:
                gt_first_author_count += 1
            if authors and normalize_name(authors[-1]) == gt_normalized:
                gt_last_author_count += 1

    return {
        "profile_paper_count": len(record.papers),
        "distinct_coauthor_count": len(distinct_coauthors),
        "earliest_paper_year": min(years) if years else "",
        "latest_paper_year": max(years) if years else "",
        "gt_shared_paper_count": gt_shared_count,
        "gt_last_author_count": gt_last_author_count,
        "gt_first_author_count": gt_first_author_count,
        "gt_earliest_shared_year": gt_earliest_year or "",
        "advisee_first_author_count": advisee_first_author_count,
    }


def bool_text(value: bool) -> str:
    return "YES" if value else "NO"


def join_list(values: Sequence[object]) -> str:
    return " | ".join(str(value) for value in values if str(value))


def disagreement_labels(local_correct: bool, nano_correct: bool, mini_correct: bool) -> list[str]:
    labels: list[str] = []
    if local_correct and nano_correct and mini_correct:
        labels.append("all_correct")
    if not local_correct and not nano_correct and not mini_correct:
        labels.append("all_wrong")
    if local_correct and not nano_correct and not mini_correct:
        labels.append("local_only_correct")
    if nano_correct and not local_correct and not mini_correct:
        labels.append("nano_only_correct")
    if mini_correct and not local_correct and not nano_correct:
        labels.append("mini_only_correct")
    if nano_correct and mini_correct and not local_correct:
        labels.append("both_llms_correct_local_wrong")
    if local_correct and not nano_correct and not mini_correct:
        labels.append("local_correct_both_llms_wrong")
    if nano_correct != mini_correct:
        labels.append("nano_mini_disagree")
    if nano_correct and not mini_correct:
        labels.append("nano_correct_mini_wrong")
    if mini_correct and not nano_correct:
        labels.append("mini_correct_nano_wrong")
    if (nano_correct or mini_correct) and not local_correct:
        labels.append("at_least_one_llm_correct_local_wrong")
    if local_correct and (not nano_correct or not mini_correct):
        labels.append("local_correct_at_least_one_llm_wrong")
    return labels


def build_case_rows(
    nano_rows: Sequence[dict[str, object]],
    mini_rows: Sequence[dict[str, object]],
    records: Sequence[ProfileRankingRecord],
) -> list[dict[str, object]]:
    nano_by_id = index_by_advisee_id(nano_rows, "nano JSONL")
    mini_by_id = index_by_advisee_id(mini_rows, "mini JSONL")
    records_by_id = index_records(records)

    common_ids = sorted(set(nano_by_id) & set(mini_by_id))
    missing_in_mini = sorted(set(nano_by_id) - set(mini_by_id))
    missing_in_nano = sorted(set(mini_by_id) - set(nano_by_id))
    if missing_in_mini or missing_in_nano:
        raise SystemExit(
            "Nano and mini result files do not contain the same advisee IDs. "
            f"Missing in mini: {missing_in_mini[:5]}; missing in nano: {missing_in_nano[:5]}"
        )

    case_rows: list[dict[str, object]] = []
    for advisee_id in common_ids:
        nano = nano_by_id[advisee_id]
        mini = mini_by_id[advisee_id]
        record = records_by_id.get(advisee_id)
        if record is None:
            raise SystemExit(f"TSV is missing advisee_id {advisee_id}")

        ground_truth = str(evaluation(nano).get("ground_truth_mentor") or record.advisor_name)
        local_top5 = local_names(nano)
        local_top1 = str(evaluation(nano).get("local_model_top1") or (local_top5[0] if local_top5 else ""))
        local_gt_rank = rank_of_name(local_top5, ground_truth, max_rank=5)
        local_top1_prob = probability_at_rank(record, 1)
        gt_rank_from_tsv = None
        try:
            gt_rank_from_tsv = int(record.advisor_model_rank)
        except (TypeError, ValueError):
            gt_rank_from_tsv = local_gt_rank
        gt_prob = probability_at_rank(record, gt_rank_from_tsv)

        nano_ranked = ranked_names(nano)
        mini_ranked = ranked_names(mini)
        local_correct = local_top1_correct(nano)
        nano_correct = top1_correct(nano)
        mini_correct = top1_correct(mini)
        labels = disagreement_labels(local_correct, nano_correct, mini_correct)

        nano_rr = reciprocal_rank(nano_ranked, ground_truth, 5)
        mini_rr = reciprocal_rank(mini_ranked, ground_truth, 5)
        local_rr = reciprocal_rank(local_top5, ground_truth, 5)

        row: dict[str, object] = {
            "advisee_id": advisee_id,
            "advisee_name": record.advisee_name,
            "ground_truth_advisor": ground_truth,
            "disagreement_labels": join_list(labels),
            "local_top1_correct": bool_text(local_correct),
            "nano_top1_correct": bool_text(nano_correct),
            "mini_top1_correct": bool_text(mini_correct),
            "local_top1": local_top1,
            "nano_top1": top1(nano),
            "mini_top1": top1(mini),
            "local_top5": join_list(local_top5),
            "nano_top5": join_list(nano_ranked),
            "mini_top5": join_list(mini_ranked),
            "local_gt_rank_top5": local_gt_rank or "",
            "nano_gt_rank_top5": rank_of_name(nano_ranked, ground_truth, max_rank=5) or "",
            "mini_gt_rank_top5": rank_of_name(mini_ranked, ground_truth, max_rank=5) or "",
            "local_reciprocal_rank_at5": f"{local_rr:.3f}",
            "nano_reciprocal_rank_at5": f"{nano_rr:.3f}",
            "mini_reciprocal_rank_at5": f"{mini_rr:.3f}",
            "gt_in_local_top5": bool_text(topk_hit(local_top5, ground_truth, 5)),
            "gt_in_nano_top5": bool_text(topk_hit(nano_ranked, ground_truth, 5)),
            "gt_in_mini_top5": bool_text(topk_hit(mini_ranked, ground_truth, 5)),
            "nano_top1_within_local_top5": bool_text(
                bool(evaluation(nano).get("llm_top1_within_local_top5"))
            ),
            "mini_top1_within_local_top5": bool_text(
                bool(evaluation(mini).get("llm_top1_within_local_top5"))
            ),
            "nano_top1_confidence": model_confidence(nano),
            "mini_top1_confidence": model_confidence(mini),
            "nano_top1_source": model_source(nano),
            "mini_top1_source": model_source(mini),
            "nano_top1_reasoning": model_reasoning(nano),
            "mini_top1_reasoning": model_reasoning(mini),
            "local_top1_probability": "" if local_top1_prob is None else f"{local_top1_prob:.6f}",
            "gt_probability": "" if gt_prob is None else f"{gt_prob:.6f}",
            "local_top1_minus_gt_probability": (
                ""
                if local_top1_prob is None or gt_prob is None
                else f"{local_top1_prob - gt_prob:.6f}"
            ),
            "gt_advisor_model_rank_tsv": record.advisor_model_rank,
            "pmids": join_list(nano.get("pmids") or []),
            "manual_error_pattern": "",
            "manual_notes": "",
        }
        row.update(paper_features(record))
        case_rows.append(row)

    return case_rows


def count_rows(rows: Sequence[dict[str, object]], label: str) -> int:
    return sum(label in str(row.get("disagreement_labels") or "").split(" | ") for row in rows)


def candidate_availability_counts(rows: Sequence[dict[str, object]]) -> list[tuple[str, int]]:
    gt_in_local_top5 = [row for row in rows if row["gt_in_local_top5"] == "YES"]
    gt_outside_local_top5 = [row for row in rows if row["gt_in_local_top5"] == "NO"]
    return [
        ("ground truth in local top 1", sum(row["local_top1_correct"] == "YES" for row in rows)),
        ("ground truth in local top 5", len(gt_in_local_top5)),
        ("ground truth outside local top 5", len(gt_outside_local_top5)),
        (
            "nano correct when ground truth in local top 5",
            sum(row["nano_top1_correct"] == "YES" for row in gt_in_local_top5),
        ),
        (
            "nano correct when ground truth outside local top 5",
            sum(row["nano_top1_correct"] == "YES" for row in gt_outside_local_top5),
        ),
        (
            "mini correct when ground truth in local top 5",
            sum(row["mini_top1_correct"] == "YES" for row in gt_in_local_top5),
        ),
        (
            "mini correct when ground truth outside local top 5",
            sum(row["mini_top1_correct"] == "YES" for row in gt_outside_local_top5),
        ),
    ]


def representative_cases(rows: Sequence[dict[str, object]], per_group: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    groups = [
        "all_correct",
        "both_llms_correct_local_wrong",
        "local_correct_both_llms_wrong",
        "nano_correct_mini_wrong",
        "mini_correct_nano_wrong",
        "all_wrong",
    ]

    for group in groups:
        candidates = [
            row
            for row in rows
            if group in str(row.get("disagreement_labels") or "").split(" | ")
            and str(row["advisee_id"]) not in seen
        ]
        candidates.sort(
            key=lambda row: (
                row["gt_in_local_top5"] != "YES",
                -int(row["profile_paper_count"]),
                str(row["advisee_id"]),
            )
        )
        for row in candidates[:per_group]:
            seen.add(str(row["advisee_id"]))
            case = dict(row)
            case["sample_group"] = group
            selected.append(case)
    return selected


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return lines


def format_summary(
    rows: Sequence[dict[str, object]],
    nano_path: Path,
    mini_path: Path,
    tsv_path: Path,
    representative: Sequence[dict[str, object]],
) -> str:
    n = len(rows)
    group_rows = [
        ("all correct", count_rows(rows, "all_correct")),
        ("all wrong", count_rows(rows, "all_wrong")),
        ("local only correct", count_rows(rows, "local_only_correct")),
        ("nano only correct", count_rows(rows, "nano_only_correct")),
        ("mini only correct", count_rows(rows, "mini_only_correct")),
        ("both LLMs correct, local wrong", count_rows(rows, "both_llms_correct_local_wrong")),
        ("local correct, both LLMs wrong", count_rows(rows, "local_correct_both_llms_wrong")),
        ("nano and mini disagree", count_rows(rows, "nano_mini_disagree")),
    ]
    availability_rows = candidate_availability_counts(rows)

    top1_rows = [
        ("local trained model", sum(row["local_top1_correct"] == "YES" for row in rows)),
        ("gpt-5.4-nano", sum(row["nano_top1_correct"] == "YES" for row in rows)),
        ("gpt-5.4-mini", sum(row["mini_top1_correct"] == "YES" for row in rows)),
    ]

    source_counter = Counter()
    for row in rows:
        source_counter[f"nano:{row['nano_top1_source'] or 'MISSING'}"] += 1
        source_counter[f"mini:{row['mini_top1_source'] or 'MISSING'}"] += 1

    lines = [
        "# Profile-Ranking Case Study Disagreement Analysis",
        "",
        "## Inputs",
        "",
        f"- Nano results: `{nano_path}`",
        f"- Mini results: `{mini_path}`",
        f"- Profile data: `{tsv_path}`",
        f"- Merged instances: {n}",
        "",
        "Definition note: top-1 counts use the explicit `evaluation.top1_match` field from each run. "
        "This is the final selected advisor, which can differ from the first parsed ranked-list item in a few malformed outputs.",
        "",
        "## Top-1 Accuracy Counts",
        "",
        *markdown_table(
            ["Model", "Correct / N", "Accuracy"],
            [(name, f"{correct} / {n}", f"{correct / n:.3f}" if n else "0.000") for name, correct in top1_rows],
        ),
        "",
        "## Disagreement Counts",
        "",
        *markdown_table(
            ["Group", "Count", "Share"],
            [(name, count, f"{count / n:.3f}" if n else "0.000") for name, count in group_rows],
        ),
        "",
        "## Candidate Availability",
        "",
        *markdown_table(
            ["Condition", "Count", "Share"],
            [(name, count, f"{count / n:.3f}" if n else "0.000") for name, count in availability_rows],
        ),
        "",
        "## LLM Source Labels",
        "",
        *markdown_table(
            ["Source", "Count"],
            sorted(source_counter.items()),
        ),
        "",
        "## Representative Cases",
        "",
        *markdown_table(
            [
                "Group",
                "Advisee",
                "Ground Truth",
                "Local Top-1",
                "Nano Top-1",
                "Mini Top-1",
                "Labels",
            ],
            [
                (
                    row["sample_group"],
                    row["advisee_name"],
                    row["ground_truth_advisor"],
                    row["local_top1"],
                    row["nano_top1"],
                    row["mini_top1"],
                    row["disagreement_labels"],
                )
                for row in representative
            ],
        ),
        "",
        "## Notes For Manual Coding",
        "",
        "Use `manual_error_pattern` and `manual_notes` in the exported representative CSV to record qualitative labels.",
        "Suggested labels are listed in `case_study_disagreement_analysis_plan.md`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build disagreement tables for profile-ranking model comparisons."
    )
    parser.add_argument("--nano-jsonl", default=DEFAULT_NANO, help="Nano profile-ranking JSONL.")
    parser.add_argument("--mini-jsonl", default=DEFAULT_MINI, help="Mini profile-ranking JSONL.")
    parser.add_argument("--profile-tsv", default=DEFAULT_TSV, help="Profile-ranking TSV.")
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_PREFIX,
        help="Prefix for generated CSV and Markdown files.",
    )
    parser.add_argument(
        "--representatives-per-group",
        type=int,
        default=3,
        help="Number of representative cases to select for each disagreement group.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    nano_path = Path(args.nano_jsonl)
    mini_path = Path(args.mini_jsonl)
    tsv_path = Path(args.profile_tsv)
    output_prefix = Path(args.output_prefix)

    rows = build_case_rows(
        load_jsonl(nano_path),
        load_jsonl(mini_path),
        load_profile_ranking_dataset(tsv_path),
    )
    representatives = representative_cases(rows, args.representatives_per_group)

    table_path = output_prefix.with_name(f"{output_prefix.name}_full_table.csv")
    representative_path = output_prefix.with_name(f"{output_prefix.name}_representative_cases.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.md")

    write_csv(table_path, rows)
    write_csv(representative_path, representatives)
    summary_path.write_text(
        format_summary(rows, nano_path, mini_path, tsv_path, representatives),
        encoding="utf-8",
    )

    print(f"Wrote {table_path}")
    print(f"Wrote {representative_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
