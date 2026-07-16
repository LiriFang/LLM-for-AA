#!/usr/bin/env python3
"""Run advisor-advisee prompting experiments against the OpenAI Responses API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import compute_classification_metrics
import compute_ranking_metrics

try:
    from openai import BadRequestError, OpenAI
except ImportError:  # pragma: no cover - handled at runtime.
    BadRequestError = None
    OpenAI = None


PAPER_MODES = {
    "paper-web-search",
    "paper-no-search",
    "paper-known-mentee",
    "paper-known-mentee-v2",
}

PROFILE_MODES = {
    "profile-no-search",
    "profile-web-search",
    "profile-ranking",
    "profile-ranking-web-search",
    "profile-ranking-v2",
    "profile-reranking-v2",
}

RANKING_MODES = {
    "profile-ranking",
    "profile-ranking-web-search",
    "profile-ranking-v2",
    "profile-reranking-v2",
}

DEFAULT_PROFILE_RANKING_TEMPLATE_V2 = Path("prompts/profile_ranking_prompt_template_v2.txt")
DEFAULT_PROFILE_RERANKING_TEMPLATE_V2 = Path("prompts/profile_rereanking_prompt_template_v2.txt")
DEFAULT_CLASSIFICATION_TEMPLATE_V2 = Path("prompts/classification_prompt_template_with_prof_v2.txt")
APPEND_LOCK = threading.Lock()


@dataclass
class PaperRecord:
    row_number: int
    doi: str
    pmid: str
    year: str
    title: str
    journal: str
    authors_raw: str
    authors: list[str]
    advisee_name: str
    advisor_name: str
    has_mentor: bool
    raw_text: str = ""

    def paper_text(self) -> str:
        if self.raw_text:
            return self.raw_text

        ordered_authors = ", ".join(
            f"({index}) {author}" for index, author in enumerate(self.authors, start=1)
        )
        return (
            f'Paper: {self.title}; {self.journal}; ({self.year}); '
            f'PMID: {self.pmid}; Authors: {ordered_authors}'
        )


@dataclass
class ProfileRankingRecord:
    row_number: int
    advisee_id: str
    advisee_name: str
    papers: list[PaperRecord]
    model_candidate_ids: list[str]
    model_candidates: list[str]
    model_probabilities: list[str]
    advisor_id: str
    advisor_name: str
    advisor_model_rank: str

    @property
    def has_mentor(self) -> bool:
        return normalize_name(self.advisor_name) != normalize_name("no_mentor")


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def clean_year(raw_year: str) -> str:
    raw_year = (raw_year or "").strip()
    if raw_year.endswith(".0"):
        return raw_year[:-2]
    return raw_year


def parse_authors(authors_raw: str) -> list[str]:
    authors = [author.strip() for author in (authors_raw or "").split(",")]
    return [author for author in authors if author]


def load_dataset(csv_path: Path) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_number, row in enumerate(reader, start=1):
            authors_raw = (row.get("authors") or "").strip()
            advisor_name = (
                row.get("advisor_name")
                or row.get("GTruth_Advisor_name")
                or ""
            ).strip()
            has_advisor_raw = (row.get("has_advisor") or "").strip().casefold()
            if has_advisor_raw in {"true", "1", "yes"}:
                has_mentor = True
            elif has_advisor_raw in {"false", "0", "no"}:
                has_mentor = False
                if not advisor_name:
                    advisor_name = "no_mentor"
            else:
                has_mentor = normalize_name(advisor_name) != normalize_name("no_mentor")

            raw_detail = (row.get("detail") or "").strip().strip('"').strip()
            raw_detail = re.sub(r"\s+", " ", raw_detail)
            record = PaperRecord(
                row_number=row_number,
                doi=(row.get("doi") or "").strip(),
                pmid=(row.get("PMID") or "").strip(),
                year=clean_year(row.get("year") or row.get("Year") or ""),
                title=(row.get("title") or "").strip(),
                journal=(row.get("journal") or "").strip(),
                authors_raw=authors_raw,
                authors=parse_authors(authors_raw),
                advisee_name=(row.get("advisee_name") or row.get("Advisee_name") or "").strip(),
                advisor_name=advisor_name,
                has_mentor=has_mentor,
                raw_text=raw_detail,
            )
            records.append(record)
    return records


def split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def parse_profile_papers(
    profile_text: str,
    row_number: int,
    advisee_name: str,
    advisor_name: str,
) -> list[PaperRecord]:
    text = (profile_text or "").strip()
    starts = [match.start() for match in re.finditer(r'"?\s*Paper:', text)]
    papers: list[PaperRecord] = []

    for index, start in enumerate(starts, start=1):
        end = starts[index] if index < len(starts) else len(text)
        raw_text = text[start:end].strip().strip('"').strip()
        raw_text = re.sub(r"\s+", " ", raw_text)
        if not raw_text:
            continue
        pmid_match = re.search(r"PMID:\s*([^;]+)", raw_text)
        year_match = re.search(r"\((\d{4})\)", raw_text)
        papers.append(
            PaperRecord(
                row_number=row_number,
                doi="",
                pmid=(pmid_match.group(1).strip() if pmid_match else ""),
                year=(year_match.group(1).strip() if year_match else ""),
                title="",
                journal="",
                authors_raw="",
                authors=[],
                advisee_name=advisee_name,
                advisor_name=advisor_name,
                has_mentor=normalize_name(advisor_name) != normalize_name("no_mentor"),
                raw_text=raw_text,
            )
        )

    return papers


def load_profile_ranking_dataset(tsv_path: Path) -> list[ProfileRankingRecord]:
    records: list[ProfileRankingRecord] = []
    with tsv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {
            "Advisee_name",
            "Advisee_pubs_profile",
            "Model_predicted_predicted_CoAUID",
            "Model_predicted_ranked_coauthor_name",
            "Probability",
            "GTruth_Advisor_id",
            "GTruth_Advisor_name",
            "GT_advisor_model_rank",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if "Advisee_id" not in (reader.fieldnames or []) and "AUID" not in (reader.fieldnames or []):
            missing_columns.add("Advisee_id or AUID")
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SystemExit(
                f"{tsv_path} is missing required profile-ranking column(s): {missing}"
            )

        for row_number, row in enumerate(reader, start=1):
            advisee_name = (row.get("Advisee_name") or "").strip()
            advisor_name = (row.get("GTruth_Advisor_name") or "").strip()
            records.append(
                ProfileRankingRecord(
                    row_number=row_number,
                    advisee_id=(row.get("Advisee_id") or row.get("AUID") or "").strip(),
                    advisee_name=advisee_name,
                    papers=parse_profile_papers(
                        row.get("Advisee_pubs_profile") or "",
                        row_number=row_number,
                        advisee_name=advisee_name,
                        advisor_name=advisor_name,
                    ),
                    model_candidate_ids=split_csv_list(
                        row.get("Model_predicted_predicted_CoAUID") or ""
                    ),
                    model_candidates=split_csv_list(
                        row.get("Model_predicted_ranked_coauthor_name") or ""
                    ),
                    model_probabilities=split_csv_list(row.get("Probability") or ""),
                    advisor_id=(row.get("GTruth_Advisor_id") or "").strip(),
                    advisor_name=advisor_name,
                    advisor_model_rank=(row.get("GT_advisor_model_rank") or "").strip(),
                )
            )
    return records


def build_prompt(
    mode: str,
    papers: Sequence[PaperRecord],
    profile_owner: str | None = None,
    collect_evidence: bool = False,
    template_path: Path | None = None,
) -> str:
    if not papers:
        raise ValueError("No papers supplied for prompt construction.")
    if mode == "paper-known-mentee-v2":
        return build_classification_template_prompt(
            papers=papers,
            profile_owner=profile_owner,
            template_path=template_path or DEFAULT_CLASSIFICATION_TEMPLATE_V2,
        )

    allow_web_search = mode in {"paper-web-search", "profile-web-search"}
    known_mentee = mode in {"paper-known-mentee", "profile-no-search", "profile-web-search"}
    asks_for_mentee_output = mode in {"paper-web-search", "paper-no-search"}

    instructions = [
        "You are an expert bibliometric analyst specializing in PhD advisor detection from publication metadata.",
        "Your task is to make a calibrated benchmark prediction about whether the supplied publication metadata supports a PhD advisor relationship.",
        "Reason step by step internally, but return only the requested output fields.",
        "Never output a name that is not present in the supplied author list.",
        "Do not use UNCERTAIN as a safe fallback; use it only when the evidence is genuinely unusable or two advisor candidates are essentially tied.",
    ]

    if allow_web_search:
        instructions.append(
            "You may use web search only to verify public evidence such as faculty pages, lab pages, university bios, dissertation pages, or CVs."
        )
    else:
        instructions.append(
            "Do not use web search or any external tools; rely only on the supplied metadata and bibliometric heuristics."
        )

    if mode in PAPER_MODES:
        instructions.append("You will receive one publication.")
    else:
        instructions.append("You will receive one advisee name plus one or more publications from that advisee's profile.")

    if known_mentee:
        instructions.append("Treat the supplied advisee as fixed and only decide whether a PhD advisor relationship is present and who the PhD advisor is.")
    else:
        instructions.append("If a PhD advisor relationship is present, identify both the PhD advisee and the PhD advisor from the author list.")

    instructions.extend(
        [
            "",
            "Use the following evidence hierarchy, from strongest to weakest:",
            "1. Direct external verification in public sources, if web search is allowed.",
            "2. Repeated collaboration across the advisee's earliest papers with one recurring senior author.",
            "3. Senior-author patterns: in many biomedical fields the advisor is often the last author, while the advisee is often an early-career first author.",
            "4. Career-stage asymmetry: the advisee is usually the more junior author, while the advisor is usually the established investigator or lab head.",
            "5. Single-paper author order hints are weak and must be treated cautiously, especially outside biomedicine.",
            "",
            "Apply these constraints:",
            "- Identify the PhD advisor specifically, not merely the most senior, most famous, or most recent collaborator.",
            "- Prefer a recurring senior collaborator over a coauthor who appears only once.",
            "- Prefer the advisor linked to the earliest training-period papers over a later supervisor, collaborator, or postdoctoral mentor.",
            "- Do not choose a likely peer or another junior author as advisor just because they are adjacent in the author list.",
            "- In fields with alphabetical or weak author-order conventions, treat author position as weak evidence.",
            "- For two-author biomedical papers, first-author / last-author asymmetry is enough to support YES when the known advisee is first author and the other author is plausibly senior, unless the metadata suggests a peer relationship or no mentor relationship.",
            "- If multiple advisor candidates remain plausible, choose the best-supported candidate when one has even a modest advantage.",
            "",
            "Decision policy:",
            "- Output YES when the metadata gives a reasonable bibliometric signal for an advisor relationship, even if the evidence is not definitive.",
            "- Output NO when the known advisee is absent, the likely advisor is absent, the paper appears to involve peers/non-training collaborators, or no author is plausibly an advisor.",
            "- Output UNCERTAIN only when there is real ambiguity that prevents a useful benchmark prediction: no usable authorship signal, severe field-convention ambiguity, or two or more advisor candidates with nearly equal evidence.",
            "- In single-paper cases, prefer YES or NO based on the best bibliometric signal rather than abstaining because external proof is unavailable.",
            "",
            "Name selection rules:",
            "- Copy names exactly as listed in the supplied author list whenever possible.",
            "- For paper modes without a fixed advisee, do not label the likely senior investigator as the advisee unless the metadata strongly supports that interpretation.",
        ]
    )

    if collect_evidence:
        instructions.extend(
            [
                "",
                "Also provide structured decision evidence, but do not reveal hidden chain-of-thought.",
                "Use only concise observable evidence tags and a short justification grounded in the supplied metadata or approved external sources.",
                "Primary Evidence must be exactly one of: author_order, repeated_collaboration, temporal_profile, field_convention, external_source, insufficient_evidence.",
                "Secondary Evidence should be another short evidence tag if useful, otherwise NONE.",
                "Confidence must be exactly one of: HIGH, MEDIUM, LOW.",
                "Short Justification must be 1-2 sentences.",
            ]
        )

    prompt_parts = ["\n".join(instructions), "", "Input:"]

    if known_mentee:
        mentee = profile_owner or papers[0].advisee_name
        prompt_parts.append(f"PhD Advisee: {mentee}")

    for paper in papers:
        prompt_parts.append(f'"{paper.paper_text()}"')

    prompt_parts.extend(["", "Output Format:", "PhD Advisor Relationship: YES / NO / UNCERTAIN"])

    if asks_for_mentee_output:
        prompt_parts.append("If YES or UNCERTAIN")
        prompt_parts.append("PhD Advisee: [Full name as listed]")
        prompt_parts.append("PhD Advisor: [Full name as listed]")
    elif known_mentee:
        prompt_parts.append("If YES or UNCERTAIN")
        prompt_parts.append("PhD Advisor: [Full name as listed]")

    if collect_evidence:
        prompt_parts.append("Confidence: HIGH / MEDIUM / LOW")
        prompt_parts.append(
            "Primary Evidence: author_order / repeated_collaboration / temporal_profile / field_convention / external_source / insufficient_evidence"
        )
        prompt_parts.append("Secondary Evidence: [short tag or NONE]")
        prompt_parts.append("Short Justification: [1-2 sentences]")

    prompt_parts.append("Return only the output fields above with no extra commentary.")
    return "\n".join(prompt_parts)


def build_classification_template_prompt(
    papers: Sequence[PaperRecord],
    profile_owner: str | None,
    template_path: Path,
) -> str:
    if not papers:
        raise ValueError("No papers supplied for prompt construction.")
    template = template_path.read_text(encoding="utf-8")
    advisee_name = profile_owner or papers[0].advisee_name
    prompt = template.replace("[ADVISEE_NAME]", advisee_name)
    paper_block = "\n".join(f'"{paper.paper_text()}"' for paper in papers)
    prompt = prompt.replace(
        '"[PAPER_1_METADATA]"\n"[PAPER_2_METADATA]"\n...\n"[PAPER_N_METADATA]"',
        paper_block,
    )
    prompt = replace_numbered_placeholders(prompt, "PAPER", [paper.paper_text() for paper in papers])
    return prompt.strip() + "\n"


def build_profile_ranking_prompt(
    mode: str,
    record: ProfileRankingRecord,
    max_papers_per_profile: int,
    template_path: Path | None = None,
) -> str:
    if mode in {"profile-ranking-v2", "profile-reranking-v2"}:
        return build_profile_template_prompt(
            mode=mode,
            record=record,
            max_papers_per_profile=max_papers_per_profile,
            template_path=template_path,
        )

    allow_web_search = mode == "profile-ranking-web-search"
    papers = record.papers[:max_papers_per_profile]
    if not papers:
        raise ValueError("No papers supplied for prompt construction.")

    instructions = [
        "You are an expert bibliometric analyst specializing in academic mentorship networks in biomedical research.",
        "Your task is to infer the most probable PhD advisor for a fixed advisee from publication co-authorship metadata.",
        "Reason step by step internally, but return only the requested output fields.",
        "Never output a name that is not present in the supplied author metadata or model-predicted candidate list.",
        "A ranked candidate list is provided by a prior local model. Prefer advisors from that list when the publication evidence supports them.",
        "If none of the ranked candidates is convincing, you may select another author from the supplied papers, but mark that item as OVERRIDE.",
    ]

    if allow_web_search:
        instructions.append(
            "You may use web search only to verify public evidence such as faculty pages, lab pages, university bios, dissertation pages, or CVs."
        )
    else:
        instructions.append(
            "Do not use web search or external tools; rely only on the supplied metadata, candidate ranking, and bibliometric heuristics."
        )

    instructions.extend(
        [
            "",
            "Ranking policy:",
            "- Output a ranked list of up to 5 advisor candidates, ordered from most likely to least likely.",
            "- Rank 1 is the model's final selected advisor and will be evaluated as the top-1 prediction.",
            "- Use YES when the top-ranked advisor is a plausible PhD advisor; use NO only when no advisor relationship is supported.",
            "- Keep each reasoning field to one short sentence.",
            "- Copy candidate names exactly as listed whenever possible.",
        ]
    )

    prompt_parts = [
        "\n".join(instructions),
        "",
        "Input:",
        f"Advisee: {record.advisee_name}",
        "Model-Predicted Candidates (Rank 1-5):",
    ]

    for rank, candidate in enumerate(record.model_candidates[:5], start=1):
        probability = ""
        if rank <= len(record.model_probabilities):
            probability = f" (prior probability: {record.model_probabilities[rank - 1]})"
        prompt_parts.append(f"{rank}. {candidate}{probability}")

    prompt_parts.append("")
    prompt_parts.append("Papers:")
    for index, paper in enumerate(papers, start=1):
        prompt_parts.append(f'{index}. "{paper.paper_text()}"')

    prompt_parts.extend(
        [
            "",
            "Output Format:",
            "Advisor Detected: YES / NO",
            "Ranked Advisors:",
            "1. [Advisor name or NONE] | Confidence: HIGH / MEDIUM / LOW | Source: IN-LIST / OVERRIDE | Reasoning: [1 sentence]",
            "2. [Advisor name or NONE] | Confidence: HIGH / MEDIUM / LOW | Source: IN-LIST / OVERRIDE | Reasoning: [1 sentence]",
            "3. [Advisor name or NONE] | Confidence: HIGH / MEDIUM / LOW | Source: IN-LIST / OVERRIDE | Reasoning: [1 sentence]",
            "4. [Advisor name or NONE] | Confidence: HIGH / MEDIUM / LOW | Source: IN-LIST / OVERRIDE | Reasoning: [1 sentence]",
            "5. [Advisor name or NONE] | Confidence: HIGH / MEDIUM / LOW | Source: IN-LIST / OVERRIDE | Reasoning: [1 sentence]",
            "Top-1 Advisor: [same as ranked advisor 1, or NONE]",
            "Top-1 Confidence: HIGH / MEDIUM / LOW",
            "Top-1 Source: IN-LIST / OVERRIDE",
            "If any OVERRIDE item is used:",
            "Override Reason: [why the ranked candidate list was insufficient]",
            "Return only the output fields above with no extra commentary.",
        ]
    )
    return "\n".join(prompt_parts)


def default_profile_template_path(mode: str) -> Path:
    if mode == "profile-ranking-v2":
        return DEFAULT_PROFILE_RANKING_TEMPLATE_V2
    if mode == "profile-reranking-v2":
        return DEFAULT_PROFILE_RERANKING_TEMPLATE_V2
    raise ValueError(f"No default profile template for mode: {mode}")


def render_paper_list(papers: Sequence[PaperRecord]) -> str:
    return "\n".join(
        f'{index}. "{paper.paper_text()}"'
        for index, paper in enumerate(papers, start=1)
    )


def render_candidate_list(record: ProfileRankingRecord) -> str:
    lines: list[str] = []
    for rank in range(1, 6):
        candidate = record.model_candidates[rank - 1] if rank <= len(record.model_candidates) else "NONE"
        probability = record.model_probabilities[rank - 1] if rank <= len(record.model_probabilities) else "N/A"
        lines.append(f"{rank}. {candidate} (prior probability: {probability})")
    return "\n".join(lines)


def replace_numbered_placeholders(template: str, prefix: str, values: Sequence[str]) -> str:
    text = template
    for index, value in enumerate(values, start=1):
        text = text.replace(f"[{prefix}_{index}]", value)
    text = re.sub(rf"\[{re.escape(prefix)}_\d+\]", "NONE", text)
    return text


def build_profile_template_prompt(
    mode: str,
    record: ProfileRankingRecord,
    max_papers_per_profile: int,
    template_path: Path | None = None,
) -> str:
    papers = record.papers[:max_papers_per_profile]
    if not papers:
        raise ValueError("No papers supplied for prompt construction.")

    path = template_path or default_profile_template_path(mode)
    template = path.read_text(encoding="utf-8")

    replacements = {
        "[ADVISEE_NAME]": record.advisee_name,
        '1. "[PAPER_1_METADATA]"\n2. "[PAPER_2_METADATA]"\n3. "[PAPER_3_METADATA]"\n...\nN. "[PAPER_N_METADATA]"': render_paper_list(papers),
    }
    if mode == "profile-reranking-v2":
        replacements[
            "1. [CANDIDATE_1] (prior probability: [PROBABILITY_1])\n"
            "2. [CANDIDATE_2] (prior probability: [PROBABILITY_2])\n"
            "3. [CANDIDATE_3] (prior probability: [PROBABILITY_3])\n"
            "4. [CANDIDATE_4] (prior probability: [PROBABILITY_4])\n"
            "5. [CANDIDATE_5] (prior probability: [PROBABILITY_5])"
        ] = render_candidate_list(record)

    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    prompt = replace_numbered_placeholders(prompt, "PAPER", [paper.paper_text() for paper in papers])
    prompt = replace_numbered_placeholders(prompt, "CANDIDATE", record.model_candidates[:5])
    prompt = replace_numbered_placeholders(prompt, "PROBABILITY", record.model_probabilities[:5])
    return prompt.strip() + "\n"


def select_paper_records(
    records: Sequence[PaperRecord],
    row_numbers: Sequence[int],
    pmids: Sequence[str],
    advisees: Sequence[str],
    limit: int | None,
) -> list[PaperRecord]:
    selected = list(records)

    if row_numbers:
        wanted = set(row_numbers)
        selected = [record for record in selected if record.row_number in wanted]

    if pmids:
        wanted = {pmid.strip() for pmid in pmids}
        selected = [record for record in selected if record.pmid in wanted]

    if advisees:
        wanted = {normalize_name(name) for name in advisees}
        selected = [
            record for record in selected if normalize_name(record.advisee_name) in wanted
        ]

    if limit is not None:
        selected = selected[:limit]

    return selected


def select_profiles(
    records: Sequence[PaperRecord],
    row_numbers: Sequence[int],
    pmids: Sequence[str],
    advisees: Sequence[str],
    limit: int | None,
    max_papers_per_profile: int,
) -> list[tuple[str, list[PaperRecord]]]:
    if row_numbers or pmids or advisees:
        seed_rows = select_paper_records(
            records=records,
            row_numbers=row_numbers,
            pmids=pmids,
            advisees=advisees,
            limit=None,
        )
        target_advisees = {record.advisee_name for record in seed_rows}
        grouped: dict[str, list[PaperRecord]] = defaultdict(list)
        for record in records:
            if record.advisee_name in target_advisees:
                grouped[record.advisee_name].append(record)
    else:
        grouped = defaultdict(list)
        for record in records:
            grouped[record.advisee_name].append(record)

    profiles: list[tuple[str, list[PaperRecord]]] = []
    seen = set()
    for record in records:
        advisee_name = record.advisee_name
        if advisee_name in seen or advisee_name not in grouped:
            continue
        seen.add(advisee_name)
        papers = sorted(
            grouped[advisee_name],
            key=lambda item: (safe_int(item.year), item.row_number),
        )
        profiles.append((advisee_name, papers[:max_papers_per_profile]))

    if limit is not None:
        profiles = profiles[:limit]

    return profiles


def select_profile_ranking_records(
    records: Sequence[ProfileRankingRecord],
    row_numbers: Sequence[int],
    advisees: Sequence[str],
    limit: int | None,
) -> list[ProfileRankingRecord]:
    selected = list(records)

    if row_numbers:
        wanted = set(row_numbers)
        selected = [record for record in selected if record.row_number in wanted]

    if advisees:
        wanted = {normalize_name(name) for name in advisees}
        selected = [
            record for record in selected if normalize_name(record.advisee_name) in wanted
        ]

    if limit is not None:
        selected = selected[:limit]

    return selected


def safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def parse_response(mode: str, text: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "mentorship_detected": None,
        "mentee": None,
        "mentor": None,
        "confidence": None,
        "primary_evidence": None,
        "secondary_evidence": None,
        "short_justification": None,
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            lowered.startswith("phd advisor relationship:")
            or lowered.startswith("advisor relationship:")
            or lowered.startswith("mentorship detected:")
        ):
            result["mentorship_detected"] = line.split(":", 1)[1].strip().upper()
        elif lowered.startswith("phd advisee:") or lowered.startswith("mentee:"):
            result["mentee"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("phd advisor:") or lowered.startswith("advisor:") or lowered.startswith("mentor:"):
            result["mentor"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip().upper()
        elif lowered.startswith("primary evidence:"):
            result["primary_evidence"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("secondary evidence:"):
            result["secondary_evidence"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("short justification:"):
            result["short_justification"] = line.split(":", 1)[1].strip()

    if mode in {"paper-known-mentee", "profile-no-search", "profile-web-search"}:
        result["mentee"] = None

    return result


def parse_profile_ranking_response(text: str) -> dict[str, object]:
    result: dict[str, object] = {
        "advisor_detected": None,
        "ranked_advisors": [],
        "top1_advisor": None,
        "top1_confidence": None,
        "top1_source": None,
        "override_reason": None,
    }

    ranked_advisors: list[dict[str, str | int | None]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("advisor detected:"):
            result["advisor_detected"] = line.split(":", 1)[1].strip().upper()
            continue
        if lowered.startswith("top-1 advisor:") or lowered.startswith("top 1 advisor:"):
            result["top1_advisor"] = line.split(":", 1)[1].strip()
            continue
        if lowered.startswith("top-1 confidence:") or lowered.startswith("top 1 confidence:"):
            result["top1_confidence"] = line.split(":", 1)[1].strip().upper()
            continue
        if lowered.startswith("top-1 source:") or lowered.startswith("top 1 source:"):
            result["top1_source"] = line.split(":", 1)[1].strip().upper()
            continue
        if lowered.startswith("override reason:"):
            result["override_reason"] = line.split(":", 1)[1].strip()
            continue

        rank_match = re.match(r"^(\d+)[.)]\s*(.+)$", line)
        if not rank_match:
            continue

        rank = int(rank_match.group(1))
        remainder = rank_match.group(2).strip()
        pieces = [piece.strip() for piece in remainder.split("|")]
        advisor = pieces[0].strip()
        ranked_item: dict[str, str | int | None] = {
            "rank": rank,
            "advisor": advisor,
            "confidence": None,
            "source": None,
            "reasoning": None,
        }
        for piece in pieces[1:]:
            key, separator, value = piece.partition(":")
            if not separator:
                continue
            normalized_key = key.strip().lower()
            if normalized_key == "confidence":
                ranked_item["confidence"] = value.strip().upper()
            elif normalized_key == "source":
                ranked_item["source"] = value.strip().upper()
            elif normalized_key == "reasoning":
                ranked_item["reasoning"] = value.strip()
        ranked_advisors.append(ranked_item)

    ranked_advisors.sort(key=lambda item: int(item["rank"]))
    result["ranked_advisors"] = ranked_advisors

    if (
        (not result["top1_advisor"] or normalize_name(str(result["top1_advisor"])) == normalize_name("none"))
        and ranked_advisors
    ):
        result["top1_advisor"] = ranked_advisors[0]["advisor"]
    if (
        (not result["top1_confidence"] or normalize_name(str(result["top1_confidence"])) == normalize_name("none"))
        and ranked_advisors
    ):
        result["top1_confidence"] = ranked_advisors[0]["confidence"]
    if (
        (not result["top1_source"] or normalize_name(str(result["top1_source"])) == normalize_name("none"))
        and ranked_advisors
    ):
        result["top1_source"] = ranked_advisors[0]["source"]

    return result


def evaluate_result(
    mode: str,
    papers: Sequence[PaperRecord],
    parsed: dict[str, str | None],
) -> dict[str, object]:
    ground_truth_has_mentor = any(paper.has_mentor for paper in papers)
    ground_truth_mentor = next((paper.advisor_name for paper in papers if paper.has_mentor), "no_mentor")
    ground_truth_mentee = papers[0].advisee_name if papers else ""

    predicted_label = (parsed.get("mentorship_detected") or "").upper()
    predicted_has_mentor = predicted_label == "YES"
    predicted_is_uncertain = predicted_label == "UNCERTAIN"
    predicted_mentor = parsed.get("mentor") or ""
    predicted_mentee = parsed.get("mentee") or ""

    mentor_match = normalize_name(predicted_mentor) == normalize_name(ground_truth_mentor)
    mentee_match = True
    if mode in {"paper-web-search", "paper-no-search"}:
        mentee_match = normalize_name(predicted_mentee) == normalize_name(ground_truth_mentee)

    label_match = predicted_label == ("YES" if ground_truth_has_mentor else "NO")

    return {
        "ground_truth_has_mentor": ground_truth_has_mentor,
        "ground_truth_mentor": ground_truth_mentor,
        "ground_truth_mentee": ground_truth_mentee,
        "predicted_label": predicted_label,
        "predicted_has_mentor": predicted_has_mentor,
        "predicted_is_uncertain": predicted_is_uncertain,
        "mentor_match": mentor_match,
        "mentee_match": mentee_match,
        "label_match": label_match,
        "strict_match": label_match and mentor_match and mentee_match,
    }


def evaluate_profile_ranking_result(
    record: ProfileRankingRecord,
    parsed: dict[str, object],
) -> dict[str, object]:
    ground_truth_mentor = record.advisor_name
    ranked_items = parsed.get("ranked_advisors") or []
    ranked_advisors = [
        "" if normalize_name(str(item.get("advisor") or "")) == normalize_name("none") else str(item.get("advisor") or "")
        for item in ranked_items
        if isinstance(item, dict)
    ]
    top1_advisor = str(parsed.get("top1_advisor") or "")
    if normalize_name(top1_advisor) == normalize_name("none"):
        top1_advisor = ""

    local_top1 = record.model_candidates[0] if record.model_candidates else ""
    local_top5 = record.model_candidates[:5]

    return {
        "ground_truth_has_mentor": record.has_mentor,
        "ground_truth_mentor": ground_truth_mentor,
        "ground_truth_mentee": record.advisee_name,
        "ground_truth_model_rank": record.advisor_model_rank,
        "top1_advisor": top1_advisor,
        "top1_match": normalize_name(top1_advisor) == normalize_name(ground_truth_mentor),
        "ranked_advisors": ranked_advisors,
        "ranked_list_contains_ground_truth": any(
            normalize_name(advisor) == normalize_name(ground_truth_mentor)
            for advisor in ranked_advisors
        ),
        "local_model_top1": local_top1,
        "local_model_top1_match": normalize_name(local_top1) == normalize_name(ground_truth_mentor),
        "local_model_top5": local_top5,
        "local_model_top5_contains_ground_truth": any(
            normalize_name(candidate) == normalize_name(ground_truth_mentor)
            for candidate in local_top5
        ),
        "llm_top1_within_local_top5": any(
            normalize_name(top1_advisor) == normalize_name(candidate)
            for candidate in local_top5
        ),
    }


def require_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise SystemExit(
            "The `openai` package is not installed. Run `pip install -r requirements.txt` first."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running non-dry-run requests.")

    return OpenAI(api_key=api_key)


def call_openai(mode: str, model: str, prompt: str) -> tuple[str, object]:
    client = require_openai_client()
    request: dict[str, object] = {
        "model": model,
        "input": prompt,
    }
    if mode in {"paper-web-search", "profile-web-search", "profile-ranking-web-search"}:
        request["tools"] = [{"type": "web_search"}]

    try:
        response = client.responses.create(**request)
    except BadRequestError as error:
        error_body = getattr(error, "body", None)
        error_code = None
        if isinstance(error_body, dict):
            error_code = (error_body.get("error") or {}).get("code")
        if error_code == "model_not_found":
            raise SystemExit(
                f"Model `{model}` is not available to this OpenAI API project. "
                "Use `gpt-5.4-nano`, `gpt-5-nano`, `gpt-4.1-nano`, or another model returned by `client.models.list()`."
            ) from error
        raise
    output_text = getattr(response, "output_text", "")
    if not output_text:
        output_text = json.dumps(response.model_dump(), indent=2)
    return output_text, response


def write_results(output_path: Path, results: Sequence[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def append_result(output_path: Path | None, result: dict[str, object]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with APPEND_LOCK:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_existing_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
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


def completed_row_numbers(rows: Sequence[dict[str, object]]) -> set[int]:
    completed: set[int] = set()
    for row in rows:
        values = row.get("row_numbers") or []
        if not isinstance(values, list):
            continue
        for value in values:
            try:
                completed.add(int(value))
            except (TypeError, ValueError):
                continue
    return completed


def metrics_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}-metrics.txt")


def emit_metrics_report(
    mode: str,
    results: Sequence[dict[str, object]],
    output_path: Path | None,
) -> None:
    if not results:
        return

    report_label = output_path or Path(f"{mode}-current-run")
    if mode in RANKING_MODES:
        metrics = compute_ranking_metrics.compute_metrics(results)
        report = compute_ranking_metrics.format_metrics(report_label, metrics)
    else:
        metrics = compute_classification_metrics.compute_metrics(results)
        report = compute_classification_metrics.format_metrics(report_label, metrics)

    print()
    print(report)

    if output_path:
        metrics_path = metrics_output_path(output_path)
        metrics_path.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")


def summarize_results(results: Sequence[dict[str, object]]) -> str:
    if not results:
        return "No requests were run."

    if results[0]["mode"] in RANKING_MODES:
        top1_matches = sum(1 for result in results if result["evaluation"]["top1_match"])
        ranked_contains = sum(
            1
            for result in results
            if result["evaluation"]["ranked_list_contains_ground_truth"]
        )
        local_top1_matches = sum(
            1 for result in results if result["evaluation"]["local_model_top1_match"]
        )
        local_top5_contains = sum(
            1
            for result in results
            if result["evaluation"]["local_model_top5_contains_ground_truth"]
        )
        return (
            f"Ran {len(results)} request(s). "
            f"LLM top-1 accuracy: {top1_matches}/{len(results)}. "
            f"LLM ranked-list contains GT: {ranked_contains}/{len(results)}. "
            f"Local top-1 accuracy: {local_top1_matches}/{len(results)}. "
            f"Local top-5 contains GT: {local_top5_contains}/{len(results)}."
        )

    strict_matches = sum(1 for result in results if result["evaluation"]["strict_match"])
    label_matches = sum(1 for result in results if result["evaluation"]["label_match"])
    return (
        f"Ran {len(results)} request(s). "
        f"Strict matches: {strict_matches}/{len(results)}. "
        f"Label matches: {label_matches}/{len(results)}."
    )


def run_parallel_jobs(
    jobs: Sequence[tuple[int, str, object]],
    worker,
    parallelism: int,
    output_path: Path | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object] | None] = [None] * len(jobs)
    if parallelism <= 1:
        for index, label, payload in jobs:
            result = worker(index, label, payload)
            results[index - 1] = result
            append_result(output_path, result)
        return [result for result in results if result is not None]

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_job = {
            executor.submit(worker, index, label, payload): (index, label)
            for index, label, payload in jobs
        }
        for future in as_completed(future_to_job):
            index, label = future_to_job[future]
            try:
                result = future.result()
                results[index - 1] = result
                append_result(output_path, result)
            except Exception as error:
                raise RuntimeError(f"Request {index} ({label}) failed.") from error

    return [result for result in results if result is not None]


def run_profile_ranking_request(
    args: argparse.Namespace,
    total_requests: int,
    index: int,
    record: ProfileRankingRecord,
) -> dict[str, object]:
    profile_template = (
        str(args.profile_template or default_profile_template_path(args.mode))
        if args.mode in {"profile-ranking-v2", "profile-reranking-v2"}
        else None
    )
    prompt = build_profile_ranking_prompt(
        args.mode,
        record,
        max_papers_per_profile=args.max_papers_per_profile,
        template_path=Path(args.profile_template) if args.profile_template else None,
    )
    response_text, response = call_openai(args.mode, args.model, prompt)
    parsed = parse_profile_ranking_response(response_text)
    evaluation = evaluate_profile_ranking_result(record, parsed)
    result = {
        "mode": args.mode,
        "model": args.model,
        "collect_evidence": args.collect_evidence,
        "profile_template": profile_template,
        "advisee_id": record.advisee_id,
        "advisee_name": record.advisee_name,
        "pmids": [paper.pmid for paper in record.papers[: args.max_papers_per_profile]],
        "row_numbers": [record.row_number],
        "model_candidates": record.model_candidates,
        "model_probabilities": record.model_probabilities,
        "prompt": prompt,
        "response_text": response_text,
        "parsed": parsed,
        "evaluation": evaluation,
        "response_id": getattr(response, "id", None),
    }
    print(
        f"[{index}/{total_requests}] "
        f"{record.advisee_name}: top1={evaluation['top1_advisor']}"
    )
    return result


def run_classification_request(
    args: argparse.Namespace,
    total_requests: int,
    index: int,
    advisee_name: str,
    papers: list[PaperRecord],
) -> dict[str, object]:
    classification_template = (
        str(args.classification_template or DEFAULT_CLASSIFICATION_TEMPLATE_V2)
        if args.mode == "paper-known-mentee-v2"
        else None
    )
    prompt = build_prompt(
        args.mode,
        papers,
        profile_owner=advisee_name,
        collect_evidence=args.collect_evidence,
        template_path=Path(args.classification_template) if args.classification_template else None,
    )
    response_text, response = call_openai(args.mode, args.model, prompt)
    parsed = parse_response(args.mode, response_text)
    evaluation = evaluate_result(args.mode, papers, parsed)
    result = {
        "mode": args.mode,
        "model": args.model,
        "collect_evidence": args.collect_evidence,
        "classification_template": classification_template,
        "advisee_name": advisee_name,
        "pmids": [paper.pmid for paper in papers],
        "row_numbers": [paper.row_number for paper in papers],
        "prompt": prompt,
        "response_text": response_text,
        "parsed": parsed,
        "evaluation": evaluation,
        "response_id": getattr(response, "id", None),
    }
    print(f"[{index}/{total_requests}] {advisee_name}: {parsed.get('mentorship_detected')}")
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mentorship prompts against the OpenAI Responses API."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(PAPER_MODES | PROFILE_MODES),
        help="Prompt mode derived from BrainstormingDoc.md.",
    )
    parser.add_argument(
        "--csv",
        default="data/raw/testing_set.csv",
        help="Path to the CSV/TSV dataset. Defaults to data/raw/testing_set.csv.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-5-mini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="How many paper rows or profiles to run. Defaults to 1. Use 0 to run all matched rows.",
    )
    parser.add_argument(
        "--row-number",
        type=int,
        action="append",
        default=[],
        help="Select a specific 1-based data row. May be repeated.",
    )
    parser.add_argument(
        "--pmid",
        action="append",
        default=[],
        help="Select a specific PMID. May be repeated.",
    )
    parser.add_argument(
        "--advisee",
        action="append",
        default=[],
        help="Select a specific advisee_name. May be repeated.",
    )
    parser.add_argument(
        "--max-papers-per-profile",
        type=int,
        default=10,
        help="Maximum number of papers to include in profile modes. Defaults to 10 earliest papers.",
    )
    parser.add_argument(
        "--profile-template",
        help=(
            "Optional prompt template path for profile-ranking-v2/profile-reranking-v2. "
            "Defaults to the matching template in prompts/."
        ),
    )
    parser.add_argument(
        "--classification-template",
        help=(
            "Optional prompt template path for paper-known-mentee-v2. "
            "Defaults to prompts/classification_prompt_template_with_prof_v2.txt."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional JSONL output path for raw results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --output JSONL by skipping completed row_numbers.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of API requests to run concurrently. Defaults to 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API.",
    )
    parser.add_argument(
        "--collect-evidence",
        action="store_true",
        help="Request confidence, evidence tags, and a short justification in the model output.",
    )
    return parser.parse_args(argv)


def iter_requests(args: argparse.Namespace, records: Sequence[PaperRecord]) -> Iterable[tuple[str, list[PaperRecord]]]:
    if args.mode in PAPER_MODES:
        selected = select_paper_records(
            records=records,
            row_numbers=args.row_number,
            pmids=args.pmid,
            advisees=args.advisee,
            limit=args.limit,
        )
        for paper in selected:
            yield paper.advisee_name, [paper]
        return

    profiles = select_profiles(
        records=records,
        row_numbers=args.row_number,
        pmids=args.pmid,
        advisees=args.advisee,
        limit=args.limit,
        max_papers_per_profile=args.max_papers_per_profile,
    )
    for advisee_name, papers in profiles:
        yield advisee_name, papers


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be 0 or greater.")
    if args.limit == 0:
        args.limit = None
    if args.parallelism < 1:
        raise SystemExit("--parallelism must be at least 1.")
    output_path = Path(args.output) if args.output else None
    existing_results: list[dict[str, object]] = []
    completed_rows: set[int] = set()
    if args.resume:
        if output_path is None:
            raise SystemExit("--resume requires --output.")
        existing_results = load_existing_results(output_path)
        completed_rows = completed_row_numbers(existing_results)
        if existing_results:
            print(f"Loaded {len(existing_results)} existing result(s) from {output_path}.")

    if args.mode in RANKING_MODES:
        ranking_records = load_profile_ranking_dataset(Path(args.csv))
        ranking_requests = select_profile_ranking_records(
            records=ranking_records,
            row_numbers=args.row_number,
            advisees=args.advisee,
            limit=args.limit,
        )
        if completed_rows:
            before_count = len(ranking_requests)
            ranking_requests = [
                record for record in ranking_requests if record.row_number not in completed_rows
            ]
            print(f"Skipping {before_count - len(ranking_requests)} completed profile row(s).")
        if not ranking_requests:
            if existing_results:
                print("No remaining profile rows to run.")
                print(summarize_results(existing_results))
                emit_metrics_report(args.mode, existing_results, output_path)
                return 0
            raise SystemExit("No dataset rows matched the provided filters.")

        if args.dry_run:
            for index, record in enumerate(ranking_requests, start=1):
                prompt = build_profile_ranking_prompt(
                    args.mode,
                    record,
                    max_papers_per_profile=args.max_papers_per_profile,
                    template_path=Path(args.profile_template) if args.profile_template else None,
                )
                print(f"=== Request {index}: {record.advisee_name} ===")
                print(prompt)
                print()
            return 0

        if args.parallelism > 1:
            print(f"Running {len(ranking_requests)} requests with parallelism={args.parallelism}.")

        total_requests = len(ranking_requests)
        jobs = [
            (index, record.advisee_name, record)
            for index, record in enumerate(ranking_requests, start=1)
        ]

        if output_path and not args.resume and output_path.exists():
            raise SystemExit(f"{output_path} already exists. Use --resume or choose a new --output path.")

        new_results = run_parallel_jobs(
            jobs,
            worker=lambda index, _label, record: run_profile_ranking_request(
                args,
                total_requests,
                index,
                record,
            ),
            parallelism=args.parallelism,
            output_path=output_path,
        )

        results = existing_results + new_results
        if output_path:
            print(f"Wrote results to {output_path}")

        print(summarize_results(results))
        emit_metrics_report(args.mode, results, output_path)
        return 0

    records = load_dataset(Path(args.csv))
    requests = list(iter_requests(args, records))
    if completed_rows:
        before_count = len(requests)
        requests = [
            (advisee_name, papers)
            for advisee_name, papers in requests
            if not any(paper.row_number in completed_rows for paper in papers)
        ]
        print(f"Skipping {before_count - len(requests)} completed paper/profile request(s).")

    if not requests:
        if existing_results:
            print("No remaining requests to run.")
            print(summarize_results(existing_results))
            emit_metrics_report(args.mode, existing_results, output_path)
            return 0
        raise SystemExit("No dataset rows matched the provided filters.")

    if args.dry_run:
        for index, (advisee_name, papers) in enumerate(requests, start=1):
            prompt = build_prompt(
                args.mode,
                papers,
                profile_owner=advisee_name,
                collect_evidence=args.collect_evidence,
                template_path=Path(args.classification_template) if args.classification_template else None,
            )
            print(f"=== Request {index}: {advisee_name} ===")
            print(prompt)
            print()
        return 0

    if args.parallelism > 1:
        print(f"Running {len(requests)} requests with parallelism={args.parallelism}.")

    total_requests = len(requests)
    jobs = [
        (index, advisee_name, (advisee_name, papers))
        for index, (advisee_name, papers) in enumerate(requests, start=1)
    ]
    if output_path and not args.resume and output_path.exists():
        raise SystemExit(f"{output_path} already exists. Use --resume or choose a new --output path.")

    new_results = run_parallel_jobs(
        jobs,
        worker=lambda index, _label, payload: run_classification_request(
            args,
            total_requests,
            index,
            payload[0],
            payload[1],
        ),
        parallelism=args.parallelism,
        output_path=output_path,
    )

    results = existing_results + new_results
    if output_path:
        print(f"Wrote results to {output_path}")

    print(summarize_results(results))
    emit_metrics_report(args.mode, results, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
