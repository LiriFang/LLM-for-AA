#!/usr/bin/env python3
"""Create temporal and subject analyses for profile-ranking results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CASE_TABLE = Path("analysis/disagreement/case_study_profile_disagreements_full_table.csv")
DEFAULT_META = Path("data/processed/llm_advisee_testing_set_with_subject.tsv")
DEFAULT_OUTPUT_DIR = Path("analysis/temporal_disciplinary")
DEFAULT_FIGURES_DIR = Path("JCDL26-LLM-for-AA/Figures")


MODELS = [
    ("local", "Local trained model"),
    ("nano", "gpt-5.4-nano"),
    ("mini", "gpt-5.4-mini"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_yes(value: str) -> bool:
    return (value or "").strip().upper() == "YES"


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def merge_rows(case_rows: Sequence[dict[str, str]], meta_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    meta_by_id = {row["AdviseeID"]: row for row in meta_rows}
    merged: list[dict[str, str]] = []
    missing: list[str] = []
    for row in case_rows:
        meta = meta_by_id.get(row["advisee_id"])
        if not meta:
            missing.append(row["advisee_id"])
            continue
        combined = dict(row)
        combined.update(
            {
                "thesis_id": meta.get("ThesisID", ""),
                "year": meta.get("Year", ""),
                "period": meta.get("period", ""),
                "subject": meta.get("subject", ""),
                "discipline_temporal": meta.get("discipline-temporal", ""),
                "advisee_total_pubs": meta.get("AdviseeTotalPubs", ""),
                "total_common_pubs": meta.get("TotalCommonPubs", ""),
            }
        )
        merged.append(combined)
    if missing:
        raise SystemExit(f"Missing metadata for {len(missing)} advisee IDs, e.g. {missing[:5]}")
    return merged


def metric_row(group_name: str, rows: Sequence[dict[str, str]]) -> dict[str, object]:
    n = len(rows)
    out: dict[str, object] = {"group": group_name, "n": n}
    for prefix, _ in MODELS:
        hits1 = sum(is_yes(row.get(f"{prefix}_top1_correct", "")) for row in rows)
        hits5 = sum(is_yes(row.get(f"gt_in_{prefix}_top5", "")) for row in rows)
        mrr = sum(as_float(row.get(f"{prefix}_reciprocal_rank_at5", "")) for row in rows)
        out[f"{prefix}_hits_at1"] = hits1
        out[f"{prefix}_recall_at1"] = hits1 / n if n else 0.0
        out[f"{prefix}_hits_at5"] = hits5
        out[f"{prefix}_recall_at5"] = hits5 / n if n else 0.0
        out[f"{prefix}_mrr_at5"] = mrr / n if n else 0.0
    return out


def group_by(rows: Sequence[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "") or "missing"].append(row)
    return dict(grouped)


def year_bin(year: int, width: int = 5) -> str:
    start = (year // width) * width
    end = start + width - 1
    return f"{start}-{end}"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown_summary(
    path: Path,
    period_rows: Sequence[dict[str, object]],
    subject_rows: Sequence[dict[str, object]],
    annual_rows: Sequence[dict[str, object]],
    year_rows: Sequence[dict[str, object]],
) -> None:
    def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
        return lines

    period_order = {"old": 0, "mid": 1, "recent": 2, "missing": 3}
    sorted_period = sorted(period_rows, key=lambda row: period_order.get(str(row["group"]), 99))
    sorted_subject = sorted(subject_rows, key=lambda row: (-int(row["n"]), str(row["group"])))

    lines = [
        "# Temporal and Subject Profile-Ranking Analysis",
        "",
        "## Period Groups",
        "",
        *table(
            ["Period", "N", "Local R@1", "Nano R@1", "Mini R@1", "Local R@5", "Nano R@5", "Mini R@5"],
            [
                (
                    row["group"],
                    row["n"],
                    row["local_recall_at1"],
                    row["nano_recall_at1"],
                    row["mini_recall_at1"],
                    row["local_recall_at5"],
                    row["nano_recall_at5"],
                    row["mini_recall_at5"],
                )
                for row in sorted_period
            ],
        ),
        "",
        "## Subject Groups",
        "",
        *table(
            ["Subject", "N", "Local R@1", "Nano R@1", "Mini R@1", "Local MRR@5", "Nano MRR@5", "Mini MRR@5"],
            [
                (
                    row["group"],
                    row["n"],
                    row["local_recall_at1"],
                    row["nano_recall_at1"],
                    row["mini_recall_at1"],
                    row["local_mrr_at5"],
                    row["nano_mrr_at5"],
                    row["mini_mrr_at5"],
                )
                for row in sorted_subject
            ],
        ),
        "",
        "## Annual Groups",
        "",
        *table(
            ["Year", "N", "Local R@1", "Nano R@1", "Mini R@1"],
            [
                (
                    row["group"],
                    row["n"],
                    row["local_recall_at1"],
                    row["nano_recall_at1"],
                    row["mini_recall_at1"],
                )
                for row in annual_rows
            ],
        ),
        "",
        "## Year-Binned Groups",
        "",
        *table(
            ["Year Bin", "N", "Local R@1", "Nano R@1", "Mini R@1"],
            [
                (
                    row["group"],
                    row["n"],
                    row["local_recall_at1"],
                    row["nano_recall_at1"],
                    row["mini_recall_at1"],
                )
                for row in year_rows
            ],
        ),
        "",
        "Generated files:",
        "",
        "- `metrics_by_period.csv`",
        "- `metrics_by_subject.csv`",
        "- `metrics_by_year.csv`",
        "- `metrics_by_year_bin.csv`",
        "- `profile_recall_by_period.svg`",
        "- `profile_recall_by_year.svg`",
        "- `profile_recall_by_year_bin.svg`",
        "- `profile_recall_by_subject.svg`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}">{text}</text>'
    )


def svg_rotated_text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    anchor: str = "middle",
    angle: int = -90,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}" '
        f'transform="rotate({angle} {x:.1f},{y:.1f})">{text}</text>'
    )


def create_period_bar_svg(path: Path, rows: Sequence[dict[str, object]]) -> None:
    period_order = {"old": 0, "mid": 1, "recent": 2}
    rows = [row for row in sorted(rows, key=lambda r: period_order.get(str(r["group"]), 99)) if row["group"] in period_order]
    width, height = 880, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 60, 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"local": "#4C78A8", "nano": "#F58518", "mini": "#54A24B"}

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Recall@1 by Dissertation Period", 18, weight="bold"),
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(svg_text(margin_left - 10, yy + 4, f"{tick:.2f}", 11, "end"))
    parts.append(f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{height-margin_bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{height-margin_bottom}" y2="{height-margin_bottom}" stroke="#333"/>')

    group_w = plot_w / len(rows)
    bar_w = group_w / 5
    for i, row in enumerate(rows):
        cx = margin_left + group_w * i + group_w / 2
        for j, (prefix, label) in enumerate(MODELS):
            value = float(row[f"{prefix}_recall_at1"])
            x = cx - bar_w * 1.5 + j * bar_w
            yy = y(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w*.85:.1f}" height="{height-margin_bottom-yy:.1f}" fill="{colors[prefix]}"/>'
            )
            parts.append(svg_text(x + bar_w * .42, yy - 5, f"{value:.2f}", 10))
        parts.append(svg_text(cx, height - margin_bottom + 22, f"{row['group']} (n={row['n']})", 12))

    legend_x = margin_left + 20
    for i, (prefix, label) in enumerate(MODELS):
        x = legend_x + i * 210
        parts.append(f'<rect x="{x}" y="{height-35}" width="14" height="14" fill="{colors[prefix]}"/>')
        parts.append(svg_text(x + 20, height - 23, label, 12, "start"))
    parts.append(svg_text(24, margin_top + plot_h / 2, "Recall@1", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def year_group_start(group: object) -> int:
    return int(str(group).split("-")[0])


def year_group_center(group: object) -> float:
    text = str(group)
    if "-" not in text:
        return float(text)
    start, end = text.split("-", 1)
    return (int(start) + int(end)) / 2


def create_year_line_svg(
    path: Path,
    rows: Sequence[dict[str, object]],
    title: str = "Recall@1 by Dissertation Year Bin",
    x_label: str = "Dissertation year bin",
) -> None:
    rows = list(rows)
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 88, 40, 60, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"local": "#4C78A8", "nano": "#F58518", "mini": "#54A24B"}

    starts = [year_group_start(row["group"]) for row in rows]
    centers = [year_group_center(row["group"]) for row in rows]
    min_year = min(starts)
    max_year = max(max(starts), max(centers))

    def x(year: float) -> float:
        return margin_left + plot_w * ((year - min_year) / (max_year - min_year))

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, 18, weight="bold"),
    ]

    # Era shading and boundaries.
    eras = [("Old", min_year, 1990, "#f7f7f7"), ("Mid", 1991, 2000, "#eef6ff"), ("Recent", 2001, max_year, "#f4fbf1")]
    for label, start, end, fill in eras:
        x1, x2 = max(margin_left, x(start)), min(width - margin_right, x(end))
        if x2 > x1:
            parts.append(f'<rect x="{x1:.1f}" y="{margin_top}" width="{x2-x1:.1f}" height="{plot_h}" fill="{fill}"/>')
            parts.append(svg_text((x1 + x2) / 2, margin_top + 18, label, 11))
    for boundary in [1990, 2001]:
        xx = x(boundary)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{margin_top}" y2="{height-margin_bottom}" stroke="#555" stroke-dasharray="5,5"/>')

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(svg_text(margin_left - 10, yy + 4, f"{tick:.2f}", 11, "end"))
    parts.append(f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{height-margin_bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{height-margin_bottom}" y2="{height-margin_bottom}" stroke="#333"/>')

    for tick in range((min_year // 10) * 10, int(max_year) + 1, 10):
        xx = x(tick)
        if margin_left <= xx <= width - margin_right:
            parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{height-margin_bottom}" y2="{height-margin_bottom+5}" stroke="#333"/>')
            parts.append(svg_text(xx, height - margin_bottom + 20, str(tick), 10))

    for prefix, label in MODELS:
        points = []
        for row in rows:
            points.append((x(year_group_center(row["group"])), y(float(row[f"{prefix}_recall_at1"]))))
        polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colors[prefix]}" stroke-width="2.5"/>')
        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{colors[prefix]}"/>')

    legend_w, legend_h = 205, 78
    legend_x = width - margin_right - legend_w + 10
    legend_y = height - margin_bottom - legend_h + 20
    parts.append(
        f'<rect x="{legend_x-10}" y="{legend_y-20}" width="{legend_w}" height="{legend_h}" '
        'fill="white" fill-opacity="0.88" stroke="#ccc" rx="4"/>'
    )
    for i, (prefix, label) in enumerate(MODELS):
        y0 = legend_y + i * 20
        parts.append(f'<line x1="{legend_x}" x2="{legend_x+18}" y1="{y0}" y2="{y0}" stroke="{colors[prefix]}" stroke-width="3"/>')
        parts.append(svg_text(legend_x + 25, y0 + 4, label, 12, "start"))
    parts.append(svg_text(width / 2, height - 18, x_label, 12))
    parts.append(svg_rotated_text(24, margin_top + plot_h / 2, "Recall@1", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str = "#111") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str = "#111") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((xy[0] - w, xy[1] - h / 2), text, font=font, fill=fill)


def create_year_line_png(
    path: Path,
    rows: Sequence[dict[str, object]],
    title: str = "Recall@1 by Dissertation Year Bin",
    x_label: str = "Dissertation year bin",
) -> None:
    rows = list(rows)
    width, height = 1960, 1120
    scale = 2
    margin_left, margin_right, margin_top, margin_bottom = 176, 80, 120, 160
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"local": "#4C78A8", "nano": "#F58518", "mini": "#54A24B"}

    starts = [year_group_start(row["group"]) for row in rows]
    centers = [year_group_center(row["group"]) for row in rows]
    min_year = min(starts)
    max_year = max(max(starts), max(centers))

    def x(year: float) -> float:
        return margin_left + plot_w * ((year - min_year) / (max_year - min_year))

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - value)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(36, bold=True)
    label_font = load_font(24)
    tick_font = load_font(20)
    small_font = load_font(22)

    draw_centered(draw, (width / 2, 56), title, title_font)

    eras = [("Old", min_year, 1990, "#f7f7f7"), ("Mid", 1991, 2000, "#eef6ff"), ("Recent", 2001, max_year, "#f4fbf1")]
    for label, start, end, fill in eras:
        x1, x2 = max(margin_left, x(start)), min(width - margin_right, x(end))
        if x2 > x1:
            draw.rectangle((x1, margin_top, x2, margin_top + plot_h), fill=fill)
            draw_centered(draw, ((x1 + x2) / 2, margin_top + 34), label, small_font, "#333")
    for boundary in [1990, 2001]:
        xx = x(boundary)
        for yy in range(margin_top, height - margin_bottom, 18):
            draw.line((xx, yy, xx, yy + 10), fill="#555", width=2)

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        draw.line((margin_left, yy, width - margin_right, yy), fill="#dddddd", width=2)
        draw_right(draw, (margin_left - 20, yy), f"{tick:.2f}", tick_font)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#333333", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#333333", width=2)

    for tick in range((min_year // 10) * 10, int(max_year) + 1, 10):
        xx = x(tick)
        if margin_left <= xx <= width - margin_right:
            draw.line((xx, height - margin_bottom, xx, height - margin_bottom + 10), fill="#333333", width=2)
            draw_centered(draw, (xx, height - margin_bottom + 38), str(tick), tick_font)

    for prefix, _ in MODELS:
        points = []
        for row in rows:
            points.append((x(year_group_center(row["group"])), y(float(row[f"{prefix}_recall_at1"]))))
        draw.line(points, fill=colors[prefix], width=5, joint="curve")
        for px, py in points:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=colors[prefix])

    legend_w, legend_h = 410, 152
    legend_x = width - margin_right - legend_w + 40
    legend_y = height - margin_bottom - legend_h + 40
    draw.rounded_rectangle((legend_x - 20, legend_y - 40, legend_x + legend_w - 20, legend_y + legend_h - 40), radius=8, fill=(255, 255, 255, 226), outline="#cccccc")
    for i, (prefix, label) in enumerate(MODELS):
        y0 = legend_y + i * 40
        draw.line((legend_x, y0, legend_x + 36, y0), fill=colors[prefix], width=6)
        draw.text((legend_x + 50, y0 - 13), label, font=small_font, fill="#111")

    draw_centered(draw, (width / 2, height - 36), x_label, label_font)
    label = "Recall@1"
    bbox = draw.textbbox((0, 0), label, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((6, 6), label, font=label_font, fill="#111")
    rotated = text_image.rotate(90, expand=True)
    image.paste(rotated, (30, int(margin_top + plot_h / 2 - rotated.height / 2)), rotated)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_subject_bar_svg(path: Path, rows: Sequence[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: (-int(row["n"]), str(row["group"])))
    width, height = 1080, 620
    margin_left, margin_right, margin_top, margin_bottom = 95, 30, 60, 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"local": "#4C78A8", "nano": "#F58518", "mini": "#54A24B"}

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Recall@1 by Subject", 18, weight="bold"),
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(svg_text(margin_left - 10, yy + 4, f"{tick:.2f}", 11, "end"))
    parts.append(f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{height-margin_bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{margin_left}" x2="{width-margin_right}" y1="{height-margin_bottom}" y2="{height-margin_bottom}" stroke="#333"/>')

    group_w = plot_w / len(rows)
    bar_w = group_w / 5
    for i, row in enumerate(rows):
        cx = margin_left + group_w * i + group_w / 2
        for j, (prefix, _) in enumerate(MODELS):
            value = float(row[f"{prefix}_recall_at1"])
            x0 = cx - bar_w * 1.5 + j * bar_w
            yy = y(value)
            parts.append(
                f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bar_w*.85:.1f}" height="{height-margin_bottom-yy:.1f}" fill="{colors[prefix]}"/>'
            )
        label = str(row["group"]).replace("_", " ")
        parts.append(
            f'<text x="{cx:.1f}" y="{height-margin_bottom+18}" font-size="10" font-family="Arial, sans-serif" '
            f'text-anchor="end" transform="rotate(-35 {cx:.1f},{height-margin_bottom+18})">{label} (n={row["n"]})</text>'
        )

    legend_x = margin_left + 20
    for i, (prefix, label) in enumerate(MODELS):
        x0 = legend_x + i * 210
        parts.append(f'<rect x="{x0}" y="{height-35}" width="14" height="14" fill="{colors[prefix]}"/>')
        parts.append(svg_text(x0 + 20, height - 23, label, 12, "start"))
    parts.append(svg_text(24, margin_top + plot_h / 2, "Recall@1", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze profile-ranking performance by year, period, and subject.")
    parser.add_argument("--case-table", type=Path, default=DEFAULT_CASE_TABLE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--year-bin-width", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = merge_rows(read_csv(args.case_table), read_tsv(args.metadata))

    period_order = {"old": 0, "mid": 1, "recent": 2, "missing": 3}
    period_rows = [
        metric_row(group, grouped)
        for group, grouped in sorted(group_by(rows, "period").items(), key=lambda item: period_order.get(item[0], 99))
    ]

    subject_rows = [
        metric_row(group, grouped)
        for group, grouped in sorted(group_by(rows, "subject").items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    by_year: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        year = as_int(row.get("year", ""))
        if year is None:
            by_year["missing"].append(row)
        else:
            by_year[str(year)].append(row)
    annual_rows = [
        metric_row(group, grouped)
        for group, grouped in sorted(
            by_year.items(),
            key=lambda item: 999999 if item[0] == "missing" else int(item[0]),
        )
    ]

    by_year_bin: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        year = as_int(row.get("year", ""))
        if year is None:
            by_year_bin["missing"].append(row)
        else:
            by_year_bin[year_bin(year, args.year_bin_width)].append(row)
    year_rows = [
        metric_row(group, grouped)
        for group, grouped in sorted(
            by_year_bin.items(),
            key=lambda item: 999999 if item[0] == "missing" else int(item[0].split("-")[0]),
        )
    ]

    write_csv(args.output_dir / "merged_profile_temporal_subject_table.csv", rows)
    write_csv(args.output_dir / "metrics_by_period.csv", period_rows)
    write_csv(args.output_dir / "metrics_by_subject.csv", subject_rows)
    write_csv(args.output_dir / "metrics_by_year.csv", annual_rows)
    write_csv(args.output_dir / "metrics_by_year_bin.csv", year_rows)
    write_markdown_summary(args.output_dir / "temporal_subject_summary.md", period_rows, subject_rows, annual_rows, year_rows)
    create_period_bar_svg(args.output_dir / "profile_recall_by_period.svg", period_rows)
    create_year_line_svg(
        args.output_dir / "profile_recall_by_year.svg",
        [row for row in annual_rows if row["group"] != "missing" and int(row["n"]) >= 10],
        title="Recall@1 by Dissertation Year",
        x_label="Dissertation year",
    )
    create_year_line_png(
        args.output_dir / "profile_recall_by_year.png",
        [row for row in annual_rows if row["group"] != "missing" and int(row["n"]) >= 10],
        title="Recall@1 by Dissertation Year",
        x_label="Dissertation year",
    )
    create_year_line_png(
        args.figures_dir / "profile_recall_by_year.png",
        [row for row in annual_rows if row["group"] != "missing" and int(row["n"]) >= 10],
        title="Recall@1 by Dissertation Year",
        x_label="Dissertation year",
    )
    create_year_line_svg(
        args.output_dir / "profile_recall_by_year_bin.svg",
        [row for row in year_rows if row["group"] != "missing" and int(row["n"]) >= 10],
    )
    create_year_line_png(
        args.output_dir / "profile_recall_by_year_bin.png",
        [row for row in year_rows if row["group"] != "missing" and int(row["n"]) >= 10],
    )
    create_year_line_png(
        args.figures_dir / "profile_recall_by_year_bin.png",
        [row for row in year_rows if row["group"] != "missing" and int(row["n"]) >= 10],
    )
    create_subject_bar_svg(args.output_dir / "profile_recall_by_subject.svg", subject_rows)

    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
