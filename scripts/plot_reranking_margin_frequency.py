#!/usr/bin/env python3
"""Plot local probability-margin frequencies by reranking correctness."""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DISAGREEMENT = Path("analysis/disagreement/reranking_v2_pub_ordered_profile_disagreements_full_table.csv")
DEFAULT_PROFILE_TSV = Path("data/processed/llm_testing_set_advisee_complete_profile_updated_pub_ordered.tsv")
DEFAULT_OUTPUT_DIR = Path("analysis/disagreement")
DEFAULT_FIGURES_DIR = Path("JCDL26-LLM-for-AA/Figures")


def split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def load_probability_gaps(profile_tsv: Path) -> dict[str, float]:
    gaps: dict[str, float] = {}
    with profile_tsv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            advisee_id = (row.get("AUID") or row.get("Advisee_id") or "").strip()
            probabilities = [as_float(value) for value in split_csv_list(row.get("Probability") or "")]
            probabilities = [value for value in probabilities if value is not None]
            if len(probabilities) >= 2:
                gaps[advisee_id] = probabilities[0] - probabilities[1]
    return gaps


def gap_bin(gap: float) -> str:
    if gap < 0.05:
        return "<0.05"
    if gap < 0.10:
        return "0.05-0.10"
    if gap < 0.15:
        return "0.10-0.15"
    if gap < 0.20:
        return "0.15-0.20"
    if gap < 0.30:
        return "0.20-0.30"
    if gap < 0.40:
        return "0.30-0.40"
    if gap < 0.50:
        return "0.40-0.50"
    return ">=0.50"


def load_counts(disagreement_csv: Path, gaps: dict[str, float]) -> tuple[list[dict[str, object]], list[str]]:
    bins = ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", ">=0.50"]
    counts: dict[str, Counter[str]] = {bin_name: Counter() for bin_name in bins}
    with disagreement_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gap = gaps.get(row["advisee_id"])
            if gap is None:
                continue
            bin_name = gap_bin(gap)
            counts[bin_name]["local_correct" if row["local_top1_correct"] == "YES" else "local_wrong"] += 1
            counts[bin_name]["mini_correct" if row["mini_top1_correct"] == "YES" else "mini_wrong"] += 1
            counts[bin_name]["nano_correct" if row["nano_top1_correct"] == "YES" else "nano_wrong"] += 1
            if row["local_top1_correct"] != "YES" and row["mini_top1_correct"] == "YES":
                counts[bin_name]["mini_rescue"] += 1
            if row["local_top1_correct"] != "YES" and row["nano_top1_correct"] == "YES":
                counts[bin_name]["nano_rescue"] += 1
            counts[bin_name]["total"] += 1

    rows: list[dict[str, object]] = []
    for bin_name in bins:
        total = counts[bin_name]["total"]
        row: dict[str, object] = {"gap_bin": bin_name, "total": total}
        for model in ["local", "mini", "nano"]:
            correct = counts[bin_name][f"{model}_correct"]
            wrong = counts[bin_name][f"{model}_wrong"]
            row[f"{model}_correct"] = correct
            row[f"{model}_wrong"] = wrong
            row[f"{model}_accuracy"] = correct / total if total else 0.0
        local_wrong = counts[bin_name]["local_wrong"]
        row["mini_rescue"] = counts[bin_name]["mini_rescue"]
        row["nano_rescue"] = counts[bin_name]["nano_rescue"]
        row["mini_rescue_rate"] = counts[bin_name]["mini_rescue"] / local_wrong if local_wrong else 0.0
        row["nano_rescue_rate"] = counts[bin_name]["nano_rescue"] / local_wrong if local_wrong else 0.0
        rows.append(row)
    return rows, bins


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str = "#111") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str = "#111") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]), xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def plot_counts(path: Path, rows: Sequence[dict[str, object]], *, include_rescue: bool = False) -> None:
    width, height = 1500, 900
    margin_left, margin_right, margin_top, margin_bottom = 130, 280, 95, 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(34, True)
    label_font = load_font(24)
    tick_font = load_font(20)
    small_font = load_font(19)
    series = [
        ("local_correct", "local correct", "#111111", "circle"),
        ("local_wrong", "local wrong", "#9A9A9A", "square"),
        ("mini_correct", "mini correct", "#2F6B9A", "circle"),
        ("mini_wrong", "mini wrong", "#8FB8D8", "square"),
        ("nano_correct", "nano correct", "#C75D2C", "circle"),
        ("nano_wrong", "nano wrong", "#E5A27C", "square"),
    ]
    if include_rescue:
        series.append(("mini_rescue", "mini correct, local wrong", "#008C8C", "diamond"))

    title = "Mini correction cases by local probability margin" if include_rescue else "Correct and wrong predictions by local probability margin"
    draw_centered(draw, (width / 2, 42), title, title_font)
    max_value = max(int(row[key]) for row in rows for key, _label, _color, _marker in series)
    y_max = ((max_value + 49) // 50) * 50

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - value / y_max)

    def x(index: int) -> float:
        if len(rows) == 1:
            return margin_left + plot_w / 2
        return margin_left + index * (plot_w / (len(rows) - 1))

    for tick in range(0, y_max + 1, 50):
        yy = y(tick)
        draw.line((margin_left, yy, width - margin_right, yy), fill="#dddddd", width=2)
        draw_right(draw, (margin_left - 12, yy), str(tick), tick_font)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#333333", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#333333", width=2)

    for index, row in enumerate(rows):
        xx = x(index)
        draw.line((xx, margin_top, xx, height - margin_bottom), fill="#eeeeee", width=1)
        draw_centered(draw, (xx, height - margin_bottom + 35), str(row["gap_bin"]), tick_font)
        draw_centered(draw, (xx, height - margin_bottom + 64), f"n={row['total']}", small_font, "#444")

    for key, label, color, marker in series:
        points = [(x(index), y(int(row[key]))) for index, row in enumerate(rows)]
        for start, end in zip(points, points[1:]):
            line_width = 7 if key == "mini_rescue" else 6 if key.startswith("local") else 4
            draw.line((*start, *end), fill=color, width=line_width)
        for xx, yy in points:
            if marker == "circle":
                draw.ellipse((xx - 8, yy - 8, xx + 8, yy + 8), fill=color, outline="white", width=2)
            elif marker == "square":
                draw.rectangle((xx - 8, yy - 8, xx + 8, yy + 8), fill=color, outline="white", width=2)
            else:
                draw.polygon([(xx, yy - 10), (xx + 10, yy), (xx, yy + 10), (xx - 10, yy)], fill=color, outline="white")

    draw_centered(draw, (margin_left + plot_w / 2, height - 28), "Local model margin: P(rank 1) - P(rank 2)", label_font)

    y_label = "Number of profiles"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    td = ImageDraw.Draw(text_image)
    td.text((6, 6), y_label, font=label_font, fill="#111")
    rotated = text_image.rotate(90, expand=True)
    image.paste(rotated, (36, int(margin_top + plot_h / 2 - rotated.height / 2)), rotated)

    legend_x = width - margin_right + 35
    legend_y = margin_top + 45
    for i, (key, label, color, marker) in enumerate(series):
        yy = legend_y + i * 34
        line_width = 7 if key == "mini_rescue" else 6 if key.startswith("local") else 4
        draw.line((legend_x, yy, legend_x + 30, yy), fill=color, width=line_width)
        if marker == "circle":
            draw.ellipse((legend_x + 9, yy - 8, legend_x + 25, yy + 8), fill=color, outline="white", width=2)
        elif marker == "square":
            draw.rectangle((legend_x + 9, yy - 8, legend_x + 25, yy + 8), fill=color, outline="white", width=2)
        else:
            draw.polygon(
                [
                    (legend_x + 17, yy - 10),
                    (legend_x + 27, yy),
                    (legend_x + 17, yy + 10),
                    (legend_x + 7, yy),
                ],
                fill=color,
                outline="white",
            )
        draw.text((legend_x + 34, yy - 13), label, font=small_font, fill="#111")
    draw.text((legend_x, legend_y + len(series) * 34 + 12), "Frequency by", font=small_font, fill="#444")
    draw.text((legend_x, legend_y + len(series) * 34 + 40), "margin bin", font=small_font, fill="#444")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300))


def plot_rescue_rates(path: Path, rows: Sequence[dict[str, object]]) -> None:
    width, height = 1500, 900
    margin_left, margin_right, margin_top, margin_bottom = 130, 250, 95, 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(34, True)
    label_font = load_font(24)
    tick_font = load_font(20)
    small_font = load_font(19)
    series = [
        ("mini_rescue_rate", "mini correction rate", "#008C8C", "circle"),
        ("nano_rescue_rate", "nano correction rate", "#C75D2C", "square"),
    ]
    y_max = 0.20

    draw_centered(draw, (width / 2, 42), "LLM correction rate when local model is wrong", title_font)

    def y(value: float) -> float:
        clipped = min(max(value, 0.0), y_max)
        return margin_top + plot_h * (1 - clipped / y_max)

    def x(index: int) -> float:
        if len(rows) == 1:
            return margin_left + plot_w / 2
        return margin_left + index * (plot_w / (len(rows) - 1))

    for tick in [0.0, 0.05, 0.10, 0.15, 0.20]:
        yy = y(tick)
        draw.line((margin_left, yy, width - margin_right, yy), fill="#dddddd", width=2)
        draw_right(draw, (margin_left - 12, yy), f"{tick:.2f}", tick_font)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#333333", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#333333", width=2)

    for index, row in enumerate(rows):
        xx = x(index)
        draw.line((xx, margin_top, xx, height - margin_bottom), fill="#eeeeee", width=1)
        draw_centered(draw, (xx, height - margin_bottom + 35), str(row["gap_bin"]), tick_font)
        draw_centered(draw, (xx, height - margin_bottom + 64), f"wrong={row['local_wrong']}", small_font, "#444")

    for key, label, color, marker in series:
        points = [(x(index), y(float(row[key]))) for index, row in enumerate(rows)]
        for start, end in zip(points, points[1:]):
            draw.line((*start, *end), fill=color, width=5)
        for index, (xx, yy) in enumerate(points):
            if marker == "circle":
                draw.ellipse((xx - 8, yy - 8, xx + 8, yy + 8), fill=color, outline="white", width=2)
            else:
                draw.rectangle((xx - 8, yy - 8, xx + 8, yy + 8), fill=color, outline="white", width=2)
            value = float(rows[index][key])
            if key == "nano_rescue_rate":
                label_y = yy - 24
            elif value > 0.17:
                label_y = yy + 28
            else:
                label_y = yy - 24 if index % 2 == 0 else yy + 24
            draw_centered(draw, (xx, label_y), f"{value:.2f}", small_font, "#222")

    draw_centered(draw, (margin_left + plot_w / 2, height - 28), "Local model margin: P(rank 1) - P(rank 2)", label_font)

    y_label = "Correction rate among local-wrong cases"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    td = ImageDraw.Draw(text_image)
    td.text((6, 6), y_label, font=label_font, fill="#111")
    rotated = text_image.rotate(90, expand=True)
    image.paste(rotated, (36, int(margin_top + plot_h / 2 - rotated.height / 2)), rotated)

    legend_x = width - margin_right - 300
    legend_y = margin_top + 28
    for i, (_key, label, color, marker) in enumerate(series):
        yy = legend_y + i * 38
        draw.line((legend_x, yy, legend_x + 30, yy), fill=color, width=5)
        if marker == "circle":
            draw.ellipse((legend_x + 9, yy - 8, legend_x + 25, yy + 8), fill=color, outline="white", width=2)
        else:
            draw.rectangle((legend_x + 9, yy - 8, legend_x + 25, yy + 8), fill=color, outline="white", width=2)
        draw.text((legend_x + 34, yy - 13), label, font=small_font, fill="#111")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300))


def plot_rescue_rates_svg(path: Path, rows: Sequence[dict[str, object]]) -> None:
    width, height = 1500, 900
    margin_left, margin_right, margin_top, margin_bottom = 130, 250, 95, 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_max = 0.20
    series = [
        ("mini_rescue_rate", "mini correction rate", "#008C8C", "circle"),
        ("nano_rescue_rate", "nano correction rate", "#C75D2C", "square"),
    ]

    def y(value: float) -> float:
        clipped = min(max(value, 0.0), y_max)
        return margin_top + plot_h * (1 - clipped / y_max)

    def x(index: int) -> float:
        if len(rows) == 1:
            return margin_left + plot_w / 2
        return margin_left + index * (plot_w / (len(rows) - 1))

    def text(xy: tuple[float, float], value: str, size: int = 24, anchor: str = "middle", weight: str = "normal", rotate: str = "") -> str:
        transform = f' transform="{rotate}"' if rotate else ""
        return (
            f'<text x="{xy[0]:.1f}" y="{xy[1]:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111"{transform}>'
            f"{html.escape(value)}</text>"
        )

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text((width / 2, 58), "LLM correction rate when local model is wrong", 34, weight="bold"),
    ]

    for tick in [0.0, 0.05, 0.10, 0.15, 0.20]:
        yy = y(tick)
        elements.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="2"/>')
        elements.append(text((margin_left - 12, yy + 7), f"{tick:.2f}", 20, anchor="end"))
    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="2"/>')
    elements.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="2"/>')

    for index, row in enumerate(rows):
        xx = x(index)
        elements.append(f'<line x1="{xx:.1f}" y1="{margin_top}" x2="{xx:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1"/>')
        elements.append(text((xx, height - margin_bottom + 43), str(row["gap_bin"]), 20))
        elements.append(text((xx, height - margin_bottom + 72), f"wrong={row['local_wrong']}", 19))

    for key, _label, color, marker in series:
        points = [(x(index), y(float(row[key]))) for index, row in enumerate(rows)]
        point_attr = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        elements.append(f'<polyline points="{point_attr}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>')
        for index, (xx, yy) in enumerate(points):
            if marker == "circle":
                elements.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="8" fill="{color}" stroke="white" stroke-width="2"/>')
            else:
                elements.append(f'<rect x="{xx - 8:.1f}" y="{yy - 8:.1f}" width="16" height="16" fill="{color}" stroke="white" stroke-width="2"/>')
            value = float(rows[index][key])
            if key == "nano_rescue_rate":
                label_y = yy - 24
            elif value > 0.17:
                label_y = yy + 30
            else:
                label_y = yy - 24 if index % 2 == 0 else yy + 30
            elements.append(text((xx, label_y + 7), f"{value:.2f}", 19))

    elements.append(text((margin_left + plot_w / 2, height - 28), "Local model margin: P(rank 1) - P(rank 2)", 24))
    elements.append(text((30, margin_top + plot_h / 2), "Correction rate among local-wrong cases", 24, rotate=f"rotate(-90 30 {margin_top + plot_h / 2})"))

    legend_x = width - margin_right - 300
    legend_y = margin_top + 28
    for i, (_key, label, color, marker) in enumerate(series):
        yy = legend_y + i * 38
        elements.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 30}" y2="{yy}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        if marker == "circle":
            elements.append(f'<circle cx="{legend_x + 17}" cy="{yy}" r="8" fill="{color}" stroke="white" stroke-width="2"/>')
        else:
            elements.append(f'<rect x="{legend_x + 9}" y="{yy - 8}" width="16" height="16" fill="{color}" stroke="white" stroke-width="2"/>')
        elements.append(text((legend_x + 34, yy + 7), label, 19, anchor="start"))

    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reranking correctness by local model probability margin.")
    parser.add_argument("--disagreement-csv", type=Path, default=DEFAULT_DISAGREEMENT)
    parser.add_argument("--profile-tsv", type=Path, default=DEFAULT_PROFILE_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, _ = load_counts(args.disagreement_csv, load_probability_gaps(args.profile_tsv))
    csv_path = args.output_dir / "reranking_v2_margin_frequency.csv"
    png_path = args.output_dir / "reranking_v2_margin_frequency.png"
    rescue_path = args.output_dir / "reranking_v2_margin_frequency_with_mini_rescue.png"
    rescue_rate_path = args.output_dir / "reranking_v2_mini_correction_rate.png"
    rescue_rate_svg_path = args.output_dir / "reranking_v2_mini_correction_rate.svg"
    figure_path = args.figures_dir / "reranking_v2_margin_frequency.png"
    figure_rescue_path = args.figures_dir / "reranking_v2_margin_frequency_with_mini_rescue.png"
    figure_rescue_rate_path = args.figures_dir / "reranking_v2_mini_correction_rate.png"
    figure_rescue_rate_svg_path = args.figures_dir / "reranking_v2_mini_correction_rate.svg"
    write_csv(csv_path, rows)
    plot_counts(png_path, rows)
    plot_counts(rescue_path, rows, include_rescue=True)
    plot_rescue_rates(rescue_rate_path, rows)
    plot_rescue_rates_svg(rescue_rate_svg_path, rows)
    plot_counts(figure_path, rows)
    plot_counts(figure_rescue_path, rows, include_rescue=True)
    plot_rescue_rates(figure_rescue_rate_path, rows)
    plot_rescue_rates_svg(figure_rescue_rate_svg_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {rescue_path}")
    print(f"Wrote {rescue_rate_path}")
    print(f"Wrote {rescue_rate_svg_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {figure_rescue_path}")
    print(f"Wrote {figure_rescue_rate_path}")
    print(f"Wrote {figure_rescue_rate_svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
