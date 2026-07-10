"""
Paper-level baseline evaluation for advisor identification.

Baselines:
1. Random Guess: randomly selects one coauthor after excluding the advisee.
2. Last Author: selects the last listed coauthor after excluding the advisee.

Expected input columns:
- Advisee_name
- authors
- GTruth_Advisor_name
- has_advisor
"""

import random
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Update this path as needed
PAPER_LEVEL_FILE = "llm_test_set_at_paper_level.tsv"


def normalize_name(x):
    """Normalize names for safer comparison."""
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def split_authors(authors):
    """Split comma-separated author string into a clean list."""
    if pd.isna(authors):
        return []
    return [a.strip() for a in str(authors).split(",") if a.strip()]


def get_candidates(authors, advisee):
    """Return author list after excluding the advisee."""
    author_list = split_authors(authors)
    advisee_norm = normalize_name(advisee)

    return [
        a for a in author_list
        if normalize_name(a) != advisee_norm
    ]


def random_guess(authors, advisee):
    """Randomly select one author after excluding the advisee."""
    candidates = get_candidates(authors, advisee)

    if len(candidates) == 0:
        return None

    return random.choice(candidates)


def heuristic_last_author(authors, advisee):
    """Select the last author after excluding the advisee."""
    candidates = get_candidates(authors, advisee)

    if len(candidates) == 0:
        return None

    return candidates[-1]


def paper_level_baseline_metric(df, col_of_interest, mtd):
    """
    Compute paper-level metrics.

    Positive papers are those where has_advisor == True.
    A true positive occurs only when the predicted advisor matches GTruth_Advisor_name.
    Since these baselines always predict an advisor when possible, negative papers become false positives.
    """

    positive = df["has_advisor"].astype(bool)

    pred = df[col_of_interest].apply(normalize_name)
    truth = df["GTruth_Advisor_name"].apply(normalize_name)

    # True Positive
    tp = (positive & (pred == truth)).sum()

    # False Positive: model predicts advisor for papers without advisor
    fp = (df[col_of_interest].fillna("") != df["GTruth_Advisor_name"].fillna("")).sum()

    # False Negative: paper has advisor but prediction is wrong
    fn = (positive & (pred != truth)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Since the baseline always predicts an advisor, TN = 0
    accuracy = tp / len(df) if len(df) else 0.0

    result_df = pd.DataFrame({
        "Approach": [mtd],
        "TP": [tp],
        "FP": [fp],
        "FN": [fn],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
    })

    return result_df


def run_baselines(input_file=PAPER_LEVEL_FILE, output_file="paper_level_baseline_results.tsv"):
    """Run random guess and last-author heuristic baselines."""

    df_paper_level = pd.read_csv(input_file, sep="\t")

    # Random Guess Approach
    df_paper_level["random_guess_advisor"] = df_paper_level.apply(
        lambda row: random_guess(row["authors"], row["Advisee_name"]),
        axis=1,
    )

    random_result = paper_level_baseline_metric(
        df_paper_level,
        "random_guess_advisor",
        "Random Guess",
    )

    # Last Author Heuristic Approach
    df_paper_level["heuristic_advisor"] = df_paper_level.apply(
        lambda row: heuristic_last_author(row["authors"], row["Advisee_name"]),
        axis=1,
    )

    heuristic_result = paper_level_baseline_metric(
        df_paper_level,
        "heuristic_advisor",
        "Last Author",
    )

    results = pd.concat([random_result, heuristic_result], ignore_index=True)

    print(results)
    results.to_csv(output_file, sep="\t", index=False)

    return df_paper_level, results


if __name__ == "__main__":
    run_baselines()
