#!/usr/bin/env python3
"""
Analyze repository ownership (Organization vs User) across programming languages (normalized).

Usage:
    python owner_distribution_multi_single.py --inputs owners_Java.csv owners_Python.csv owners_TypeScript.csv --output owner_distribution_normalized.png

Requirements:
    pip install pandas matplotlib
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def extract_language(filename: str):
    """Extract language name from filename (based on suffix before .csv)."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    # Take the last token after underscore or dash
    return parts[-1].capitalize() if len(parts) > 1 else name.capitalize()


def process_file(owner_path: str):
    """Count organization vs user ownership from a single CSV file."""
    df = pd.read_csv(owner_path)

    if not {"original_link_repository", "owner_type"}.issubset(df.columns):
        raise ValueError(f"{owner_path} must contain 'original_link_repository' and 'owner_type' columns.")

    # Count occurrences of Organization vs User
    counts = df["owner_type"].value_counts().reindex(["Organization", "User"], fill_value=0)
    return counts


def main(file_list, output_path):
    results = {}

    for path in file_list:
        lang = extract_language(path)
        counts = process_file(path)
        results[lang] = counts

        total = counts.sum()
        org_pct = (counts.get("Organization", 0) / total * 100) if total > 0 else 0
        user_pct = (counts.get("User", 0) / total * 100) if total > 0 else 0

        print(f"=== {lang} ===")
        print(f"Organization: {org_pct:.1f}% | User: {user_pct:.1f}% (Total: {int(total)})\n")

    # Build DataFrame (languages as rows)
    df = pd.DataFrame(results).fillna(0).T
    df = df[["Organization", "User"]]  # consistent order

    # Normalize to percentage
    df_normalized = df.div(df.sum(axis=1), axis=0) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    df_normalized.plot(
        kind="bar",
        edgecolor="black",
        color=["#1f77b4", "#ff7f0e"],
        ax=ax
    )

    plt.title("Repository Ownership by Programming Language (Percentage)")
    plt.xlabel("Programming Language")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Owner Type", loc="upper right")

    # Add percentage labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Chart saved as: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze normalized ownership distribution across multiple languages (single files).")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of CSV files, one per language (each must have 'owner_type')."
    )
    parser.add_argument(
        "--output",
        required=False,
        default="owner_distribution_normalized.png",
        help="Output PNG filename (default: owner_distribution_normalized.png)"
    )
    args = parser.parse_args()

    main(args.inputs, args.output)
