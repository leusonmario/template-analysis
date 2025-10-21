#!/usr/bin/env python3
"""
Plot LLM category distribution across programming languages.

Usage:
    python plot_category_distribution.py --inputs llm_classified_random_Java.csv llm_classified_random_Python.csv

Requirements:
    pip install pandas matplotlib
"""

import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict


def extract_language_from_filename(filename):
    """Infer programming language from filename (last word before .csv)."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    return parts[-1]  # assumes last underscore-separated token is language


def parse_category(json_text):
    """Extract 'category' field from clean JSON inside triple backticks."""
    if pd.isna(json_text):
        return None
    text = str(json_text).strip()

    # Remove markdown formatting if present
    if text.startswith("```"):
        text = text.strip("`")
        # remove optional 'json' hint
        text = text.replace("json", "", 1).strip()
    if text.startswith("{") and text.endswith("}"):
        pass
    else:
        # Try to find JSON block inside
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        data = json.loads(text)
        return data.get("category", "").strip()
    except Exception:
        print(json_text)
        return None


def main(files):
    all_counts = defaultdict(Counter)
    all_categories = set()

    for file in files:
        lang = extract_language_from_filename(file)
        print(f"\n📂 Processing {file} (Language: {lang})")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
            continue

        if "llm_output" not in df.columns:
            print(f"⚠️ No 'llm_output' column in {file}")
            continue

        categories = df["llm_output"].apply(parse_category).dropna()

        if categories.empty:
            print("⚠️ No valid categories found. Example data:")
            print(df["llm_output"].head(3).to_string(index=False))
            continue

        counts = Counter(categories)
        all_counts[lang].update(counts)
        all_categories.update(counts.keys())
        print(f"✅ Parsed {len(categories)} valid categories for {lang}.")

    if not all_counts:
        print("❌ No valid data found across files.")
        return

    # Prepare data for plotting
    categories_sorted = sorted(all_categories)
    langs = sorted(all_counts.keys())

    data = []
    for lang in langs:
        data.append([all_counts[lang].get(cat, 0) for cat in categories_sorted])

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / len(langs)
    x = range(len(categories_sorted))

    for i, lang in enumerate(langs):
        offset = [pos + i * bar_width for pos in x]
        ax.bar(offset, data[i], width=bar_width, label=lang)

    ax.set_xticks([pos + bar_width * (len(langs) - 1) / 2 for pos in x])
    ax.set_xticklabels(categories_sorted, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of LLM-classified Categories by Programming Language")
    ax.legend(title="Language")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLM category distributions across programming languages.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to CSV files (one per language).")
    args = parser.parse_args()
    main(args.inputs)
