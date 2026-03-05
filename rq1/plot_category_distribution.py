#!/usr/bin/env python3
"""
Plot and save LLM category distribution across programming languages (with larger text and percentages on bars).
Handles minor category variants (e.g., "Automation/ DevOps" vs "Automation/DevOps").
"""

import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict
import re

MIN_PCT_TO_PLOT = 5.0  # percentage threshold

def extract_language_from_filename(filename):
    """Infer programming language from filename (last word before .csv)."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    return parts[-1]


def parse_category(json_text):
    """Extract 'category' field from clean JSON inside triple backticks."""
    if pd.isna(json_text):
        return None
    text = str(json_text).strip()

    # Remove markdown formatting if present
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        data = json.loads(text)
        return data.get("category", "").strip()
    except Exception:
        print(f"⚠️ Could not parse JSON:\n{text[:200]}...")
        print(text)
        return None


def normalize_category(cat):
    """Normalize category names to avoid duplicates due to spacing, slashes, or capitalization."""
    if not cat:
        return None

    cat = cat.strip()

    # Normalize spacing around slashes and punctuation
    cat = re.sub(r"\s*/\s*", "/", cat)
    cat = re.sub(r"\s{2,}", " ", cat)

    # Normalize capitalization (first letter of each word)
    cat = cat.title()

    # Handle known manual equivalences if necessary
    synonyms = {
        "Regular Project": "Non-Specific Domain",
        "Automation/Devops": "Automation/DevOps",
        "Machine Learning/Ai": "Machine Learning/AI",
        "Ml/Ai": "Machine Learning/AI",
    }
    return synonyms.get(cat, cat)


def main(files, output_file):
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
        categories = categories.apply(normalize_category).dropna()

        if categories.empty:
            print("⚠️ No valid categories found.")
            continue

        counts = Counter(categories)
        all_counts[lang].update(counts)
        all_categories.update(counts.keys())
        print(f"✅ Parsed {len(categories)} valid categories for {lang}.")

    if not all_counts:
        print("❌ No valid data found across files.")
        return

    # Prepare data and compute percentages
    categories_sorted = sorted(all_categories)
    langs = sorted(all_counts.keys())

    data_counts = []
    data_percents = []
    for lang in langs:
        total = sum(all_counts[lang].values())
        counts = [all_counts[lang].get(cat, 0) for cat in categories_sorted]
        percents = [c / total * 100 if total > 0 else 0 for c in counts]
        data_counts.append(counts)
        data_percents.append(percents)

    # Save summary CSV
    summary = []
    for i, lang in enumerate(langs):
        for cat, count, pct in zip(categories_sorted, data_counts[i], data_percents[i]):
            summary.append({"Language": lang, "Category": cat, "Count": count, "Percent": pct})
    summary_df = pd.DataFrame(summary)
    csv_output = os.path.splitext(output_file)[0] + "_summary.csv"
    summary_df.to_csv(csv_output, index=False)
    print(f"\n💾 Summary saved to {csv_output}")

    # -------------------------------------------------
    # Remove categories that are < 1% in ALL languages
    # -------------------------------------------------
    max_pct_per_category = []
    for cat_idx in range(len(categories_sorted)):
        max_pct = max(data_percents[lang_idx][cat_idx] for lang_idx in range(len(langs)))
        max_pct_per_category.append(max_pct)

    # Keep only categories that reach the threshold in at least one language
    kept_indices = [
        i for i, max_pct in enumerate(max_pct_per_category)
        if max_pct >= MIN_PCT_TO_PLOT
    ]

    categories_sorted = [categories_sorted[i] for i in kept_indices]
    data_counts = [
        [row[i] for i in kept_indices] for row in data_counts
    ]
    data_percents = [
        [row[i] for i in kept_indices] for row in data_percents
    ]

    print(f"📉 Removed {len(max_pct_per_category) - len(kept_indices)} low-frequency categories (< {MIN_PCT_TO_PLOT}%).")

    # ---- Plot ----
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 12,
        "legend.fontsize": 10
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.8 / len(langs)
    x = range(len(categories_sorted))

    max_height = 0
    for i, lang in enumerate(langs):
        offset = [pos + i * bar_width for pos in x]
        filtered_heights = [
            pct if pct >= MIN_PCT_TO_PLOT else 0
            for pct in data_percents[i]
        ]

        bars = ax.bar(offset, filtered_heights, width=bar_width, label=lang)

        if filtered_heights:
            max_height = max(max_height, max(filtered_heights))

        # Add percentage labels only for visible bars
        for bar, pct in zip(bars, data_percents[i]):
            if pct >= MIN_PCT_TO_PLOT:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max_height * 0.01),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    rotation=90,
                    fontweight="medium"
                )

    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks([pos + bar_width * (len(langs) - 1) / 2 for pos in x])
    ax.set_xticklabels(categories_sorted, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Domains Supported by Programming Language Templates", pad=20)
    # Place legend on the right side, outside the chart
    """ax.legend(
        title="Language",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # (x, y) — slightly outside the right border
        borderaxespad=0,
        frameon=False
    )"""
    ax.legend(
        title="Language",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.3
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room on the right for the legend

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"📊 Figure saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLM category distributions across programming languages.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to CSV files (one per language).")
    parser.add_argument("--output", default="category_distribution.png", help="Output plot filename (default: category_distribution.png)")
    args = parser.parse_args()
    main(args.inputs, args.output)

    # python .\plot_category_distribution.py --inputs "deepseek_judger_C#.csv" "deepseek_judger_Java.csv" "deepseek_judger_JavaScript.csv" "deepseek_judger_TypeScript.csv" "deepseek_judger_Python.csv" --output "category_distribution.png"
