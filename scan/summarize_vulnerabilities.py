#!/usr/bin/env python3
"""
Summarize vulnerability scan results PER PROJECT.

Usage:
    python summarize_vulnerabilities.py --input results/summary.csv --outdir aggregated_results

Requirements:
    pip install pandas matplotlib
"""

import argparse
import pandas as pd
import json
import os
from collections import Counter
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def load_json(path):
    """Safely load JSON file if path exists and is valid."""
    if path is None:
        return None
    path = str(path).strip()
    if not path or path.lower() == "nan" or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_pip_audit(data):
    """Return severity counts from pip-audit JSON output."""
    counts = Counter()
    if not data:
        return counts
    if isinstance(data, list):
        # typical pip-audit output
        for pkg in data:
            for v in pkg.get("vulns", []):
                sev = (v.get("severity") or "UNKNOWN").upper()
                counts[sev] += 1
    elif isinstance(data, dict):
        deps = data.get("dependencies", [])
        for pkg in deps:
            for v in pkg.get("vulns", []):
                sev = (v.get("severity") or "UNKNOWN").upper()
                counts[sev] += 1
    return counts


def parse_bandit(data):
    """Return severity counts from Bandit JSON output."""
    counts = Counter()
    if not data or "results" not in data:
        return counts
    for item in data["results"]:
        sev = (item.get("issue_severity") or "UNKNOWN").upper()
        counts[sev] += 1
    return counts


def parse_semgrep(data):
    """Return severity counts from Semgrep JSON output."""
    counts = Counter()
    if not data or "results" not in data:
        return counts
    for item in data["results"]:
        sev = item.get("extra", {}).get("severity", "UNKNOWN").upper()
        counts[sev] += 1
    return counts


def merge_counts(*args):
    total = Counter()
    for c in args:
        total.update(c)
    return total


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def main(args):
    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    per_repo_summary = []

    for _, row in df.iterrows():
        repo_id = row.get("repo_id", "unknown")
        repo_url = row.get("repo_url", "")
        pip_audit_path = str(row.get("pip_audit", "")).strip()
        bandit_path = str(row.get("bandit", "")).strip()
        semgrep_path = str(row.get("semgrep", "")).strip()

        pip_audit_data = load_json(pip_audit_path)
        bandit_data = load_json(bandit_path)
        semgrep_data = load_json(semgrep_path)

        pip_counts = parse_pip_audit(pip_audit_data)
        bandit_counts = parse_bandit(bandit_data)
        semgrep_counts = parse_semgrep(semgrep_data)

        combined = merge_counts(pip_counts, bandit_counts, semgrep_counts)

        # Normalize severity categories
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        row_summary = {
            "repo_id": repo_id,
            "repo_url": repo_url,
            "pip_audit_total": sum(pip_counts.values()),
            "bandit_total": sum(bandit_counts.values()),
            "semgrep_total": sum(semgrep_counts.values()),
            "total_vulns": sum(combined.values()),
        }
        for sev in severities:
            row_summary[sev.lower()] = combined.get(sev, 0)

        per_repo_summary.append(row_summary)

    # Convert to DataFrame
    out_df = pd.DataFrame(per_repo_summary)
    out_csv = os.path.join(args.outdir, "vulnerability_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Per-project summary saved to: {out_csv}")

    # -----------------------------------------------------------------
    # Aggregate across all projects (optional chart)
    # -----------------------------------------------------------------
    total_counts = Counter()
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        total_counts[sev] = out_df[sev.lower()].sum()

    print("\n=== Overall totals ===")
    for sev, count in total_counts.items():
        print(f"{sev}: {count}")

    # Plot chart
    plt.figure(figsize=(8, 5))
    plt.bar(total_counts.keys(), total_counts.values(), color=["#d62728","#ff7f0e","#ffbb78","#2ca02c","#c7c7c7"])
    plt.title("Overall Vulnerabilities by Severity (All Projects)")
    plt.ylabel("Count")
    for i, (sev, val) in enumerate(total_counts.items()):
        plt.text(i, val + 0.5, str(val), ha="center")
    plt.tight_layout()
    chart_path = os.path.join(args.outdir, "vulnerability_totals.png")
    plt.savefig(chart_path)
    plt.show()
    print(f"📊 Chart saved: {chart_path}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize vulnerabilities per project.")
    parser.add_argument("--input", required=True, help="Path to summary.csv generated by scan_repos_vulns.py")
    parser.add_argument("--outdir", default="aggregated_results", help="Directory to save aggregated outputs")
    args = parser.parse_args()
    main(args)
