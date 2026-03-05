#!/usr/bin/env python3
"""
FINAL comparator that uses a "mismatched_solved" CSV IN THE CALCULATION.

Inputs:
1) LLM CSV (required): repo_url,description,topics,llm_output,source_file
2) Manual CSV (required): repo_url + one of [Domain, domain, manual_category, category, manual_domain]
3) Solved CSV (optional but recommended): repo_url + "Final Category" (+ optional comment)

How it's used:
- We extract llm_category/confidence from llm_output.
- We merge manual labels.
- If a repo appears in solved CSV with a non-empty "Final Category", that label OVERRIDES manual_category
  for the FINAL evaluation (match/status/kappa/summary).

Outputs (written to out_dir):
- detailed_comparison.csv
- mismatches.csv
- missing_manual.csv
- summary.csv

Also prints:
- Missing manual (FINAL)
- Mismatches (FINAL)
- Fixed-by-solution count
- Cohen's Kappa (FINAL)
"""

import re
import json
import csv
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score


# ---------------------------
# IO + parsing helpers
# ---------------------------

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="warn"
    )


def extract_json_from_llm_output(text: str) -> dict | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    s = str(text).strip()

    # 1) If fenced, extract inside ```...```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    else:
        # 2) Otherwise try first {...} block
        block = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if block:
            s = block.group(1).strip()

    # Fix doubled quotes from CSV export: ""key"" -> "key"
    s = s.replace('""', '"')

    # Trim after last closing brace
    if "}" in s:
        s = s[: s.rfind("}") + 1]

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_label(label: str) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    s = str(label).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("&", "and")
    s = s.replace("\\", "/")
    s = re.sub(r"\s*/\s*", "/", s)
    return s


def pick_manual_col(manual_df: pd.DataFrame) -> str:
    for c in ["Domain", "domain", "manual_category", "category", "manual_domain"]:
        if c in manual_df.columns:
            return c
    raise ValueError(
        "Manual CSV must contain a category column named one of: "
        "Domain, domain, manual_category, category, manual_domain"
    )


def pick_final_category_col(solved_df: pd.DataFrame) -> str:
    for c in ["Final Category", "final_category", "final category", "FinalCategory", "finalCategory"]:
        if c in solved_df.columns:
            return c
    raise ValueError(
        "Solved CSV must contain a 'Final Category' column (or variants like final_category)."
    )


def pick_comment_col(solved_df: pd.DataFrame) -> str | None:
    for c in ["comment", "Comment", "notes", "Notes"]:
        if c in solved_df.columns:
            return c
    return None


def compute_status(df: pd.DataFrame, manual_norm_col: str, llm_norm_col: str, match_col: str) -> pd.Series:
    status = pd.Series(["OK"] * len(df), index=df.index, dtype=str)

    status.loc[df[manual_norm_col] == ""] = "MISSING_MANUAL"
    status.loc[df[llm_norm_col] == ""] = "MISSING_LLM"

    both_exist = (df[manual_norm_col] != "") & (df[llm_norm_col] != "")
    status.loc[both_exist & (~df[match_col])] = "MISMATCH"
    return status


# ---------------------------
# Main
# ---------------------------

def main(
    llm_csv_path: str,
    manual_csv_path: str,
    solved_csv_path: str | None = "mismatched_solved.csv",
    out_dir: str = ".",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Read inputs ---
    llm_df = safe_read_csv(llm_csv_path)
    manual_df = safe_read_csv(manual_csv_path)

    solved_df = None
    if solved_csv_path:
        p = Path(solved_csv_path)
        if p.exists():
            solved_df = safe_read_csv(str(p))
        else:
            print(f"ℹ️ Solved file not found: {solved_csv_path} (continuing without it)")

    # --- Validate required columns ---
    if "repo_url" not in llm_df.columns or "llm_output" not in llm_df.columns:
        raise ValueError("LLM CSV must contain columns: repo_url, llm_output")

    if "repo_url" not in manual_df.columns:
        raise ValueError("Manual CSV must contain column: repo_url")

    manual_col = pick_manual_col(manual_df)

    # --- Extract from llm_output ---
    llm_category = []
    llm_confidence = []
    llm_parse_ok = []

    for v in llm_df["llm_output"].tolist():
        obj = extract_json_from_llm_output(v)
        if obj:
            llm_category.append(obj.get("category", "") or obj.get("Category", "") or "")
            llm_confidence.append(obj.get("confidence", "") or obj.get("Confidence", "") or "")
            llm_parse_ok.append(True)
        else:
            llm_category.append("")
            llm_confidence.append("")
            llm_parse_ok.append(False)

    llm_df["llm_category"] = llm_category
    llm_df["llm_confidence"] = llm_confidence
    llm_df["llm_parse_ok"] = llm_parse_ok

    # --- Merge LLM + manual ---
    merged = llm_df.merge(
        manual_df[["repo_url", manual_col]].rename(columns={manual_col: "manual_category"}),
        on="repo_url",
        how="left"
    )

    # --- Prepare + merge solved file WITHOUT column collisions ---
    if solved_df is not None:
        if "repo_url" not in solved_df.columns:
            raise ValueError("Solved CSV must contain column: repo_url")

        final_col = pick_final_category_col(solved_df)
        comment_col = pick_comment_col(solved_df)

        # Keep only what we need from solved (prevents llm_category_x/_y problems)
        keep_cols = ["repo_url", final_col]
        if comment_col:
            keep_cols.append(comment_col)

        solved_df2 = solved_df[keep_cols].copy()
        solved_df2 = solved_df2.rename(columns={final_col: "Final Category"})
        if comment_col:
            solved_df2 = solved_df2.rename(columns={comment_col: "comment"})
        else:
            solved_df2["comment"] = ""

        # If duplicates for repo_url exist, keep last
        solved_df2 = solved_df2.drop_duplicates(subset=["repo_url"], keep="last")

        merged = merged.merge(solved_df2, on="repo_url", how="left")
    else:
        merged["Final Category"] = ""
        merged["comment"] = ""

    # --- Initial evaluation (manual vs llm) ---
    merged["llm_category_norm"] = merged["llm_category"].map(normalize_label)
    merged["manual_category_norm"] = merged["manual_category"].map(normalize_label)

    merged["match_initial"] = (
        (merged["llm_category_norm"] != "") &
        (merged["manual_category_norm"] != "") &
        (merged["llm_category_norm"] == merged["manual_category_norm"])
    )
    merged["status_initial"] = compute_status(merged, "manual_category_norm", "llm_category_norm", "match_initial")

    # --- FINAL evaluation: override manual with Final Category when present ---
    merged["final_manual_category"] = merged["manual_category"]
    has_final = merged["Final Category"].map(normalize_label) != ""
    merged.loc[has_final, "final_manual_category"] = merged.loc[has_final, "Final Category"]

    merged["final_manual_norm"] = merged["final_manual_category"].map(normalize_label)

    merged["match"] = (
        (merged["llm_category_norm"] != "") &
        (merged["final_manual_norm"] != "") &
        (merged["llm_category_norm"] == merged["final_manual_norm"])
    )
    merged["status"] = compute_status(merged, "final_manual_norm", "llm_category_norm", "match")

    merged["fixed_by_solution"] = (merged["status_initial"] == "MISMATCH") & has_final & (merged["status"] == "OK")

    # --- Print diagnostics (FINAL) ---
    missing_manual_rows = merged[merged["final_manual_norm"] == ""]
    print(f"\n⚠️ Missing manual entries (FINAL): {len(missing_manual_rows)}")
    if len(missing_manual_rows) > 0:
        for _, row in missing_manual_rows.iterrows():
            print(f"- {row['repo_url']} | LLM: {row.get('llm_category','')}")

    mismatch_rows = merged[merged["status"] == "MISMATCH"]
    print(f"\n❌ Mismatches (FINAL): {len(mismatch_rows)}")
    if len(mismatch_rows) > 0:
        for _, row in mismatch_rows.iterrows():
            print(f"- {row['repo_url']} | LLM: {row.get('llm_category','')} | FinalManual: {row.get('final_manual_category','')}")

    print(f"\n🧩 Fixed by solved file (mismatch -> OK): {int(merged['fixed_by_solution'].sum())}")

    # --- Build outputs (your requested columns + useful extras) ---
    # You asked for: repo_url,Final Category,comment,llm_category,manual_category,match,status,llm_confidence,llm_parse_ok,source_file,description,topics
    detailed_cols = [
        "repo_url",
        "Final Category",
        "comment",
        "llm_category",
        "manual_category",
        "final_manual_category",
        "match",
        "status",
        "llm_confidence",
        "llm_parse_ok",
        "source_file",
        "description",
        "topics",
        # extras for debugging:
        "match_initial",
        "status_initial",
        "fixed_by_solution",
    ]
    detailed_cols = [c for c in detailed_cols if c in merged.columns]
    detailed = merged[detailed_cols].copy()

    # --- Save files ---
    detailed_path = out_dir / "detailed_comparison.csv"
    detailed.to_csv(detailed_path, index=False)

    mismatches_path = out_dir / "mismatches.csv"
    detailed[detailed["status"] == "MISMATCH"].to_csv(mismatches_path, index=False)

    missing_manual_path = out_dir / "missing_manual.csv"
    detailed[detailed["status"] == "MISSING_MANUAL"].to_csv(missing_manual_path, index=False)

    summary = pd.DataFrame({
        "metric": [
            "total",
            "ok_final",
            "mismatch_final",
            "missing_manual_final",
            "missing_llm_final",
            "llm_parse_failed",
            "repos_with_final_category",
            "fixed_by_solution",
            "mismatch_initial",
        ],
        "count": [
            len(detailed),
            int((detailed["status"] == "OK").sum()),
            int((detailed["status"] == "MISMATCH").sum()),
            int((detailed["status"] == "MISSING_MANUAL").sum()),
            int((detailed["status"] == "MISSING_LLM").sum()),
            int((merged["llm_parse_ok"] == False).sum()),
            int(has_final.sum()),
            int(merged["fixed_by_solution"].sum()),
            int((merged["status_initial"] == "MISMATCH").sum()),
        ]
    })

    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # --- Kappa (FINAL) ---
    valid = merged[(merged["llm_category_norm"] != "") & (merged["final_manual_norm"] != "")]
    if len(valid) > 0:
        kappa = cohen_kappa_score(valid["final_manual_norm"], valid["llm_category_norm"])
        print(f"\nCohen's Kappa (FINAL): {kappa:.4f}")
    else:
        print("\nCohen's Kappa (FINAL): N/A (no valid rows with both labels)")

    print("\n✅ Done")
    print(f"- {detailed_path}")
    print(f"- {mismatches_path}")
    print(f"- {missing_manual_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    llm_csv = "combined_random_sample.csv"
    manual_csv = "manual_analysis_100.csv"

    # If your file has a space, use: "mismatched solved.csv"
    solved_csv = "mismatches_resolved.csv"

    out_dir = "."
    main(llm_csv, manual_csv, solved_csv, out_dir)