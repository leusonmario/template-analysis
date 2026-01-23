import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests



# Optional: add quick effect-size bins for Spearman (useful in tables)
def _bin_effect(v):
    if pd.isna(v):
        return np.nan
    a = abs(v)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    return "large"

languages = ["Java", "Python", "TypeScript", "JavaScript", "C#"]

for name in languages:
    #############################################
    # 1. Load datasets
    #############################################

    print("📌 Loading datasets...")

    os.mkdir(name)

    df_meta = pd.read_csv("../all_templates_update_metrics_" + name + ".csv")
    # df_sonar = pd.read_csv("../vulnerability_results_Python_all.csv")
    df_sonar = pd.read_csv("sonar_maintenance_metrics_" + name + "_complete.csv")

    if "repo_url" not in df_sonar.columns:
        raise ValueError("❌ ERROR: 'repo_url' column missing in Sonar results file.")

    #############################################
    # 2. Clean + Transform Metadata
    #############################################

    print("🔧 Cleaning and transforming metadata...")

    # --- Convert dates ---
    df_meta["created_at"] = pd.to_datetime(df_meta["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df_meta["updated_at"] = pd.to_datetime(df_meta["updated_at"], errors="coerce", utc=True).dt.tz_localize(None)

    today = pd.Timestamp.now().tz_localize(None)

    # --- Compute durations ---
    df_meta["repo_age_days"] = (today - df_meta["created_at"]).dt.days
    # df_meta["staleness_days"] = (today - df_meta["updated_at"]).dt.days

    # --- Log transforms ---
    df_meta["log_stars"] = np.log1p(df_meta["stars"].fillna(0))
    df_meta["log_forks"] = np.log1p(df_meta["forks"].fillna(0))

    # --- Owner type ---
    df_meta["owner_is_org"] = df_meta["owner_type"].apply(
        lambda x: 1 if str(x).lower() == "organization" else 0
    )

    #############################################
    # 3. Merge Metadata + Sonar
    #############################################

    print("🔀 Merging metadata and Sonar results...")

    df = df_meta.merge(
        df_sonar,
        left_on="original_link_repository",
        right_on="repo_url",
        how="inner"
    )

    print(f"👉 Merged dataset size: {len(df)} rows")

    #############################################
    # 4. Correlation Analysis (effect + p-values + multiple-testing correction)
    #############################################

    print("\n📊 CORRELATION ANALYSIS (effect sizes + p-values + corrections)\n")

    metrics = [
        "bugs_per_kloc", "code_smells_per_kloc",
        "vulnerabilities_per_kloc", "security_hotspots_per_kloc",
        "vuln_blocker_per_kloc", "vuln_critical_per_kloc",
        "vuln_major_per_kloc", "vuln_minor_per_kloc", "vuln_info_per_kloc"
    ]

    predictors = [
        "staleness_days",
        "commits_last_12m",
        "repo_age_days",
        "log_stars",
        "log_forks",
        "owner_is_org"
    ]

    corr_rows = []

    #############################################
    # Size-normalized metrics
    #############################################

    EPS = 1e-6  # avoid division by zero

    df["kloc"] = df["ncloc"] / 1000.0

    for m in [
        "bugs", "code_smells", "vulnerabilities", "security_hotspots",
        "vuln_blocker", "vuln_critical", "vuln_major", "vuln_minor", "vuln_info"
    ]:
        df[f"{m}_per_kloc"] = df[m] / (df["kloc"] + EPS)
        df[f"{m}_per_file"] = df[m] / (df["files"] + EPS)
        df[f"{m}_per_function"] = df[m] / (df["functions"] + EPS)

    for metric in metrics:
        print(f"🔍 Correlation with {metric}")

        for pred in predictors:
            # Pairwise complete observations + numeric coercion
            tmp = df[[pred, metric]].copy()
            tmp[pred] = pd.to_numeric(tmp[pred], errors="coerce")
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp = tmp.dropna()

            n = len(tmp)
            if n < 3:
                print(f"  {pred}: skipped (n={n})")
                corr_rows.append([metric, pred, n, np.nan, np.nan, np.nan, np.nan])
                continue

            x = tmp[pred].to_numpy()
            y = tmp[metric].to_numpy()

            r, p_r = pearsonr(x, y)
            rho, p_rho = spearmanr(x, y)

            print(
                f"  {pred}: Pearson r={r:.3f} (p={p_r:.2e}), "
                f"Spearman ρ={rho:.3f} (p={p_rho:.2e}), n={n}"
            )

            corr_rows.append([metric, pred, n, r, p_r, rho, p_rho])

        print()

    corr_df = pd.DataFrame(
        corr_rows,
        columns=[
            "metric", "predictor", "n",
            "pearson_r", "pearson_p",
            "spearman_rho", "spearman_p",
        ],
    )

    # ----------------------------
    # Multiple-testing correction
    # ----------------------------
    alpha = 0.05

    mask_p = corr_df["pearson_p"].notna()
    mask_s = corr_df["spearman_p"].notna()

    # Bonferroni (conservative)
    corr_df.loc[mask_p, "pearson_p_bonf"] = multipletests(
        corr_df.loc[mask_p, "pearson_p"].values, method="bonferroni"
    )[1]
    corr_df.loc[mask_s, "spearman_p_bonf"] = multipletests(
        corr_df.loc[mask_s, "spearman_p"].values, method="bonferroni"
    )[1]

    # Benjamini–Hochberg FDR (often preferable for many tests)
    corr_df.loc[mask_p, "pearson_p_fdr"] = multipletests(
        corr_df.loc[mask_p, "pearson_p"].values, method="fdr_bh"
    )[1]
    corr_df.loc[mask_s, "spearman_p_fdr"] = multipletests(
        corr_df.loc[mask_s, "spearman_p"].values, method="fdr_bh"
    )[1]

    # Significance flags
    corr_df["pearson_sig_bonf"] = corr_df["pearson_p_bonf"] < alpha
    corr_df["spearman_sig_bonf"] = corr_df["spearman_p_bonf"] < alpha
    corr_df["pearson_sig_fdr"] = corr_df["pearson_p_fdr"] < alpha
    corr_df["spearman_sig_fdr"] = corr_df["spearman_p_fdr"] < alpha

    corr_df["spearman_effect_bin"] = corr_df["spearman_rho"].apply(_bin_effect)

    # Save results
    corr_df.to_csv(name+"/correlation_results_with_pvalues"+name+".csv", index=False)
    print("💾 Saved correlation results to correlation_results_with_pvalues.csv")

    output_lines = []
    output_lines.append("\n👥 OWNER TYPE ANALYSIS")

    for metric in metrics:
        org = df[df["owner_is_org"] == 1][metric]
        user = df[df["owner_is_org"] == 0][metric]

        stat, p = mannwhitneyu(org, user, alternative="two-sided")

        line = f"  {metric}: U={stat:.1f}, p={p:.4f}"
        print(line)
        output_lines.append(line)

    # Save to file
    with open(name+"/owner_type_analysis_"+name+".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\n📁 Results saved to owner_type_analysis.txt")


