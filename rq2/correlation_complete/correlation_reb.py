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
all_mw_rows = []

for name in languages:
    #############################################
    # 1. Load datasets
    #############################################

    print("📌 Loading datasets...")

    os.makedirs(name, exist_ok=True)

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
    corr_df.to_csv(name+"/correlation_results_with_pvalues_reb_"+name+".csv", index=False)
    print("💾 Saved correlation results to correlation_results_with_pvalues.csv")

    #############################################
    # 5. Owner Type Analysis: Mann–Whitney U
    #############################################

    print("\n👥 OWNER TYPE ANALYSIS: Mann–Whitney U")

    mw_rows_lang = []

    for metric in metrics:
        tmp = df[["owner_is_org", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

        org = tmp[tmp["owner_is_org"] == 1][metric]
        user = tmp[tmp["owner_is_org"] == 0][metric]

        n_org = len(org)
        n_user = len(user)

        if n_org == 0 or n_user == 0:
            row = {
                "language": name,
                "metric": metric,
                "n_org": n_org,
                "n_user": n_user,
                "median_org": np.nan,
                "median_user": np.nan,
                "mean_org": np.nan,
                "mean_user": np.nan,
                "U": np.nan,
                "p_raw": np.nan,
                "rank_biserial": np.nan,
                "effect_bin": np.nan,
                "direction": "not_tested"
            }
        else:
            stat, p = mannwhitneyu(org, user, alternative="two-sided")

            # Rank-biserial correlation.
            # Positive: organization-owned templates tend to have higher values.
            # Negative: organization-owned templates tend to have lower values.
            rrb = (2 * stat) / (n_org * n_user) - 1

            if rrb > 0:
                direction = "org_higher"
            elif rrb < 0:
                direction = "org_lower"
            else:
                direction = "no_direction"

            row = {
                "language": name,
                "metric": metric,
                "n_org": n_org,
                "n_user": n_user,
                "median_org": org.median(),
                "median_user": user.median(),
                "mean_org": org.mean(),
                "mean_user": user.mean(),
                "U": stat,
                "p_raw": p,
                "rank_biserial": rrb,
                "effect_bin": _bin_effect(rrb),
                "direction": direction
            }

            print(
                f"  {metric}: U={stat:.1f}, p={p:.4g}, "
                f"rrb={rrb:.3f} ({_bin_effect(rrb)}), "
                f"direction={direction}, "
                f"median_org={org.median():.4f}, median_user={user.median():.4f}, "
                f"n_org={n_org}, n_user={n_user}"
            )

        mw_rows_lang.append(row)
        all_mw_rows.append(row)

    mw_df_lang = pd.DataFrame(mw_rows_lang)

    mw_df_lang.to_csv(
        f"{name}/owner_type_mannwhitney_raw_reb_{name}.csv",
        index=False
    )

    print(f"\n📁 Raw Mann–Whitney results saved for {name}.")

    #############################################
    # 6. Multiple-comparison correction for Mann–Whitney U
    #############################################

    mw_all = pd.DataFrame(all_mw_rows)

    alpha = 0.05
    mask = mw_all["p_raw"].notna()

    # Raw significance
    mw_all["sig_raw"] = mw_all["p_raw"] < alpha

    # Global correction across all ownership-comparison tests
    mw_all["p_adj_bh_global"] = np.nan
    mw_all["sig_bh_global"] = False

    if mask.sum() > 0:
        reject_global, p_adj_global, _, _ = multipletests(
            mw_all.loc[mask, "p_raw"],
            alpha=alpha,
            method="fdr_bh"
        )

        mw_all.loc[mask, "p_adj_bh_global"] = p_adj_global
        mw_all.loc[mask, "sig_bh_global"] = reject_global

    # Correction by language
    mw_all["p_adj_bh_by_language"] = np.nan
    mw_all["sig_bh_by_language"] = False

    for language, group in mw_all.groupby("language"):
        idx = group[group["p_raw"].notna()].index

        if len(idx) == 0:
            continue

        reject_lang, p_adj_lang, _, _ = multipletests(
            mw_all.loc[idx, "p_raw"],
            alpha=alpha,
            method="fdr_bh"
        )

        mw_all.loc[idx, "p_adj_bh_by_language"] = p_adj_lang
        mw_all.loc[idx, "sig_bh_by_language"] = reject_lang

    # Correction by metric
    mw_all["p_adj_bh_by_metric"] = np.nan
    mw_all["sig_bh_by_metric"] = False

    for metric, group in mw_all.groupby("metric"):
        idx = group[group["p_raw"].notna()].index

        if len(idx) == 0:
            continue

        reject_metric, p_adj_metric, _, _ = multipletests(
            mw_all.loc[idx, "p_raw"],
            alpha=alpha,
            method="fdr_bh"
        )

        mw_all.loc[idx, "p_adj_bh_by_metric"] = p_adj_metric
        mw_all.loc[idx, "sig_bh_by_metric"] = reject_metric

    mw_all.to_csv(
        "owner_type_mannwhitney_all_languages_with_fdr.csv",
        index=False
    )

    print("\n✅ Saved corrected Mann–Whitney results to owner_type_mannwhitney_all_languages_with_fdr.csv")

    summary = (
        mw_all.groupby("language")
        .agg(
            total_tests=("p_raw", "count"),
            raw_significant=("sig_raw", "sum"),
            bh_by_language_significant=("sig_bh_by_language", "sum"),
            bh_global_significant=("sig_bh_global", "sum")
        )
        .reset_index()
    )

    summary.to_csv(
        "owner_type_mannwhitney_summary_by_language.csv",
        index=False
    )

    print("\n📊 Mann–Whitney summary by language:")
    print(summary)


