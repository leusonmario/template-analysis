import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================

LANGUAGES = ["Java", "Python", "C#", "TypeScript", "JavaScript"]

PREDICTORS = [
    "staleness_days",
    "commits_last_12m",
    "repo_age_days",
    "log_stars",
    "log_forks",
    "owner_is_org"
]

COUNT_OUTCOMES = [
    "bugs",
    "code_smells",
    "vulnerabilities",
    "security_hotspots",
    "vuln_blocker",
    "vuln_critical",
    "vuln_major",
    "vuln_minor",
    "vuln_info"
]

ALPHA = 0.05


# ============================================================
# Negative Binomial Regression
# ============================================================

def run_negative_binomial(dep_var, df, predictors, language, exposure_col="nloc"):
    """
    Negative Binomial GLM for count outcomes with robust SE (HC3),
    using an offset to model rates when exposure_col is available.

    Returns:
        model: fitted statsmodels model
        result_df: coefficient-level dataframe
    """

    print(f"\n📈 NEGATIVE BINOMIAL REGRESSION ({dep_var}) [{language}]\n")

    # ---------------------------
    # 1) Select and clean columns
    # ---------------------------
    needed_cols = [dep_var] + predictors
    use_offset = exposure_col in df.columns

    if use_offset:
        needed_cols.append(exposure_col)

    d = df[needed_cols].copy()

    for col in needed_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    # Dependent variable: non-negative integer
    d[dep_var] = d[dep_var].clip(lower=0)
    d[dep_var] = d[dep_var].round().astype(int)

    # ---------------------------
    # 2) Cap extreme vulnerability-severity counts
    # ---------------------------
    if dep_var.startswith("vuln_"):
        q99 = np.percentile(d[dep_var], 99)
        cap = int(max(q99, 10))

        num_capped = (d[dep_var] > cap).sum()

        print(f"99th percentile for {dep_var}: {q99:.3f} → cap used: {cap}")
        print(f"Number capped: {num_capped} out of {len(d)}")

        d[dep_var] = d[dep_var].clip(upper=cap)

    d = d[d[dep_var] >= 0]

    # ---------------------------
    # 3) Offset: log(exposure in KLOC)
    # ---------------------------
    offset = None

    if use_offset:
        d = d[d[exposure_col] > 0]
        offset = np.log(d[exposure_col] / 1000.0)

    print("Number of rows used in model:", len(d))

    if len(d) == 0:
        raise ValueError(f"No valid rows available for {dep_var} in {language}.")

    if not np.isfinite(d[dep_var]).all():
        raise ValueError("Dependent variable has non-finite values after cleaning.")

    if not np.isfinite(d[predictors].values).all():
        raise ValueError("Predictors contain non-finite values after cleaning.")

    if offset is not None and not np.isfinite(offset).all():
        raise ValueError("Offset contains non-finite values after cleaning.")

    # ---------------------------
    # 4) Build and fit model
    # ---------------------------
    rhs = " + ".join(predictors)
    formula = f"{dep_var} ~ {rhs}"

    nb_family = sm.families.NegativeBinomial(alpha=1.0)

    nb_mod = smf.glm(
        formula=formula,
        data=d,
        family=nb_family,
        offset=offset
    )

    try:
        print("➡️ Trying standard NB GLM (IRLS)...")
        model = nb_mod.fit(cov_type="HC3", maxiter=100)
        used_regularized = False

    except ValueError as e:
        print("⚠️ Standard NB GLM failed:", e)
        print("➡️ Using ridge-regularized Negative Binomial.")

        model = nb_mod.fit_regularized(
            alpha=1e-2,
            L1_wt=0.0,
            maxiter=500
        )

        used_regularized = True

    # ---------------------------
    # 5) Save model summary
    # ---------------------------
    try:
        summary_text = str(model.summary())
        print(model.summary())

    except NotImplementedError:
        summary_text = "Coefficients:\n" + str(model.params)
        print(summary_text)

        if hasattr(model, "bse"):
            summary_text += "\n\nStandard errors:\n" + str(model.bse)

    fname = f"{language}/regression_reb_{dep_var}_{language}.txt"

    with open(fname, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"💾 Saved regression output to {fname}")

    # ---------------------------
    # 6) Extract coefficient-level results
    # ---------------------------
    rows = []
    params = model.params

    if not used_regularized:
        bse = model.bse
        pvalues = model.pvalues
        conf = model.conf_int()

        for term in params.index:
            if term == "Intercept":
                continue

            coef = params[term]

            rows.append({
                "language": language,
                "outcome": dep_var,
                "predictor": term,
                "n": len(d),
                "coef": coef,
                "std_error": bse[term],
                "p_raw": pvalues[term],
                "IRR": np.exp(coef),
                "IRR_CI_low": np.exp(conf.loc[term, 0]),
                "IRR_CI_high": np.exp(conf.loc[term, 1]),
                "used_regularized": used_regularized
            })

    else:
        for term in params.index:
            if term == "Intercept":
                continue

            coef = params[term]

            rows.append({
                "language": language,
                "outcome": dep_var,
                "predictor": term,
                "n": len(d),
                "coef": coef,
                "std_error": np.nan,
                "p_raw": np.nan,
                "IRR": np.exp(coef),
                "IRR_CI_low": np.nan,
                "IRR_CI_high": np.nan,
                "used_regularized": used_regularized
            })

    result_df = pd.DataFrame(rows)

    return model, result_df


# ============================================================
# Multiple-comparison correction
# ============================================================

def apply_corrections(regression_results):
    """
    Adds raw, BH, and Bonferroni significance columns:
      - globally
      - by language
      - by outcome
    """

    regression_results = regression_results.copy()

    mask = regression_results["p_raw"].notna()

    regression_results["sig_raw"] = regression_results["p_raw"] < ALPHA

    # ---------------------------
    # Global corrections
    # ---------------------------
    regression_results["p_adj_bh_global"] = np.nan
    regression_results["sig_bh_global"] = False
    regression_results["p_adj_bonf_global"] = np.nan
    regression_results["sig_bonf_global"] = False

    if mask.sum() > 0:
        reject_bh, p_bh, _, _ = multipletests(
            regression_results.loc[mask, "p_raw"],
            alpha=ALPHA,
            method="fdr_bh"
        )

        reject_bonf, p_bonf, _, _ = multipletests(
            regression_results.loc[mask, "p_raw"],
            alpha=ALPHA,
            method="bonferroni"
        )

        regression_results.loc[mask, "p_adj_bh_global"] = p_bh
        regression_results.loc[mask, "sig_bh_global"] = reject_bh

        regression_results.loc[mask, "p_adj_bonf_global"] = p_bonf
        regression_results.loc[mask, "sig_bonf_global"] = reject_bonf

    # ---------------------------
    # Corrections by language
    # ---------------------------
    regression_results["p_adj_bh_by_language"] = np.nan
    regression_results["sig_bh_by_language"] = False
    regression_results["p_adj_bonf_by_language"] = np.nan
    regression_results["sig_bonf_by_language"] = False

    for language, group in regression_results.groupby("language"):
        idx = group[group["p_raw"].notna()].index

        if len(idx) == 0:
            continue

        reject_bh, p_bh, _, _ = multipletests(
            regression_results.loc[idx, "p_raw"],
            alpha=ALPHA,
            method="fdr_bh"
        )

        reject_bonf, p_bonf, _, _ = multipletests(
            regression_results.loc[idx, "p_raw"],
            alpha=ALPHA,
            method="bonferroni"
        )

        regression_results.loc[idx, "p_adj_bh_by_language"] = p_bh
        regression_results.loc[idx, "sig_bh_by_language"] = reject_bh

        regression_results.loc[idx, "p_adj_bonf_by_language"] = p_bonf
        regression_results.loc[idx, "sig_bonf_by_language"] = reject_bonf

    # ---------------------------
    # Corrections by outcome
    # ---------------------------
    regression_results["p_adj_bh_by_outcome"] = np.nan
    regression_results["sig_bh_by_outcome"] = False
    regression_results["p_adj_bonf_by_outcome"] = np.nan
    regression_results["sig_bonf_by_outcome"] = False

    for outcome, group in regression_results.groupby("outcome"):
        idx = group[group["p_raw"].notna()].index

        if len(idx) == 0:
            continue

        reject_bh, p_bh, _, _ = multipletests(
            regression_results.loc[idx, "p_raw"],
            alpha=ALPHA,
            method="fdr_bh"
        )

        reject_bonf, p_bonf, _, _ = multipletests(
            regression_results.loc[idx, "p_raw"],
            alpha=ALPHA,
            method="bonferroni"
        )

        regression_results.loc[idx, "p_adj_bh_by_outcome"] = p_bh
        regression_results.loc[idx, "sig_bh_by_outcome"] = reject_bh

        regression_results.loc[idx, "p_adj_bonf_by_outcome"] = p_bonf
        regression_results.loc[idx, "sig_bonf_by_outcome"] = reject_bonf

    return regression_results


# ============================================================
# Main analysis
# ============================================================

all_regression_rows = []

for language in LANGUAGES:

    print("\n" + "=" * 80)
    print(f"Processing language: {language}")
    print("=" * 80)

    os.makedirs(language, exist_ok=True)

    # ---------------------------
    # 1) Load datasets
    # ---------------------------
    df_meta = pd.read_csv(f"../all_templates_update_metrics_{language}.csv")
    df_sonar = pd.read_csv(
        f"../correlation_complete/sonar_maintenance_metrics_{language}_complete.csv"
    )

    if "repo_url" not in df_sonar.columns:
        raise ValueError(f"❌ ERROR: 'repo_url' column missing in Sonar results for {language}.")

    # ---------------------------
    # 2) Clean and transform metadata
    # ---------------------------
    df_meta["created_at"] = (
        pd.to_datetime(df_meta["created_at"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    df_meta["updated_at"] = (
        pd.to_datetime(df_meta["updated_at"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    today = pd.Timestamp.now().tz_localize(None)

    df_meta["repo_age_days"] = (today - df_meta["created_at"]).dt.days

    # Keep this only if staleness_days is not already present.
    if "staleness_days" not in df_meta.columns:
        df_meta["staleness_days"] = (today - df_meta["updated_at"]).dt.days

    df_meta["log_stars"] = np.log1p(df_meta["stars"].fillna(0))
    df_meta["log_forks"] = np.log1p(df_meta["forks"].fillna(0))

    df_meta["owner_is_org"] = df_meta["owner_type"].apply(
        lambda x: 1 if str(x).lower() == "organization" else 0
    )

    # ---------------------------
    # 3) Merge metadata and Sonar metrics
    # ---------------------------
    df = df_meta.merge(
        df_sonar,
        left_on="original_link_repository",
        right_on="repo_url",
        how="inner"
    )

    print(f"👉 Merged dataset size for {language}: {len(df)} rows")

    # ---------------------------
    # 4) VIF
    # ---------------------------
    print("\n📏 Checking multicollinearity (VIF)...")

    X = (
        df[PREDICTORS]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    X_vif = X.copy()
    X_vif["intercept"] = 1.0

    vif_rows = []

    for i, col in enumerate(X_vif.columns):
        vif = variance_inflation_factor(X_vif.values, i)
        vif_rows.append([col, vif])

    vif_df = pd.DataFrame(vif_rows, columns=["variable", "VIF"])

    vif_df_no_intercept = (
        vif_df[vif_df["variable"] != "intercept"]
        .sort_values("VIF", ascending=False)
    )

    print(vif_df_no_intercept)

    vif_out = f"{language}/vif_results_reb_{language}.csv"
    vif_df_no_intercept.to_csv(vif_out, index=False)

    print(f"💾 Saved VIF results to {vif_out}")

    # ---------------------------
    # 5) Negative Binomial regressions
    # ---------------------------
    models = {}

    for outcome in COUNT_OUTCOMES:
        if outcome not in df.columns:
            print(f"⚠️ Skipping {outcome}: column not found in {language}.")
            continue

        model, result_df = run_negative_binomial(
            dep_var=outcome,
            df=df,
            predictors=PREDICTORS,
            language=language,
            exposure_col="nloc"
        )

        models[outcome] = model
        all_regression_rows.append(result_df)


# ============================================================
# Apply corrections after all languages are processed
# ============================================================

print("\n" + "=" * 80)
print("Applying multiple-comparison corrections")
print("=" * 80)

regression_results = pd.concat(all_regression_rows, ignore_index=True)

regression_results = apply_corrections(regression_results)

regression_results.to_csv(
    "negative_binomial_regression_results_with_bh_and_bonferroni.csv",
    index=False
)

print("✅ Saved full corrected regression results.")


# ============================================================
# Summary by language
# ============================================================

summary_by_language = (
    regression_results
    .groupby("language")
    .agg(
        total_tests=("p_raw", "count"),
        raw_sig=("sig_raw", "sum"),
        bh_by_language_sig=("sig_bh_by_language", "sum"),
        bonf_by_language_sig=("sig_bonf_by_language", "sum"),
        bh_global_sig=("sig_bh_global", "sum"),
        bonf_global_sig=("sig_bonf_global", "sum")
    )
    .reset_index()
)

summary_by_language.to_csv(
    "negative_binomial_correction_summary_by_language.csv",
    index=False
)

print("\n📊 Correction summary by language:")
print(summary_by_language)


# ============================================================
# Supported claims summary
# ============================================================

supported_claims_summary = (
    regression_results
    .groupby(["language", "outcome", "predictor"])
    .agg(
        n=("n", "first"),
        coef=("coef", "first"),
        IRR=("IRR", "first"),
        IRR_CI_low=("IRR_CI_low", "first"),
        IRR_CI_high=("IRR_CI_high", "first"),
        p_raw=("p_raw", "first"),
        p_adj_bh_by_language=("p_adj_bh_by_language", "first"),
        p_adj_bonf_by_language=("p_adj_bonf_by_language", "first"),
        sig_raw=("sig_raw", "first"),
        sig_bh_by_language=("sig_bh_by_language", "first"),
        sig_bonf_by_language=("sig_bonf_by_language", "first"),
        p_adj_bh_global=("p_adj_bh_global", "first"),
        p_adj_bonf_global=("p_adj_bonf_global", "first"),
        sig_bh_global=("sig_bh_global", "first"),
        sig_bonf_global=("sig_bonf_global", "first"),
        used_regularized=("used_regularized", "first")
    )
    .reset_index()
)

supported_claims_summary.to_csv(
    "negative_binomial_supported_claims_summary_with_bonferroni.csv",
    index=False
)

print("\n✅ Saved supported-claims summary.")

print("\nAssociations surviving BH correction by language:")
print(
    supported_claims_summary[
        supported_claims_summary["sig_bh_by_language"] == True
    ][
        [
            "language",
            "outcome",
            "predictor",
            "IRR",
            "p_raw",
            "p_adj_bh_by_language"
        ]
    ]
)

print("\nAssociations surviving Bonferroni correction by language:")
print(
    supported_claims_summary[
        supported_claims_summary["sig_bonf_by_language"] == True
    ][
        [
            "language",
            "outcome",
            "predictor",
            "IRR",
            "p_raw",
            "p_adj_bonf_by_language"
        ]
    ]
)