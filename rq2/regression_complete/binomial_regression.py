import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_negative_binomial(dep_var, df, predictors, name, exposure_col="nloc"):
    """
    Negative Binomial GLM for count outcomes with robust SE (HC3),
    using an offset to model *rates* when exposure_col is available.
    Includes:
      - cleaning (NaN/inf removal)
      - clipping/capping extreme counts
      - regularized fallback if IRLS fails
    """

    print(f"\n📈 NEGATIVE BINOMIAL REGRESSION ({dep_var}) [{name}]\n")

    # ---------------------------
    # 1) Select and clean columns
    # ---------------------------
    needed_cols = [dep_var] + predictors
    use_offset = exposure_col in df.columns

    if use_offset:
        needed_cols.append(exposure_col)

    d = df[needed_cols].copy()

    # Make sure everything is numeric
    for col in needed_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Drop NaN / inf
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    # Dep var: non-negative integer
    d[dep_var] = d[dep_var].clip(lower=0)
    d[dep_var] = d[dep_var].round().astype(int)

    # ---------------------------
    # 2) Cap extreme counts (like before)
    # ---------------------------
    if dep_var.startswith("vuln_"):
        q99 = np.percentile(d[dep_var], 99)
        cap = int(max(q99, 10))  # at least 10
        print(f"99th percentile for {dep_var}: {q99:.3f} → cap used: {cap}")
        num_capped = (d[dep_var] > cap).sum()
        print(f"Number capped: {num_capped} out of {len(d)}")
        d[dep_var] = d[dep_var].clip(upper=cap)
        print(f"🔒 Capping {dep_var} at {cap} to avoid extreme counts.")

    # Ensure non-negative after capping (just in case)
    d = d[d[dep_var] >= 0]

    # ---------------------------
    # 3) Offset: log(exposure in KLOC)
    # ---------------------------
    offset = None
    if use_offset:
        # remove zero/negative exposure
        d = d[d[exposure_col] > 0]
        offset = np.log(d[exposure_col] / 1000.0)

    print("Number of rows used in model:", len(d))

    # Sanity check for finiteness (debug helper)
    if not np.isfinite(d[dep_var]).all():
        raise ValueError("Dependent variable has non-finite values after cleaning.")
    if not np.isfinite(d[predictors].values).all():
        raise ValueError("Predictors contain non-finite values after cleaning.")
    if offset is not None and not np.isfinite(offset).all():
        raise ValueError("Offset contains non-finite values after cleaning.")

    # ---------------------------
    # 4) Build formula and model
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

    # ---------------------------
    # 5) Try standard IRLS, fallback to regularized
    # ---------------------------
    try:
        print("➡️ Trying standard NB GLM (IRLS)...")
        model = nb_mod.fit(cov_type="HC3", maxiter=100)
        used_regularized = False
    except ValueError as e:
        print("⚠️ Standard NB GLM failed:", e)
        print("➡️ Using ridge-regularized Negative Binomial to obtain a finite solution.")
        model = nb_mod.fit_regularized(
            alpha=1e-2,   # penalty strength (can tweak)
            L1_wt=0.0,
            maxiter=500
        )
        used_regularized = True

    # ---------------------------
    # 6) Print + save summary
    # ---------------------------
    try:
        summary_text = str(model.summary())
        print(model.summary())
    except NotImplementedError:
        # regularized GLM often lacks summary()
        print("ℹ️ summary() not implemented for this model (likely regularized GLM).")
        summary_text = "Coefficients:\n" + str(model.params)
        print(summary_text)
        if hasattr(model, "bse"):
            se_text = "\n\nStandard errors:\n" + str(model.bse)
            summary_text += se_text
            print(se_text)

    fname = f"{name}/regression_{dep_var}_{name}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"💾 Saved regression output to {fname}\n")

    if used_regularized:
        print("⚠️ NOTE: This fit used ridge-regularized Negative Binomial; "
              "coefficients are shrunk and some warnings are expected.")

    return model
languages = ["Java", "Python", "C#", "TypeScript", "JavaScript"]

for name in languages:

    os.mkdir(name)
    #############################################
    # 1. Load datasets
    #############################################

    print("📌 Loading datasets...")

    df_meta = pd.read_csv("../all_templates_update_metrics_" + name + ".csv")
    # df_sonar = pd.read_csv("../vulnerability_results_Python_all.csv")
    df_sonar = pd.read_csv("../correlation_complete/sonar_maintenance_metrics_" + name + "_complete.csv")

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

    df = df_meta.merge(
        df_sonar,
        left_on="original_link_repository",
        right_on="repo_url",
        how="inner"
    )

    #############################################
    # 6. Multicollinearity Diagnostics (VIF)
    #############################################

    print("\n📏 Checking multicollinearity (VIF)...")

    predictors = [
        "staleness_days",
        "commits_last_12m",
        "repo_age_days",
        "log_stars",
        "log_forks",
        "owner_is_org"
    ]

    # Use the exact predictors used in the regressions (drop missing rows consistently)
    X = df[predictors].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    # Add intercept for VIF computation (required by statsmodels VIF)
    X_vif = X.copy()
    X_vif["intercept"] = 1.0

    vif_rows = []
    for i, col in enumerate(X_vif.columns):
        vif = variance_inflation_factor(X_vif.values, i)
        vif_rows.append([col, vif])

    vif_df = pd.DataFrame(vif_rows, columns=["variable", "VIF"])

    # Intercept VIF is not interpretable; keep or drop it
    vif_df_no_intercept = vif_df[vif_df["variable"] != "intercept"].sort_values("VIF", ascending=False)

    print(vif_df_no_intercept)

    vif_out = f"{name}/vif_results_{name}.csv"
    vif_df_no_intercept.to_csv(vif_out, index=False)
    print(f"\n💾 Saved VIF results to {vif_out}\n")

    #############################################
    # 7. Negative Binomial Regressions (with offset)
    #############################################

    # Example: run NB on RAW counts with offset (preferred if you have nloc)
    count_outcomes = [
        "bugs", "code_smells", "vulnerabilities", "security_hotspots",
        "vuln_blocker", "vuln_critical", "vuln_major", "vuln_minor", "vuln_info"
    ]

    models = {}
    for y in count_outcomes:
        if y in df.columns:
            models[y] = run_negative_binomial(y, df, predictors, name, exposure_col="nloc")

