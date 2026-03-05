import os
import numpy as np
import pandas as pd

# target programming languages
LANGUAGES = ["Java", "Python", "TypeScript", "JavaScript", "C#"]

META_PATH_TEMPLATE = "../rq2/all_templates_update_metrics_{lang}.csv"
SONAR_PATH_TEMPLATE = "../rq2/correlation_complete/sonar_maintenance_metrics_{lang}_complete.csv"

OUTDIR = "rq3_samples"
os.makedirs(OUTDIR, exist_ok=True)

ISSUE_METRICS = [
    "bugs", "code_smells", "vulnerabilities", "security_hotspots",
    "vuln_blocker", "vuln_critical", "vuln_major", "vuln_minor", "vuln_info",
]

EXPOSURE_CANDIDATES = ["ncloc", "files", "functions"]

# standard sample sizes
N_BEST = 20
N_WORST = 20
N_MIXED = 10
RANDOM_SEED = 42

# Stratification controls
STRATIFY_BY_OWNER = True
STRATIFY_BY_POPULARITY = True

POPULARITY_BINS = [0, 1.5, 3.0, 5.0, np.inf]
POPULARITY_LABELS = ["very_low", "low", "mid", "high"]


def _safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def robust_percentile_rank(x):
    x = _safe_to_numeric(x)
    return x.rank(pct=True)

def choose_exposure(df):
    for c in EXPOSURE_CANDIDATES:
        if c in df.columns:
            s = _safe_to_numeric(df[c])
            return c, s
    return None, pd.Series(np.nan, index=df.index)

def make_strata(df):
    strata_cols = []

    if STRATIFY_BY_OWNER and "owner_is_org" in df.columns:
        df["str_owner"] = (
            df["owner_is_org"]
            .fillna(-1)
            .astype(int)
            .map({1: "org", 0: "user"})
            .fillna("unknown")
        )
        strata_cols.append("str_owner")

    if STRATIFY_BY_POPULARITY and "stars" in df.columns:
        df["log_stars"] = np.log1p(_safe_to_numeric(df["stars"]).fillna(0))
        df["str_pop"] = pd.cut(
            df["log_stars"],
            bins=POPULARITY_BINS,
            labels=POPULARITY_LABELS,
            include_lowest=True,
        )
        df["str_pop"] = df["str_pop"].astype("object").fillna("unknown")
        strata_cols.append("str_pop")

    if not strata_cols:
        df["strata"] = "all"
    else:
        df["strata"] = df[strata_cols].astype(str).agg("|".join, axis=1)

    return df


def stratified_pick(df, n, sort_col, ascending=True, seed=RANDOM_SEED):
    df = df.copy().sort_values(sort_col, ascending=ascending)

    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    if "strata" not in df.columns:
        return df.head(n).copy()

    strata = df["strata"].fillna("all")
    unique_strata = strata.unique().tolist()

    base = n // len(unique_strata)
    rem = n % len(unique_strata)

    picks = []
    used_idx = set()

    _ = np.random.default_rng(seed)

    for i, st in enumerate(unique_strata):
        quota = base + (1 if i < rem else 0)
        subset = df[df["strata"] == st]
        if quota <= 0 or subset.empty:
            continue

        chosen = subset.head(quota)
        picks.append(chosen)
        used_idx.update(chosen.index.tolist())

    picked = pd.concat(picks, axis=0) if picks else df.iloc[0:0].copy()

    if len(picked) < n:
        remaining = df.loc[~df.index.isin(list(used_idx))]
        needed = n - len(picked)
        picked = pd.concat([picked, remaining.head(needed)], axis=0)

    return picked


def build_scores(df, issue_metrics):
    df = df.copy()
    exposure_col, exposure = choose_exposure(df)
    EPS = 1e-9

    available = [m for m in issue_metrics if m in df.columns]
    if not available:
        raise ValueError("None of the issue metrics were found in the merged dataframe.")

    if exposure_col is not None:
        denom = exposure.fillna(0) + EPS
        for m in available:
            df[f"{m}_per_{exposure_col}"] = _safe_to_numeric(df[m]) / denom
        metric_cols = [f"{m}_per_{exposure_col}" for m in available]
    else:
        metric_cols = available

    for c in metric_cols:
        df[f"pr_{c}"] = robust_percentile_rank(df[c])

    pr_cols = [f"pr_{c}" for c in metric_cols]

    df["composite_score"] = df[pr_cols].mean(axis=1, skipna=True)
    df["mixedness_score"] = df[pr_cols].std(axis=1, skipna=True)
    df["n_metrics_used"] = df[pr_cols].notna().sum(axis=1)

    return df, metric_cols, pr_cols, exposure_col


def add_repo_context_fields(df):
    df = df.copy()

    # parse columns if they exist
    for col in ["created_at", "updated_at", "last_commit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    today = pd.Timestamp.now().tz_localize(None)

    # compute repo_age_days if not available
    if "repo_age_days" not in df.columns and "created_at" in df.columns:
        print("Adding age days")
        df["repo_age_days"] = (today - df["created_at"]).dt.days

    # Compute staleness_days if not available
    if "staleness_days" not in df.columns and "updated_at" in df.columns:
        print("Adding staleness days")
        df["staleness_days"] = (today - df["updated_at"]).dt.days

    return df


def make_repo_context_string(row):
    parts = []

    if pd.notna(row.get("repo_age_days")):
        parts.append(f"age={int(row.repo_age_days)}d")

    if pd.notna(row.get("staleness_days")):
        parts.append(f"stale={int(row.staleness_days)}d")

    if pd.notna(row.get("commits_last_12m")):
        parts.append(f"commits12m={int(row.commits_last_12m)}")

    if pd.notna(row.get("stars")):
        parts.append(f"stars={int(row.stars)}")

    if pd.notna(row.get("forks")):
        parts.append(f"forks={int(row.forks)}")

    return ", ".join(parts)


def run_for_language(lang):
    meta_path = META_PATH_TEMPLATE.format(lang=lang)
    sonar_path = SONAR_PATH_TEMPLATE.format(lang=lang)

    df_meta = pd.read_csv(meta_path)
    df_sonar = pd.read_csv(sonar_path)

    if "repo_url" not in df_sonar.columns:
        raise ValueError(f"[{lang}] 'repo_url' column missing in Sonar file")

    # Owner type -> owner_is_org
    if "owner_type" in df_meta.columns and "owner_is_org" not in df_meta.columns:
        df_meta["owner_is_org"] = df_meta["owner_type"].apply(
            lambda x: 1 if str(x).lower() == "organization" else 0
        )

    # Merge
    df = df_meta.merge(
        df_sonar,
        left_on="original_link_repository",
        right_on="repo_url",
        how="inner",
    )

    # Add contextual fields (dates/age/staleness) BEFORE sampling/export
    df = add_repo_context_fields(df)

    # Build scores (ranking)
    df, metric_cols, pr_cols, exposure_col = build_scores(df, ISSUE_METRICS)

    # Require at least some fraction of metrics present
    min_needed = max(3, int(0.5 * len(metric_cols)))  # at least 3 or 50% of metrics
    df = df[df["n_metrics_used"] >= min_needed].copy()

    # Strata for balanced sampling
    df = make_strata(df)

    # Best / Worst
    best = stratified_pick(df, N_BEST, "composite_score", ascending=True)
    worst = stratified_pick(df, N_WORST, "composite_score", ascending=False)

    # Mixed
    selected_idx = set(best.index.tolist()) | set(worst.index.tolist())
    mixed_pool = df.loc[~df.index.isin(list(selected_idx))].copy()
    mixed = stratified_pick(mixed_pool, N_MIXED, "mixedness_score", ascending=False)

    sample = pd.concat(
        [
            best.assign(sample_group="best"),
            worst.assign(sample_group="worst"),
            mixed.assign(sample_group="mixed"),
        ],
        axis=0,
    )

    # Explainability: top-3 worst metrics per repo (by percentile ranks)
    def top_k_metrics(row, k=3):
        vals = []
        for c in pr_cols:
            v = row.get(c)
            if pd.notna(v):
                vals.append((c.replace("pr_", ""), float(v)))
        vals.sort(key=lambda t: t[1], reverse=True)
        return "; ".join([f"{m}:{p:.2f}" for m, p in vals[:k]])

    sample["top_worst_metrics"] = sample.apply(top_k_metrics, axis=1)

    # Optional: short context string for quick manual scanning
    sample["repo_context"] = sample.apply(make_repo_context_string, axis=1)

    # Helpful columns to export (context + identity + ranking)
    keep_cols = [
        # Grouping
        "sample_group",

        # Identity
        "repo_url",
        "original_link_repository",

        # Ownership / governance
        "owner_type" if "owner_type" in sample.columns else None,
        "owner_is_org",

        # Dates / time
        "created_at" if "created_at" in sample.columns else None,
        "updated_at" if "updated_at" in sample.columns else None,
        "last_commit_date" if "last_commit_date" in sample.columns else None,
        "repo_age_days" if "repo_age_days" in sample.columns else None,
        "staleness_days" if "staleness_days" in sample.columns else None,

        # Adoption / activity
        "stars" if "stars" in sample.columns else None,
        "forks" if "forks" in sample.columns else None,
        "commits_last_12m" if "commits_last_12m" in sample.columns else None,

        # Size / structure (often useful for interpretation)
        "ncloc" if "ncloc" in sample.columns else None,
        "files" if "files" in sample.columns else None,
        "functions" if "functions" in sample.columns else None,

        # Scores
        "composite_score",
        "mixedness_score",
        "n_metrics_used",

        # Stratification and explainability
        "strata",
        "repo_context",
        "top_worst_metrics",
    ]
    keep_cols = [c for c in keep_cols if c is not None]

    # Save outputs
    lang_dir = os.path.join(OUTDIR, lang)
    os.makedirs(lang_dir, exist_ok=True)

    sample_out = os.path.join(lang_dir, f"sample_selected_{lang}.csv")
    score_out = os.path.join(lang_dir, f"scoring_table_{lang}.csv")

    sample[keep_cols].drop_duplicates("repo_url").to_csv(sample_out, index=False)
    df.to_csv(score_out, index=False)

    print(f"[{lang}] merged={len(df)} | sample={len(sample)} | exposure={exposure_col}")
    print(f"  -> {sample_out}")
    print(f"  -> {score_out}")


if __name__ == "__main__":
    for lang in LANGUAGES:
        run_for_language(lang)
