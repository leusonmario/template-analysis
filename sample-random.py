#!/usr/bin/env python3
"""
Stratified sampler for GitHub repos by star distribution.

Requirements:
  - Input CSV must have a 'stars' column (int-like).
  - Optional columns honored when flags provided:
      * 'fork' (bool/int) with --drop-forks
      * 'archived' (bool/int) with --drop-archived

Plan:
  1) (Optional) Certainty head: include all repos with stars >= CERTAINTY_MIN_STARS
     and/or in the top CERTAINTY_TOP_PCT percent by stars.
  2) On remaining repos, create star strata via thresholds.
  3) Allocate with square-root allocation + per-stratum minimum.
  4) Sample without replacement; combine with certainty set.
  5) Ensure final sample size equals --n (cap if certainty > n).

Usage (example):
  python strat_sample_repos.py \
      --input repos.csv \
      --output sample.csv \
      --n 320 \
      --seed 42 \
      --bins 0,5,20,100,500,2000,inf \
      --certainty-min-stars 2000 \
      --certainty-top-pct 0.0 \
      --min-per-stratum 20 \
      --drop-forks \
      --drop-archived
"""

import argparse
import math
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_bins(bins_arg: str) -> List[float]:
    vals: List[float] = []
    for tok in bins_arg.split(","):
        tok = tok.strip().lower()
        if tok in ("inf", "+inf", "infinity"):
            vals.append(float("inf"))
        else:
            try:
                vals.append(float(tok))
            except ValueError:
                raise SystemExit(f"Invalid bin edge: {tok}")
    if not math.isinf(vals[-1]):
        vals.append(float("inf"))
    if any(vals[i+1] <= vals[i] for i in range(len(vals)-1)):
        raise SystemExit("Bin edges must be strictly increasing.")
    return vals


def build_strata_labels(bins: List[float]) -> List[str]:
    labels = []
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]
        if math.isinf(high):
            labels.append(f"[{int(low)}–∞]")
        else:
            labels.append(f"[{int(low)}–{int(high - 1)}]")
    return labels


def assign_strata(series: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
    # Treat intervals as [low, high) on integer stars, matching labels like [0–4], [5–19], ...
    # stars are coerced to int >= 0
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return pd.cut(s, bins=bins, right=False, labels=labels, include_lowest=True)


def neyman_sqrt_allocation(N_h: np.ndarray, n_total: int, min_per_stratum: int) -> np.ndarray:
    """Square-root allocation with guaranteed minimum per non-empty stratum and cap by N_h."""
    k = len(N_h)
    alloc = np.zeros(k, dtype=int)
    nonempty = N_h > 0

    # Satisfy per-stratum minimums (capped by N_h)
    base_need = np.minimum(N_h, np.where(nonempty, min_per_stratum, 0))
    base_sum = int(base_need.sum())

    if base_sum >= n_total:
        # We have to trim down to n_total.
        # Greedy trim from strata with larger N_h first.
        alloc = base_need.copy()
        over = base_sum - n_total
        order = np.argsort(-N_h)  # descending by size
        for idx in order:
            if over == 0:
                break
            can_trim = max(0, alloc[idx] - 1)
            if can_trim == 0:
                continue
            delta = min(can_trim, over)
            alloc[idx] -= delta
            over -= delta
        return alloc

    # Allocate remaining by sqrt(N_h)
    alloc = base_need.copy()
    n_left = n_total - base_sum
    weights = np.sqrt(N_h.astype(float))
    weights[~nonempty] = 0.0
    wsum = weights.sum()
    if wsum <= 0 or n_left <= 0:
        return alloc

    fractional = weights / wsum * n_left
    extra_floor = np.floor(fractional).astype(int)
    rema = fractional - extra_floor

    alloc += extra_floor
    remaining = n_left - int(extra_floor.sum())

    if remaining > 0:
        order = np.argsort(-rema)  # largest remainders first
        for i in range(remaining):
            alloc[order[i]] += 1

    # Cap by available items
    alloc = np.minimum(alloc, N_h)

    # If capping created a deficit, redistribute to strata with room by remainder priority
    deficit = n_total - int(alloc.sum())
    if deficit > 0:
        room = N_h - alloc
        order = np.argsort(-rema)
        for idx in order:
            if deficit == 0:
                break
            if room[idx] > 0:
                alloc[idx] += 1
                deficit -= 1

    return alloc


def main():
    p = argparse.ArgumentParser(description="Stratified subsampling of GitHub repos by star distribution.")
    p.add_argument("--input", required=True, help="Path to input CSV (must contain 'stars').")
    p.add_argument("--output", required=True, help="Path to write sampled CSV.")
    p.add_argument("--n", type=int, required=True, help="Target sample size.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Strata / certainty config
    p.add_argument("--bins", default="0,5,20,100,500,2000,inf",
                   help="Comma-separated star thresholds (default: 0,5,20,100,500,2000,inf).")
    p.add_argument("--min-per-stratum", type=int, default=20,
                   help="Minimum per non-empty stratum (capped by stratum size). Default: 20.")
    p.add_argument("--certainty-min-stars", type=int, default=2000,
                   help="Include all repos with stars >= this value (default: 2000). Set 0 to disable.")
    p.add_argument("--certainty-top-pct", type=float, default=0.0,
                   help="Also include top X%% by stars with certainty (default: 0.0).")

    # Optional filters
    p.add_argument("--drop-forks", action="store_true", help="If 'fork' column exists, drop fork==True/1.")
    p.add_argument("--drop-archived", action="store_true", help="If 'archived' column exists, drop archived==True/1.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load
    df = pd.read_csv(args.input)
    if "stars" not in df.columns:
        sys.exit("ERROR: Input CSV must contain a 'stars' column.")
    df = df.copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype(int)
    if (df["stars"] < 0).any():
        # sanitize any negative stars just in case
        df.loc[df["stars"] < 0, "stars"] = 0

    # Optional filters
    if args.drop_forks and "fork" in df.columns:
        df = df[~df["fork"].astype(bool)].copy()
    if args.drop_archived and "archived" in df.columns:
        df = df[~df["archived"].astype(bool)].copy()

    N_pop = len(df)
    if N_pop == 0:
        sys.exit("ERROR: No rows to sample after filtering.")

    # --- Certainty head selection
    certainty_idx = pd.Index([])
    if args.certainty_min_stars and args.certainty_min_stars > 0:
        certainty_idx = certainty_idx.union(df.index[df["stars"] >= int(args.certainty_min_stars)])

    if args.certainty_top_pct and args.certainty_top_pct > 0:
        k = max(1, int(math.ceil(N_pop * (args.certainty_top_pct / 100.0))))
        head_idx = df.nlargest(k, "stars").index
        certainty_idx = certainty_idx.union(head_idx)

    certainty_idx = certainty_idx.unique()
    certainty = df.loc[certainty_idx]
    remaining = df.drop(index=certainty_idx)

    # If certainty already exceeds target n, cap to top-n by stars and finish.
    if len(certainty) >= args.n:
        capped = certainty.nlargest(args.n, "stars")
        capped.to_csv(args.output, index=False)
        print(f"[INFO] Certainty set ({len(certainty)}) >= target n ({args.n}). "
              f"Wrote top-{args.n} certainty rows to {args.output}.")
        return

    # --- Build strata on remaining
    bins = parse_bins(args.bins)
    labels = build_strata_labels(bins)
    if remaining.empty:
        # No remaining to stratify; just return certainty (already < n)
        out = certainty.sample(frac=1.0, random_state=args.seed)  # shuffle
        out.to_csv(args.output, index=False)
        print(f"[INFO] No remaining repos after certainty. Wrote {len(out)} rows.")
        return

    remaining = remaining.copy()
    remaining["_stratum"] = assign_strata(remaining["stars"], bins, labels)
    # Ensure all labels exist in counts (even if zero)
    counts = remaining["_stratum"].value_counts().reindex(labels, fill_value=0)
    N_h = counts.to_numpy()

    # --- Allocation on remaining
    n_target_remaining = args.n - len(certainty)
    alloc = neyman_sqrt_allocation(N_h=N_h, n_total=n_target_remaining, min_per_stratum=args.min_per_stratum)

    # --- Draw per stratum
    sampled_parts = [certainty]
    for label, need in zip(labels, alloc):
        if need <= 0:
            continue
        pool = remaining[remaining["_stratum"] == label]
        if len(pool) == 0:
            continue
        choose = min(int(need), len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=choose, replace=False)
        sampled_parts.append(pool.loc[idx].drop(columns=["_stratum"]))

    sample_df = pd.concat(sampled_parts, axis=0)

    # Safety: if due to small strata we still fell short, top up randomly from any unused remaining rows
    short = args.n - len(sample_df)
    if short > 0:
        unused = remaining.drop(columns=["_stratum"]).drop(index=sample_df.index.intersection(remaining.index), errors="ignore")
        if not unused.empty:
            take = min(short, len(unused))
            idx = rng.choice(unused.index.to_numpy(), size=take, replace=False)
            sample_df = pd.concat([sample_df, unused.loc[idx]], axis=0)

    # Final safety: cap if somehow we overshot
    if len(sample_df) > args.n:
        sample_df = sample_df.nlargest(args.n, "stars")  # deterministic cap by stars

    # Shuffle for neutrality
    sample_df = sample_df.sample(frac=1.0, random_state=args.seed)

    # Write
    sample_df.to_csv(args.output, index=False)

    # Small console summary
    print(f"Population: {N_pop} | Certainty: {len(certainty)} | "
          f"Strata (remaining): {dict(zip(labels, N_h))} | "
          f"Allocated: {list(map(int, alloc))} | "
          f"Written: {len(sample_df)} -> {args.output}")


if __name__ == "__main__":
    main()
