import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# input files
human_df = pd.read_csv("manual_analysis.csv")
llm_df = pd.read_csv("results.csv")

# checking project columns in input files
project_col = None
for col in human_df.columns:
    if col.strip().lower() in ["project", "project_url", "repo", "repository"]:
        project_col = col
        break

if project_col is None:
    raise ValueError(
        f"Could not find project column. Available columns: {list(human_df.columns)}"
    )

human_df = human_df.rename(columns={project_col: "project_url"})

# getting guidelines and pitfalls based on their ids
item_cols = [
    c for c in human_df.columns
    if (c.startswith("G") or c.startswith("P"))
]

human_long = human_df.melt(
    id_vars=["project_url"],
    value_vars=item_cols,
    var_name="item_id",
    value_name="human_label"
)

# normalize yes/no
human_long["human_label"] = (
    human_long["human_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

mapping = {"yes": 1, "no": 0}
human_long["human_bin"] = human_long["human_label"].map(mapping)

# llm normalization
llm_df["llm_label"] = (
    llm_df["adherence"]
    .astype(str)
    .str.strip()
    .str.lower()
)

llm_df["llm_bin"] = llm_df["llm_label"].map(mapping)

llm_df = llm_df[["project_url", "item_id", "llm_bin"]]

df = pd.merge(
    human_long,
    llm_df,
    on=["project_url", "item_id"],
    how="inner"
).dropna(subset=["human_bin", "llm_bin"])

print(f"Total aligned comparisons: {len(df)}")

overall_kappa = cohen_kappa_score(df["human_bin"], df["llm_bin"])
overall_accuracy = accuracy_score(df["human_bin"], df["llm_bin"])

overall_precision = precision_score(
    df["human_bin"],
    df["llm_bin"],
    average="macro",
    zero_division=0
)

overall_recall = recall_score(
    df["human_bin"],
    df["llm_bin"],
    average="macro",
    zero_division=0
)

overall_f1 = f1_score(
    df["human_bin"],
    df["llm_bin"],
    average="macro",
    zero_division=0
)

print("\n=== Overall Metrics (Macro Average) ===")
print(f"Cohen's Kappa: {overall_kappa:.4f}")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Macro Precision: {overall_precision:.4f}")
print(f"Macro Recall: {overall_recall:.4f}")
print(f"Macro F1: {overall_f1:.4f}")

print("\n=== Agreement by Item ===")

for item in sorted(df["item_id"].unique()):
    subset = df[df["item_id"] == item]

    acc = accuracy_score(subset["human_bin"], subset["llm_bin"])
    prec = precision_score(subset["human_bin"], subset["llm_bin"], average="macro", zero_division=0)
    rec = recall_score(subset["human_bin"], subset["llm_bin"], average="macro", zero_division=0)
    f1 = f1_score(subset["human_bin"], subset["llm_bin"], average="macro", zero_division=0)

    if (
        len(subset) >= 2
        and subset["human_bin"].nunique() > 1
        and subset["llm_bin"].nunique() > 1
    ):
        kappa = cohen_kappa_score(subset["human_bin"], subset["llm_bin"])
        print(
            f"{item} (n={len(subset)}): "
            f"Kappa={kappa:.4f}, "
            f"Acc={acc:.4f}, "
            f"Prec={prec:.4f}, "
            f"Rec={rec:.4f}, "
            f"F1={f1:.4f}"
        )
    else:
        print(
            f"{item} (n={len(subset)}): "
            f"Kappa=NA, "
            f"Acc={acc:.4f}, "
            f"Prec={prec:.4f}, "
            f"Rec={rec:.4f}, "
            f"F1={f1:.4f}"
        )