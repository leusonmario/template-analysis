import re
import csv
import argparse
from pathlib import Path

import pandas as pd


# ==========================
# Exclusion detection rules
# ==========================

EXCLUDED_KEYWORDS = [
    # Generic bot indicators
    "bot",
    "[bot]",
    "-bot",
    "_bot",

    # Dependency/update bots
    "dependabot",
    "dependabot-preview",
    "renovate",
    "renovatebot",
    "greenkeeper",
    "snyk",
    "whitesource",

    # CI/CD and automation services
    "github-actions",
    "github-actions[bot]",
    "github action",
    "actions-user",
    "actions-template-sync",
    "web-flow",
    "bors",
    "mergify",
    "codecov",
    "coveralls",
    "travis",
    "circleci",
    "appveyor",
    "azure-pipelines",
    "semantic-release",
    "release-please",
    "allcontributors",
    "stale",
    "pre-commit-ci",
    "imgbot",

    # Other service/no-reply identities observed
    "weblate",
    "gitee",
    "claude",
    "fly.io",
    "octopus deploy",
]

EXCLUDED_EMAIL_PATTERNS = [
    # Generic no-reply variants
    r".*noreply.*",
    r".*no-reply.*",
    r".*donotreply.*",
    r".*do-not-reply.*",

    # GitHub no-reply/private emails
    r"noreply@github\.com",
    r".*@users\.noreply\.github\.com\.?$",
    r".*@user\.noreply\.github\.com\.?$",
    r".*@noreply\.users\.github\.com\.?$",

    # GitLab/Gitee/Weblate private emails
    r".*@users\.noreply\.gitlab\.com\.?$",
    r".*@user\.noreply\.gitee\.com\.?$",
    r".*@users\.noreply\.hosted\.weblate\.org\.?$",

    # Known automation/service emails
    r".*\[bot\]@users\.noreply\.github\.com\.?$",
    r".*bot.*@users\.noreply\.github\.com\.?$",
    r"dependabot.*",
    r"renovate.*",
    r"github-actions.*",
    r".*@actions-template-sync\.noreply\.github\.com\.?$",

    # Usually service accounts, not contactable contributors
    r".*@github\.com",
]

EXCLUDED_NAME_PATTERNS = [
    r".*\[bot\].*",
    r".*\bbot\b.*",
    r"dependabot.*",
    r"renovate.*",
    r"github-actions.*",
    r".*github action.*",
    r".*weblate.*",
    r".*gitee.*",
    r".*claude.*",
]


# ==========================
# Helper functions
# ==========================

def max_commit_count(values):
    """
    Preserve the maximum commit_count when duplicate rows are merged.

    Usually, duplicate rows for the same language/project/email should have
    the same commit_count. We use max as a safe choice.
    """

    numeric_values = pd.to_numeric(values, errors="coerce").dropna()

    if numeric_values.empty:
        return 0

    return int(numeric_values.max())

def normalize_text(value):
    """
    Normalize text for matching.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def is_excluded_row(row):
    """
    Return True if a row should be excluded.

    We exclude:
    1. Bot accounts
    2. Automation/service accounts
    3. GitHub/GitLab/Gitee/Weblate private noreply emails
    4. Generic no-reply emails

    Important:
    Some noreply emails may belong to real humans using private email settings.
    We exclude them here because they are not contactable email addresses.
    """

    email = normalize_text(row.get("email", ""))
    name = normalize_text(row.get("name", ""))

    combined = f"{email} {name}"

    # Direct keyword matching over email + name
    for keyword in EXCLUDED_KEYWORDS:
        if keyword in combined:
            return True

    # Regex matching over email
    for pattern in EXCLUDED_EMAIL_PATTERNS:
        if re.fullmatch(pattern, email):
            return True

    # Regex matching over name
    for pattern in EXCLUDED_NAME_PATTERNS:
        if re.fullmatch(pattern, name):
            return True

    return False


def first_non_empty(values):
    """
    Return the first non-empty value.
    """

    for value in values:
        value = str(value).strip()

        if value:
            return value

    return ""


def merge_roles(values):
    """
    Merge duplicated git_roles values.

    Example:
    author
    committer
    author,committer

    becomes:
    author,committer
    """

    roles = set()

    for value in values:
        value = str(value).strip()

        if not value:
            continue

        for role in value.split(","):
            role = role.strip()

            if role:
                roles.add(role)

    return ",".join(sorted(roles))


def clean_people_csv(input_file, output_file, removed_file):
    """
    Read a CSV, remove excluded rows, remove duplicates, and save results.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    df = pd.read_csv(input_path)

    required_columns = {
        "language",
        "original_link_repository",
        "email",
        "name",
        "git_roles",
        "commit_count"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Original rows: {len(df)}")

    # Normalize relevant columns
    df["email"] = df["email"].fillna("").astype(str).str.strip().str.lower()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["git_roles"] = df["git_roles"].fillna("").astype(str).str.strip()
    df["language"] = df["language"].fillna("").astype(str).str.strip()
    df["original_link_repository"] = (
        df["original_link_repository"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df["commit_count"] = pd.to_numeric(
        df["commit_count"],
        errors="coerce"
    ).fillna(0).astype(int)

    # Remove rows without email
    df = df[df["email"] != ""].copy()

    print(f"Rows after removing empty emails: {len(df)}")

    # Identify excluded rows
    excluded_mask = df.apply(is_excluded_row, axis=1)

    removed_rows = df[excluded_mask].copy()
    cleaned = df[~excluded_mask].copy()

    print(f"Rows identified as bots/no-reply/service identities: {len(removed_rows)}")
    print(f"Rows after filtering: {len(cleaned)}")

    # Merge duplicate language-project-email pairs
    cleaned = (
        cleaned
        .groupby(
            ["language", "original_link_repository", "email"],
            as_index=False
        )
        .agg({
            "name": first_non_empty,
            "git_roles": merge_roles,
            "commit_count": max_commit_count
        })
    )

    # Also clean duplicates in removed rows for easier inspection
    removed_rows = (
        removed_rows
        .groupby(
            ["language", "original_link_repository", "email"],
            as_index=False
        )
        .agg({
            "name": first_non_empty,
            "git_roles": merge_roles,
            "commit_count": max_commit_count
        })
    )

    cleaned = cleaned.sort_values(
        by=["language", "original_link_repository", "commit_count", "email"]
    )

    removed_rows = removed_rows.sort_values(
        by=["language", "original_link_repository", "commit_count", "email"]
    )

    cleaned.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    removed_rows.to_csv(removed_file, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nSaved cleaned file to: {output_file}")
    print(f"Saved removed rows to: {removed_file}")
    print(f"Final cleaned rows: {len(cleaned)}")
    print(f"Removed rows: {len(removed_rows)}")


# ==========================
# Main execution
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Filter bot, automation, service, and no-reply identities from project people CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file, e.g., project_committers_emails_minimal.csv"
    )

    parser.add_argument(
        "--output",
        default="project_people_contactable_cleaned_minimal.csv",
        help="Output cleaned CSV file"
    )

    parser.add_argument(
        "--removed",
        default="project_people_removed_bots_and_noreply_minimal.csv",
        help="CSV file containing removed bot/no-reply/service rows"
    )

    args = parser.parse_args()

    clean_people_csv(
        input_file=args.input,
        output_file=args.output,
        removed_file=args.removed
    )


if __name__ == "__main__":
    main()