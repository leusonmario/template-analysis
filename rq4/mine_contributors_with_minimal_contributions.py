import os
import re
import csv
import shutil
import tempfile
import subprocess
from pathlib import Path

import pandas as pd


# ==========================
# Configuration
# ==========================

INPUT_FOLDER = "../data_collection/"  # folder containing your CSV files

OUTPUT_FILE = "project_committers_emails_minimal.csv"
ERROR_FILE = "failed_repositories_minimal.csv"
PROCESSED_FILE = "processed_repositories_minimal.csv"

REPOSITORY_COLUMN = "original_link_repository"

MIN_COMMITS = 5


# ==========================
# Helper functions
# ==========================

def normalize_github_url(url):
    """
    Normalize GitHub repository URLs to HTTPS clone URLs.
    """

    if pd.isna(url):
        return None

    url = str(url).strip()

    if not url:
        return None

    url = url.rstrip("/")
    url = url.split("#")[0].split("?")[0]

    ssh_match = re.match(r"git@github\.com:(.+?)/(.+?)(\.git)?$", url)

    if ssh_match:
        owner = ssh_match.group(1)
        repo = ssh_match.group(2).replace(".git", "")
        return f"https://github.com/{owner}/{repo}.git"

    if url.startswith("github.com/"):
        url = "https://" + url

    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)

    if not match:
        return None

    owner = match.group(1)
    repo = match.group(2).replace(".git", "")

    return f"https://github.com/{owner}/{repo}.git"


def infer_language_from_filename(csv_file):
    """
    Infer the programming language from the CSV filename.
    """

    filename = csv_file.stem.lower()

    language_map = {
        "typescript": "TypeScript",
        "javascript": "JavaScript",
        "python": "Python",
        "java": "Java",
        "csharp": "C#",
        "c_sharp": "C#",
        "c#": "C#",
        "cpp": "C++",
        "c++": "C++",
        "go": "Go",
        "ruby": "Ruby",
        "php": "PHP",
        "rust": "Rust",
        "kotlin": "Kotlin",
        "swift": "Swift",
        "scala": "Scala",
    }

    for key, language in language_map.items():
        if key in filename:
            return language

    return "Unknown"


def collect_repositories(input_folder):
    """
    Read all CSV files in the input folder and collect unique repository URLs.

    Returns:
    {
        repo_url: set(languages)
    }
    """

    repositories = {}
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    for csv_file in csv_files:
        language = infer_language_from_filename(csv_file)

        print(f"Reading: {csv_file} | language={language}")

        try:
            df = pd.read_csv(csv_file)

            if REPOSITORY_COLUMN not in df.columns:
                print(f"  Skipping {csv_file}: column '{REPOSITORY_COLUMN}' not found")
                continue

            for value in df[REPOSITORY_COLUMN].dropna():
                normalized_url = normalize_github_url(value)

                if normalized_url:
                    if normalized_url not in repositories:
                        repositories[normalized_url] = set()

                    repositories[normalized_url].add(language)

        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")

    return repositories


def run_command(command, cwd=None):
    """
    Run a command and return stdout.
    Raises an exception if the command fails.
    """

    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    return result.stdout


def clone_repository(repo_url, target_dir):
    """
    Clone a repository as a mirror.
    """

    command = [
        "git",
        "clone",
        "--mirror",
        "--filter=blob:none",
        repo_url,
        target_dir
    ]

    run_command(command)


def get_unique_people_from_mirror_repo(repo_dir):
    """
    Extract unique people from all commits and count how many commits
    are associated with each person.

    We count a commit for a person if they appear as author or committer.

    If the same person is both author and committer for the same commit,
    the commit is counted only once for that person.

    %H = commit hash
    %aE = author email
    %aN = author name
    %cE = committer email
    %cN = committer name
    """

    command = [
        "git",
        "--git-dir",
        repo_dir,
        "log",
        "--all",
        "--format=%H\t%aE\t%aN\t%cE\t%cN"
    ]

    output = run_command(command)

    people = {}

    for line in output.splitlines():
        parts = line.strip().split("\t")

        if len(parts) != 5:
            continue

        commit_hash, author_email, author_name, committer_email, committer_name = parts

        author_email = author_email.strip().lower()
        author_name = author_name.strip()

        committer_email = committer_email.strip().lower()
        committer_name = committer_name.strip()

        people_in_this_commit = {}

        if author_email:
            people_in_this_commit[author_email] = {
                "name": author_name,
                "role": "author"
            }

        if committer_email:
            if committer_email not in people_in_this_commit:
                people_in_this_commit[committer_email] = {
                    "name": committer_name,
                    "role": "committer"
                }
            else:
                people_in_this_commit[committer_email]["role"] = "author,committer"

        for email, info in people_in_this_commit.items():
            if email not in people:
                people[email] = {
                    "email": email,
                    "name": info["name"],
                    "roles": set(),
                    "commit_hashes": set()
                }

            for role in info["role"].split(","):
                people[email]["roles"].add(role)

            people[email]["commit_hashes"].add(commit_hash)

    for email in people:
        people[email]["commit_count"] = len(people[email]["commit_hashes"])
        del people[email]["commit_hashes"]

    return people


def append_rows_to_csv(rows, output_file):
    """
    Append rows to a CSV file.
    Creates the file with a header if it does not exist yet.
    """

    if not rows:
        return

    file_exists = os.path.exists(output_file)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not file_exists,
        quoting=csv.QUOTE_MINIMAL
    )


def mark_repository_as_processed(repo_url, language, status):
    """
    Save processed repository status.
    """

    row = {
        "language": language,
        "original_link_repository": repo_url,
        "status": status
    }

    append_rows_to_csv([row], PROCESSED_FILE)


def load_processed_repositories():
    """
    Load repository-language pairs already processed successfully.
    """

    if not os.path.exists(PROCESSED_FILE):
        return set()

    try:
        df = pd.read_csv(PROCESSED_FILE)

        required_columns = {
            "language",
            "original_link_repository",
            "status"
        }

        if not required_columns.issubset(df.columns):
            return set()

        successful = df[df["status"] == "success"]

        return set(
            zip(
                successful["original_link_repository"].dropna().astype(str),
                successful["language"].dropna().astype(str)
            )
        )

    except Exception:
        return set()


def process_repositories(repositories):
    """
    Clone each repository temporarily, collect unique people, save only people
    with at least MIN_COMMITS commits, then delete the cloned repository.
    """

    processed_repositories = load_processed_repositories()

    repo_language_pairs = []

    for repo_url, languages in repositories.items():
        for language in languages:
            if (repo_url, language) not in processed_repositories:
                repo_language_pairs.append((repo_url, language))

    print(f"Repository-language pairs already processed: {len(processed_repositories)}")
    print(f"Repository-language pairs remaining: {len(repo_language_pairs)}")

    total = len(repo_language_pairs)

    for i, (repo_url, language) in enumerate(repo_language_pairs, start=1):
        print(f"\n[{i}/{total}] Processing: {repo_url} | language={language}")

        temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
        repo_name = repo_url.removesuffix(".git").split("/")[-1]
        clone_path = os.path.join(temp_dir, f"{repo_name}.git")

        try:
            clone_repository(repo_url, clone_path)

            people = get_unique_people_from_mirror_repo(clone_path)

            print(f"  Found {len(people)} unique people before commit-count filtering")

            repo_rows = []

            for email, info in people.items():
                commit_count = info["commit_count"]

                if commit_count < MIN_COMMITS:
                    continue

                repo_rows.append({
                    "language": language,
                    "original_link_repository": repo_url,
                    "email": info["email"],
                    "name": info["name"],
                    "git_roles": ",".join(sorted(info["roles"])),
                    "commit_count": commit_count
                })

            append_rows_to_csv(repo_rows, OUTPUT_FILE)

            mark_repository_as_processed(repo_url, language, "success")

            print(f"  Saved {len(repo_rows)} people with at least {MIN_COMMITS} commits")

        except Exception as e:
            print(f"  Failed: {e}")

            failure_row = {
                "language": language,
                "original_link_repository": repo_url,
                "error": str(e)
            }

            append_rows_to_csv([failure_row], ERROR_FILE)

            mark_repository_as_processed(repo_url, language, "failed")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def clean_output_file():
    """
    Removes duplicate language-project-email pairs and sorts the output file.
    """

    if not os.path.exists(OUTPUT_FILE):
        return

    df = pd.read_csv(OUTPUT_FILE)

    if df.empty:
        return

    df = df.drop_duplicates(
        subset=["language", "original_link_repository", "email"]
    )

    df = df.sort_values(
        by=["language", "original_link_repository", "commit_count", "email"],
        ascending=[True, True, False, True]
    )

    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nCleaned output file: {OUTPUT_FILE}")
    print(f"Total unique language-project-email pairs with at least {MIN_COMMITS} commits: {len(df)}")


# ==========================
# Main execution
# ==========================

def main():
    repositories = collect_repositories(INPUT_FOLDER)

    total_repo_language_pairs = sum(len(languages) for languages in repositories.values())

    print(f"\nTotal unique repositories found: {len(repositories)}")
    print(f"Total repository-language pairs found: {total_repo_language_pairs}")
    print(f"Minimum commits per practitioner: {MIN_COMMITS}")

    process_repositories(repositories)

    clean_output_file()

    print("\nDone.")


if __name__ == "__main__":
    main()