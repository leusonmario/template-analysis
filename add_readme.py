import csv
import os
import requests
import base64
import time
import pandas as pd

import config

# -------------------- CONFIG --------------------
INPUT_CSV = "repos_mined_all_info_TypeScript.csv"
OUTPUT_CSV = "repos_mined_all_info_with_readme_TypeScript.csv"
TOKEN = config.GITHUB_TOKEN  # <-- your GitHub personal access token
SLEEP_TIME = 0.2   # seconds between requests
# ------------------------------------------------

def fetch_readme(repo_full_name):
    """Fetch README content from GitHub API."""
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            content = r.json().get("content", "")
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        else:
            return None
    except Exception as e:
        print(f"⚠️ Error fetching {repo_full_name}: {e}")
        return None

def already_processed(repo_full_name):
    """Check if this repo is already saved in the output CSV."""
    if not os.path.exists(OUTPUT_CSV):
        return False
    try:
        processed = pd.read_csv(OUTPUT_CSV, usecols=["repo_full_name"])
        return repo_full_name in set(processed["repo_full_name"])
    except Exception:
        return False

def main():
    with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["readme_content"]

        # Create output CSV if it doesn't exist
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

        START_AT_ROW = -1
        current_row = 0

        for row in reader:

            if current_row < START_AT_ROW:
                continue

            repo_full_name = row["repo_full_name"]

            if already_processed(repo_full_name):
                print(f"⏩ Skipping (already processed): {repo_full_name}")
                continue

            print(f"📘 Fetching README for: {repo_full_name} {current_row}")
            readme_content = fetch_readme(repo_full_name)
            row["readme_content"] = readme_content if readme_content else ""

            with open(OUTPUT_CSV, "a", newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writerow(row)

            time.sleep(SLEEP_TIME)
            current_row += 1

    print("\n✅ All repositories processed and saved incrementally!")

if __name__ == "__main__":
    main()