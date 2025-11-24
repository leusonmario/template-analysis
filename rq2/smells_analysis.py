#!/usr/bin/env python3
import os
import subprocess
import shutil
import requests
import time
import pandas as pd
import json
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
SONAR_URL = "http://localhost:9000"
SONAR_TOKEN = "sqa_06fff9c4d82203aacccf6942a045a836a52a1b39"  # <-- Replace this
REPO_DIR = "./temp_repos"
OUTPUT_CSV = "vulnerability_results_Python.csv"
INPUT_CSV = "all_templates_update_metrics_Python.csv"  # must have 'original_link_repository' column
JSON_OUTPUT_DIR = "./vulnerability_details"
METRICS = ["vulnerabilities", "security_hotspots", "code_smells", "bugs"]
SONAR_SCANNER = r"C:\Users\leu_m\Downloads\sonar-scanner\sonar-scanner\bin\sonar-scanner.bat"

os.makedirs(REPO_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def run_cmd(cmd):
    """Run shell command and stream output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    return result.returncode == 0


def get_vulnerability_details(project_key, repo_name):
    """Fetch detailed vulnerability information and severity counts."""
    url = f"{SONAR_URL}/api/issues/search"
    params = {
        "componentKeys": project_key,
        "types": "VULNERABILITY",
        "ps": 500  # max per page
    }
    all_issues = []
    page = 1

    while True:
        params["p"] = page
        r = requests.get(url, auth=(SONAR_TOKEN, ""), params=params)
        if r.status_code != 200:
            print(f"⚠️ Failed to get issues for {repo_name}: {r.text}")
            break
        data = r.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        if len(issues) < 500:
            break
        page += 1

    # Save raw JSON for detailed analysis
    if all_issues:
        json_path = os.path.join(JSON_OUTPUT_DIR, f"{repo_name}_vulnerabilities.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_issues, f, indent=2)
        print(f"💾 Saved {len(all_issues)} vulnerability issues to {json_path}")

    # Count severities
    severities = [issue.get("severity", "UNKNOWN") for issue in all_issues]
    counts = Counter(severities)
    return dict(counts)


def analyze_repo(git_url):
    """Clone, scan, get results, and delete repo."""
    repo_name = git_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(REPO_DIR, repo_name)

    # Skip if already analyzed
    if os.path.exists(os.path.join(JSON_OUTPUT_DIR, f"{repo_name}_vulnerabilities.json")):
        print(f"⏩ Skipping {repo_name} (already analyzed)")
        return None

    # Clone
    if not run_cmd(["git", "clone", "--depth", "1", git_url, repo_path]):
        print(f"⚠️ Failed to clone {git_url}")
        return None

    # Run sonar-scanner
    project_key = repo_name.replace("/", "_")
    repo_path_abs = os.path.abspath(repo_path).replace("\\", "/")

    sonar_cmd = [
        SONAR_SCANNER,
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.sources={repo_path_abs}",
        "-Dsonar.scm.exclusions.disabled=true",
        "-Dsonar.inclusions=**/*",
        f"-Dsonar.host.url={SONAR_URL}",
        f"-Dsonar.token={SONAR_TOKEN}",
    ]

    print(f"🔍 Analyzing {repo_name}...")
    run_cmd(sonar_cmd)

    # Give SonarQube a few seconds to finalize analysis
    time.sleep(10)

    # Query aggregate metrics
    metrics_str = ",".join(METRICS)
    api_url = f"{SONAR_URL}/api/measures/component"
    params = {"component": project_key, "metricKeys": metrics_str}
    r = requests.get(api_url, auth=(SONAR_TOKEN, ""), params=params)

    if r.status_code != 200:
        print(f"⚠️ Failed to get metrics for {repo_name}: {r.text}")
        result = None
    else:
        data = r.json()
        measures = {m["metric"]: m["value"] for m in data.get("component", {}).get("measures", [])}
        result = {"repo": repo_name, **measures}

        # Add severity breakdown
        severity_counts = get_vulnerability_details(project_key, repo_name)
        for sev in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]:
            result[f"vuln_{sev.lower()}"] = severity_counts.get(sev, 0)

    # Clean up
    shutil.rmtree(repo_path, ignore_errors=True)
    print(f"🧹 Deleted local clone of {repo_name}")
    return result


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    repos = df["original_link_repository"].dropna().tolist()

    results = []
    for i, repo_url in enumerate(repos, 1):
        print(f"\n=== ({i}/{len(repos)}) {repo_url} ===")
        info = analyze_repo(repo_url)
        if info:
            results.append(info)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Saved partial results to {OUTPUT_CSV}")

    print("\n🎉 Done! All repositories processed.")

def delete_project(project_key):
    """
    Delete a SonarQube project cleanly, with full diagnostics.
    Requires an admin-level token (stored in SONAR_TOKEN).
    """

    token = SONAR_TOKEN
    base_url = "http://localhost:9000"

    # 1. Check if the project exists
    check_url = f"{base_url}/api/projects/search?projects={project_key}"
    check = requests.get(check_url, auth=(token, ""))

    if check.status_code != 200:
        print(f"❌ Could not verify project existence: {check.text}")
        return

    project_count = check.json().get("paging", {}).get("total", 0)
    if project_count == 0:
        print(f"⚠️ Project '{project_key}' does not exist → nothing to delete.")
        return

    # 2. Try deletion
    delete_url = f"{base_url}/api/projects/delete?project={project_key}"
    r = requests.post(delete_url, auth=(token, ""))

    # 3. Handle all cases
    if r.status_code == 204:
        print(f"🗑️ Deleted project: {project_key}")
    elif r.status_code == 403:
        print("❌ Insufficient privileges to delete the project.")
        print("👉 This means SONAR_TOKEN does NOT belong to an admin user.")
        print("   Log in at http://localhost:9000 as admin and generate a new token.")
        print("   Token must have: Administer System + Administer Projects.")
    elif r.status_code == 404:
        print(f"⚠️ Project '{project_key}' not found (maybe already deleted?)")
    else:
        print(f"❌ Failed to delete {project_key}: {r.text} (code: {r.status_code})")


if __name__ == "__main__":
    main()
    #delete_project("sceptre-hook-template")
