import argparse
import pandas as pd
import subprocess
import shutil
import tempfile
import os
import json
import csv
import time
from tqdm import tqdm
from collections import defaultdict

import sys, os

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

def run(cmd, cwd=None, timeout=None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",   # avoid UnicodeDecodeError
            timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"

def repo_name_from_url(url):
    name = os.path.splitext(os.path.basename(url))[0]
    return name.replace("/", "__")

def detect_package_file(repo_dir):
    for name in ["requirements.txt","pyproject.toml","Pipfile","setup.py"]:
        path = os.path.join(repo_dir, name)
        if os.path.exists(path):
            return name, path
    return None, None

def scan_repo(repo_url, out_dir, tools):
    """Clone to a temp dir, scan, delete temp dir."""
    result = {"repo_url": repo_url, "errors": []}
    repo_id = repo_name_from_url(repo_url)
    result["repo_id"] = repo_id
    repo_tmp = tempfile.mkdtemp(prefix="repo_")
    result_dir = os.path.join(out_dir, repo_id)
    os.makedirs(result_dir, exist_ok=True)

    try:
        # ---- Clone (shallow)
        rc, _, err = run(["git", "clone", "--depth", "1", repo_url, repo_tmp], timeout=180)
        if rc != 0:
            result["errors"].append(f"clone failed: {err.strip()}")
            return result

        # ---- Commit hash
        rc, out, _ = run(["git", "rev-parse", "HEAD"], cwd=repo_tmp)
        result["commit_hash"] = out.strip() if rc == 0 else ""

        # ---- Detect package manager file
        pkg_type, pkg_path = detect_package_file(repo_tmp)
        result["package_file"] = pkg_path or ""
        result["package_type"] = pkg_type or ""

        # ---- Run tools
        if tools.get("pip_audit", True):
            outp = os.path.join(result_dir, "pip_audit.json")
            args = ["pip-audit", "-r", pkg_path, "-f", "json"] if pkg_path else ["pip-audit", "-f", "json"]
            rc, out, err = run(args, cwd=repo_tmp, timeout=120)
            with open(outp, "w", encoding="utf-8") as f:
                f.write(out or err)
            result["pip_audit"] = outp

        if tools.get("bandit", True):
            outp = os.path.join(result_dir, "bandit.json")
            rc, out, err = run(["bandit", "-r", repo_tmp, "-f", "json", "-o", outp], timeout=180)
            result["bandit"] = outp if os.path.exists(outp) else ""
            if rc != 0 and not os.path.exists(outp):
                result["errors"].append(f"bandit error: {err.strip()}")

        if tools.get("semgrep", True):
            outp = os.path.join(result_dir, "semgrep.json")
            rc, out, err = run(["semgrep", "--config", "p/ci", "--json", "--output", outp, repo_tmp], timeout=240)
            result["semgrep"] = outp if os.path.exists(outp) else ""
            if rc != 0 and not os.path.exists(outp):
                result["errors"].append(f"semgrep error: {err.strip()}")

    except Exception as e:
        result["errors"].append(str(e))
    finally:
        # ---- Always cleanup (delete clone)
        try:
            shutil.rmtree(repo_tmp, ignore_errors=True)
        except Exception as e:
            result["errors"].append(f"cleanup failed: {e}")

    return result


def main(args):
    df = pd.read_csv(args.input)
    repo_col = next((c for c in ["original_link_repository", "repo_url", "clone_url"] if c in df.columns), None)
    if not repo_col:
        print("❌ Could not find repo URL column.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.csv")

    # ----------------------------------------------------------------------
    # Load already processed repos (to skip them)
    processed_repos = set()
    if os.path.exists(summary_path):
        try:
            existing_df = pd.read_csv(summary_path)
            processed_repos = set(existing_df["repo_url"].dropna().astype(str))
            print(f"🧩 Found {len(processed_repos)} previously analyzed repositories. They will be skipped.")
        except Exception as e:
            print(f"⚠️ Could not read summary file ({e}). Proceeding without skip list.")
    else:
        # If no summary yet, create header
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "repo_url", "repo_id", "commit_hash", "package_type", "package_file",
                "pip_audit", "bandit", "semgrep", "errors"
            ])
            writer.writeheader()

    tools = {"pip_audit": True, "bandit": True, "semgrep": True}

    # ----------------------------------------------------------------------
    # Process new repos only
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning repos"):
        repo_url = str(row[repo_col]).strip()

        # Skip if already analyzed
        if repo_url in processed_repos:
            tqdm.write(f"⏭️  Skipping already analyzed repo: {repo_url}")
            continue

        res = scan_repo(repo_url, args.outdir, tools)

        # Append result immediately (safe incremental writes)
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "repo_url", "repo_id", "commit_hash", "package_type", "package_file",
                "pip_audit", "bandit", "semgrep", "errors"
            ])
            writer.writerow({
                "repo_url": res.get("repo_url"),
                "repo_id": res.get("repo_id"),
                "commit_hash": res.get("commit_hash", ""),
                "package_type": res.get("package_type"),
                "package_file": res.get("package_file"),
                "pip_audit": res.get("pip_audit", ""),
                "bandit": res.get("bandit", ""),
                "semgrep": res.get("semgrep", ""),
                "errors": "; ".join(res.get("errors", []))
            })

        # Add to processed set (avoid duplicates during same run)
        processed_repos.add(repo_url)

        time.sleep(args.delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with repo URLs")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
