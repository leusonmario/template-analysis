#!/usr/bin/env python3
"""
gh_mine_rest.py  — streaming version (writes each row as soon as it's processed)

Input CSV columns (required):
    id,username,project_name,original_link_repository,stars,forks,created_at,updated_at,language
"""

import argparse
import csv
import os
import sys
import time
import re
import base64
from typing import Dict, Optional, Tuple, List

import requests
import config

# ----------------------- CONFIG -----------------------
GITHUB_TOKEN = config.GITHUB_TOKEN
BASE = "https://api.github.com"
UA = "RepoMiner/1.0 (+https://github.com)"
SLEEP_BETWEEN_CALLS = 0.1
BACKOFF_MIN = 5.0
BACKOFF_MAX = 60.0
TIMEOUT = 30
# ------------------------------------------------------

def parse_owner_repo(original_link: str, username: str, project_name: str) -> Optional[Tuple[str, str]]:
    if original_link:
        m = re.search(r"github\.com[:/]+(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git|/)?$", original_link.strip())
        if m:
            return m.group("owner"), m.group("repo")
    if username and project_name:
        return username.strip(), project_name.strip()
    return None

def make_session() -> requests.Session:
    s = requests.Session()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": UA}
    if GITHUB_TOKEN and GITHUB_TOKEN != "ghp_your_token_here":
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    s.headers.update(headers)
    return s

def get_json(session: requests.Session, url: str, params: Optional[Dict]=None):
    while True:
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException:
            time.sleep(2.0)
            continue

        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = r.headers.get("X-RateLimit-Reset")
            if reset:
                try:
                    wait = max(0.0, float(reset) - time.time())
                except Exception:
                    wait = BACKOFF_MIN
                time.sleep(min(BACKOFF_MAX, max(BACKOFF_MIN, wait)))
            else:
                time.sleep(BACKOFF_MIN)
            continue

        try:
            data = r.json() if r.content else {}
        except ValueError:
            data = {}
        return r.status_code, data, r.headers

def get_repo(session, owner, repo) -> Dict:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}")
    return data if code == 200 else {}

def get_topics(session, owner, repo) -> List[str]:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/topics")
    if code != 200 or not isinstance(data, dict):
        return []
    names = data.get("names", [])
    if isinstance(names, list):
        if all(isinstance(x, str) for x in names):
            return names
        return [(x.get("name") if isinstance(x, dict) else str(x)) for x in names if (isinstance(x, dict) and x.get("name")) or isinstance(x, (str, int, float))]
    topics = data.get("topics")
    if isinstance(topics, list):
        return [(x.get("name") if isinstance(x, dict) else str(x)) for x in topics if (isinstance(x, dict) and x.get("name")) or isinstance(x, (str, int, float))]
    return []

def get_count_via_pagination(session, url: str, params: Optional[Dict]=None) -> int:
    p = dict(params or {})
    p["per_page"] = 1
    code, data, headers = get_json(session, url, params=p)
    if code != 200:
        return 0
    link = headers.get("Link") or headers.get("link")
    if link:
        m = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    if isinstance(data, list):
        return len(data)
    return 0

def get_releases_info(session, owner, repo) -> Dict:
    count = get_count_via_pagination(session, f"{BASE}/repos/{owner}/{repo}/releases")
    latest = {}
    if count > 0:
        code, items, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/releases/latest")
        if code == 200 and isinstance(items, dict):
            latest = {
                "latest_release_tag": items.get("tag_name"),
                "latest_release_published_at": items.get("published_at"),
                "latest_release_assets_count": len(items.get("assets", []) or []),
            }
    return {"releases_count": count, **latest}

def get_actions_workflows(session, owner, repo) -> Dict:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/actions/workflows")
    if code == 200 and isinstance(data, dict):
        wfs = data.get("workflows", []) or []
        names = [w.get("name") for w in wfs if isinstance(w, dict) and w.get("name")]
        return {"actions_workflows_count": len(wfs), "actions_workflows": "; ".join(names)}
    return {"actions_workflows_count": 0, "actions_workflows": ""}

def get_labels_count(session, owner, repo) -> int:
    return get_count_via_pagination(session, f"{BASE}/repos/{owner}/{repo}/labels")

def get_pages_enabled(session, owner, repo) -> Dict:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/pages")
    if code == 200 and isinstance(data, dict):
        return {
            "has_pages": True,
            "pages_status": data.get("status"),
            "pages_cname": data.get("cname"),
            "pages_source": (data.get("source") or {}).get("branch"),
        }
    return {"has_pages": False, "pages_status": "", "pages_cname": "", "pages_source": ""}

def get_community_profile(session, owner, repo) -> Dict:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/community/profile")
    if code != 200 or not isinstance(data, dict):
        return {
            "community_health_percentage": "",
            "has_readme": "",
            "has_contributing": "",
            "has_code_of_conduct": "",
            "code_of_conduct_key": "",
            "has_support": "",
            "has_issue_template": "",
            "has_pr_template": "",
        }
    files = data.get("files", {}) or {}
    coc = files.get("code_of_conduct", {}) or {}
    return {
        "community_health_percentage": data.get("health_percentage", ""),
        "has_readme": bool(files.get("readme")),
        "has_contributing": bool(files.get("contributing")),
        "has_code_of_conduct": bool(files.get("code_of_conduct")),
        "code_of_conduct_key": coc.get("key") or coc.get("name") or "",
        "has_support": bool(files.get("support")),
        "has_issue_template": bool(files.get("issue_template")),
        "has_pr_template": bool(files.get("pull_request_template")),
    }

def get_funding_info(session, owner, repo) -> Dict:
    code, data, _ = get_json(session, f"{BASE}/repos/{owner}/{repo}/contents/.github/FUNDING.yml")
    if code == 200 and isinstance(data, dict):
        content = data.get("content")
        if content:
            try:
                raw = base64.b64decode(content).decode("utf-8", errors="ignore")
                keys = []
                for line in raw.splitlines():
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        k = line.split(":",1)[0].strip()
                        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                            keys.append(k)
                keys = sorted(set(keys))
                return {"has_funding_file": True, "funding_platforms": "; ".join(keys)}
            except Exception:
                return {"has_funding_file": True, "funding_platforms": ""}
        return {"has_funding_file": True, "funding_platforms": ""}
    return {"has_funding_file": False, "funding_platforms": ""}

def enrich_one(session: requests.Session, row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    owner_repo = parse_owner_repo(row.get("original_link_repository",""), row.get("username",""), row.get("project_name",""))
    if not owner_repo:
        out.update({"error": "Could not resolve owner/repo from inputs"})
        return out
    owner, repo = owner_repo

    repo_data = get_repo(session, owner, repo)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if not repo_data:
        out.update({"error": "Repository not found or inaccessible"})
        return out

    out.update({
        "repo_full_name": repo_data.get("full_name",""),
        "repo_private": repo_data.get("private", False),
        "repo_fork": repo_data.get("fork", False),
        "repo_is_template": repo_data.get("is_template", False),
        "repo_archived": repo_data.get("archived", False),
        "repo_disabled": repo_data.get("disabled", False),
        "repo_description": repo_data.get("description") or "",
        "default_branch": repo_data.get("default_branch") or "",
        "homepage": repo_data.get("homepage") or "",
        "open_issues_count": repo_data.get("open_issues_count") or 0,
        "owner_type": (repo_data.get("owner") or {}).get("type",""),
        "owner_login": (repo_data.get("owner") or {}).get("login",""),
        "license_spdx_id": (repo_data.get("license") or {}).get("spdx_id") if repo_data.get("license") else "",
        "license_name": (repo_data.get("license") or {}).get("name") if repo_data.get("license") else "",
        "network_count": repo_data.get("network_count", ""),
        "subscribers_count": repo_data.get("subscribers_count", ""),
        "watchers_count": repo_data.get("watchers_count", ""),
        "has_wiki": repo_data.get("has_wiki", ""),
        "has_discussions": repo_data.get("has_discussions", ""),
    })

    topics = get_topics(session, owner, repo)
    time.sleep(SLEEP_BETWEEN_CALLS)
    out["topics"] = "; ".join(topics) if topics else ""

    out.update(get_releases_info(session, owner, repo))
    time.sleep(SLEEP_BETWEEN_CALLS)

    out.update(get_actions_workflows(session, owner, repo))
    time.sleep(SLEEP_BETWEEN_CALLS)

    out["labels_count"] = get_labels_count(session, owner, repo)
    time.sleep(SLEEP_BETWEEN_CALLS)

    out.update(get_pages_enabled(session, owner, repo))
    time.sleep(SLEEP_BETWEEN_CALLS)

    out.update(get_community_profile(session, owner, repo))
    time.sleep(SLEEP_BETWEEN_CALLS)

    out.update(get_funding_info(session, owner, repo))
    return out

def main():
    # If you want to stick with your per-language batch, keep this list:
    required_cols = ["id","username","project_name","original_link_repository","stars","forks","created_at","updated_at","language"]

    languages = ["JavaScript"]
    session = make_session()

    # Define a STABLE header upfront so we can stream rows safely
    input_cols = required_cols.copy()
    extra_cols = [
        "repo_full_name","repo_private","repo_fork","repo_is_template","repo_archived","repo_disabled",
        "repo_description","default_branch","homepage","open_issues_count","owner_type","owner_login",
        "license_spdx_id","license_name","network_count","subscribers_count","watchers_count",
        "has_wiki","has_discussions","topics",
        "releases_count","latest_release_tag","latest_release_published_at","latest_release_assets_count",
        "actions_workflows_count","actions_workflows",
        "labels_count","has_pages","pages_status","pages_cname","pages_source",
        "community_health_percentage","has_readme","has_contributing","has_code_of_conduct","code_of_conduct_key",
        "has_support","has_issue_template","has_pr_template",
        "has_funding_file","funding_platforms",
        "error",
    ]
    header = input_cols + extra_cols

    for language in languages:
        in_path = f"repos_all_keywords__{language}.csv"
        out_path = f"repos_mined_all_info_{language}.csv"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        with open(in_path, newline='', encoding="utf-8") as fin, \
             open(out_path, "a", newline="", encoding="utf-8") as fout:

            reader = csv.DictReader(fin)
            for rc in required_cols:
                if rc not in reader.fieldnames:
                    sys.exit(f"ERROR: Input CSV missing required column: {rc}")

            writer = csv.DictWriter(fout, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            #START_AT_ROW = 2944
            #START_AT_ROW = 288
            START_AT_ROW = 1234

            total = 0
            for i, row in enumerate(reader, 1):
                if i < START_AT_ROW:
                    continue
                total += 1
                try:
                    enriched = enrich_one(session, row)
                except Exception as e:
                    # Never stop streaming: record the error and continue
                    enriched = dict(row)
                    enriched["error"] = f"{type(e).__name__}: {e}"

                # Make sure all header keys exist (fill missing with "")
                for k in header:
                    if k not in enriched:
                        enriched[k] = ""

                writer.writerow(enriched)
                fout.flush()  # <- write progress to disk immediately

                print(f"[{i}] {row.get('username')}/{row.get('project_name')} → written")

        print(f"Done. Wrote {total} rows to {out_path}")

if __name__ == "__main__":
    main()
