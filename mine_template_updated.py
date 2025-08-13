import requests
import time
import csv
from collections import OrderedDict

BASE_URL = "https://api.github.com/search/repositories"
UA = "GH-Sampler/1.0"

# ----- Config -----
PER_PAGE = 100                     # max
MAX_PAGES = 10                     # hard GitHub cap
PAGE_SLEEP = 0.5                   # between pages
QUERY_SLEEP = 0.8                  # between different queries (avoid abuse)
SPLIT_THRESHOLD = 1000             # GitHub returns only first 1000 hits
RETRY = 3                          # network retries
TIMEOUT = 30

from datetime import date
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

def five_year_cutoff_iso():
    # Today is your local date; GitHub expects ISO UTC date (no time needed).
    cutoff = date.today() - relativedelta(years=2)
    print(cutoff)
    return cutoff.isoformat()  # 'YYYY-MM-DD'

def gh_get(url, headers, params):
    for attempt in range(RETRY):
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r
        # Handle secondary rate limits / abuse detection
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            wait = 60 if reset is None else max(1, int(reset) - int(time.time()) + 1)
            time.sleep(wait)
            continue
        # transient?
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r  # just in case

def total_count_for_range(base_query, star_min, star_max, headers):
    # Build stars qualifier
    if star_max is None:
        stars_q = f"stars:>={star_min}"
    else:
        stars_q = f"stars:{star_min}..{star_max}"
    q = f"{base_query} {stars_q}".strip()
    params = {"q": q, "per_page": 1}  # only want total_count
    r = gh_get(BASE_URL, headers, params)
    data = r.json()
    return data.get("total_count", 0)

def split_ranges_adaptively(base_query, min_stars, max_stars, headers):
    """
    Returns a list of (lo, hi) inclusive star ranges such that each query yields <= SPLIT_THRESHOLD results.
    If max_stars is None, the top range is open-ended with stars:>=lo.
    """
    ranges = []
    stack = [(min_stars, max_stars)]
    while stack:
        lo, hi = stack.pop()
        cnt = total_count_for_range(base_query, lo, hi, headers)
        if cnt <= SPLIT_THRESHOLD:
            ranges.append((lo, hi, cnt))
            time.sleep(QUERY_SLEEP)
            continue

        # Need to split
        if hi is None:
            # open-ended: split by growing the lower bound geometrically
            # pick a pivot by multiplying lo (avoid infinite loop on low lo)
            pivot = max(lo + 1, int(lo * 2))  # crude but effective
            left = (lo, pivot - 1)
            right = (pivot, None)
        else:
            if hi <= lo:
                # degenerate; fall back to accept
                ranges.append((lo, hi, cnt))
                continue
            pivot = (lo + hi) // 2
            left = (lo, pivot)
            right = (pivot + 1, hi)

        # Push right then left so left is processed first (LIFO)
        stack.append(right)
        stack.append(left)
        time.sleep(QUERY_SLEEP)

    # Sort by lo ascending
    ranges.sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))
    return ranges

def save_repo_info_to_csv(rows, filename_base="template_repos_all_new", language=""):
    filename_final = f"{filename_base}_{language}.csv"
    fieldnames = [
        "id", "username", "project_name", "original_link_repository",
        "stars", "forks", "created_at", "updated_at", "language"
    ]
    # Write header if new
    try:
        with open(filename_final, "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
    except FileExistsError:
        pass
    # Append
    with open(filename_final, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerows(rows)

def fetch_range(base_query, star_min, star_max, headers, language):
    if star_max is None:
        stars_q = f"stars:>={star_min}"
    else:
        stars_q = f"stars:{star_min}..{star_max}"
    q = f"{base_query} {stars_q}".strip()

    seen_ids = set()
    out_rows = []

    for page in range(1, MAX_PAGES + 1):
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": PER_PAGE,
            "page": page
        }
        r = gh_get(BASE_URL, headers, params)
        data = r.json()
        items = data.get("items", []) or []
        if not items:
            break

        for repo in items:
            # Only keep actual templates (avoid extra /repos/{full_name} call)
            if not repo.get("is_template", False):
                continue

            rid = repo.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            full_name = repo.get("full_name", "")
            username, project_name = (full_name.split("/", 1) + [""])[:2]
            out_rows.append({
                "id": rid,
                "username": username,
                "project_name": project_name,
                "original_link_repository": repo.get("html_url"),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "language": language
            })

        # stop early if last page had < PER_PAGE
        if len(items) < PER_PAGE:
            break
        time.sleep(PAGE_SLEEP)

    return out_rows

def mine_language(language, token, filename_base, min_stars=0, max_stars=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": UA,
    }

    cutoff = five_year_cutoff_iso()

    # Base query: narrow to language, template-ish keywords, public, non-archived, non-fork.
    # Tip: include readme in text search; it helps surface templates.
    base_query = (
        f"language:{language} "
        f"in:name,description,readme "
        f"(template OR boilerplate OR starter OR skeleton) "
        f"fork:false archived:false is:public"
        f"pushed:>={cutoff}"
    )

    print(f"\n=== {language} ===")
    print("Building adaptive star ranges...")
    ranges = split_ranges_adaptively(base_query, min_stars, max_stars, headers)
    for lo, hi, cnt in ranges:
        print(f"  range {lo}..{('∞' if hi is None else hi)} => ~{cnt} repos")

    # Mine each range
    global_seen = set()
    for lo, hi, _ in ranges:
        rows = fetch_range(base_query, lo, hi, headers, language)
        # Dedup across ranges via id
        new_rows = []
        for r in rows:
            if r["id"] in global_seen:
                continue
            global_seen.add(r["id"])
            new_rows.append(r)
        if new_rows:
            save_repo_info_to_csv(new_rows, filename_base, language)
            print(f"Saved {len(new_rows)} rows for range {lo}..{('∞' if hi is None else hi)}")
        time.sleep(QUERY_SLEEP)

def main():
    # ---------- YOUR SETTINGS ----------
    import config  # expects GITHUB_TOKEN
    filename_base = "repos_all_recent_"
    languages = ["C#", "Java", "TypeScript", "Python", "JavaScript"]
    min_stars = 2
    max_stars = None  # None = open ended (>= min_stars)
    # -----------------------------------

    for lang in languages:
        mine_language(
            language=lang,
            token=config.GITHUB_TOKEN,
            filename_base=filename_base,
            min_stars=min_stars,
            max_stars=max_stars
        )

if __name__ == "__main__":
    main()
