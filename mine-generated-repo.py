import requests
import time
import csv

import config

BASE_URL = "https://api.github.com"
GITHUB_TOKEN = config.GITHUB_TOKEN
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def save_selected_repo_to_csv(data, filename="selected_template_repos.csv"):
    fieldnames = [
        "username", "project_name", "original_link",
        "stars", "forks", "data_creation", "data_update",
        "is_template", "target_template_repository"
    ]

    try:
        with open(filename, 'x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(data)


def search_github_templates_with_star_ranges(base_query="", token=None, per_page=100, max_pages=30, target_template=None, star_ranges=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    current_page = 1

    while current_page <= max_pages:
        base_url = f"{BASE_URL}/search/repositories"
        params = {
            "q": f"{base_query}".strip(),
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": current_page
        }

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"[!] Error: {response.status_code} - {response.json().get('message')}")
            break

        data = response.json()
        repos = data.get("items", [])
        if not repos:
            break

        selected_repos = []

        for repo in repos:
            full_name = repo["full_name"]
            details = requests.get(f"{BASE_URL}/repos/{full_name}", headers=headers)

            if details.status_code != 200:
                print(f"⚠️ Failed to get {full_name}: {details.json().get('message')}")
                continue

            repo_data = details.json()
            template_info = repo_data.get("template_repository")
            # We only consider repositories that generated from the target template repository
            if template_info and template_info.get("full_name") == target_template:
                print(f"✅ {repo['full_name']} ({repo['stargazers_count']} ⭐)\n   {repo['html_url']}")

                username, project_name = repo["full_name"].split("/")
                selected_repos.append({
                    "username": username,
                    "project_name": project_name,
                    "original_link": repo["html_url"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "data_creation": repo["created_at"],
                    "data_update": repo["updated_at"],
                    "is_template": repo["is_template"],
                    "target_template_repository": template_info["full_name"]
                })

            time.sleep(0.3)

        if selected_repos:
            save_selected_repo_to_csv(selected_repos)

        current_page += 1
        time.sleep(1)

if __name__ == "__main__":
    token = config.GITHUB_TOKEN

    with open("template_repos.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            username = row["username"]
            project_name = row["project_name"]
            target_template = username+"/"+project_name
            base_query = f"language:python in:readme sort:updated {target_template}"

            print(f"\n🔎 Searching for template-based repos from: {username}/{project_name}")
            search_github_templates_with_star_ranges(
                base_query=base_query,
                token=token,
                target_template=target_template
            )