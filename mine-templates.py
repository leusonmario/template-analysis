import requests
import time
import csv

import config


def save_repo_info_to_csv(data, filename="template_repos.csv"):
    fieldnames = [
        "username", "project_name", "original_link_repository",
        "stars", "forks", "data_creation", "data_update"
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

def search_github_templates_with_star_ranges(base_query="", token=None, per_page=100, max_pages=30, star_ranges=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    current_page = 1

    while current_page <= max_pages:
        base_url = "https://api.github.com/search/repositories"
        params = {
            "q": f" {base_query}".strip(),
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

        repo_infos = []
        for repo in repos:
            #We consider only repositories that are true templates.
            if repo.get('is_template', True):
                full_name = repo["full_name"]
                username, project_name = full_name.split("/")
                original_link_repository = repo["html_url"]
                stars = repo["stargazers_count"]
                forks = repo["forks_count"]
                data_creation = repo["created_at"]
                data_update = repo["updated_at"]

                print(f"✅ {full_name} ({stars} ⭐)\n   {original_link_repository}")

                repo_infos.append({
                    "username": username,
                    "project_name": project_name,
                    "original_link_repository": original_link_repository,
                    "stars": stars,
                    "forks": forks,
                    "data_creation": data_creation,
                    "data_update": data_update
                })

                time.sleep(0.3)

        save_repo_info_to_csv(repo_infos)

        current_page += 1
        time.sleep(1)


if __name__ == "__main__":
    search_github_templates_with_star_ranges(
        base_query="language:python created:>2023-01-01 pushed:>2024-01-01 stars:>4 template_repository:",
        token=config.GITHUB_TOKEN
    )