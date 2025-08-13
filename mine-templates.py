import requests
import time
import csv

import config
BASE_URL = "https://api.github.com"

def save_repo_info_to_csv(data, filename="template_repos_all_new",language=""):
    filename_final = filename + "_" + language + ".csv"

    fieldnames = [
        "username", "project_name", "original_link_repository",
        "stars", "forks", "data_creation", "data_update"
    ]

    try:
        with open(filename_final, 'x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(filename_final, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(data)

def search_github_templates_with_star_ranges(base_query="", token=None, per_page=100, max_pages=30, language=""):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    star_ranges = [
        "stars:>=5000",
        "stars:1000..4999",
        "stars:500..999",
        "stars:300..499",
        "stars:100..299",
        "stars:80..99",
        "stars:60..79",
        "stars:50..59",
        "stars:40..49",
        "stars:30..39",
        "stars:20..29",
        "stars:10..19",
        "stars:7..9",
        "stars:5..6",
        "stars:4..4",
        "stars:3..3",
        "stars:2..2",
        #"stars:1..1",
        #"stars:0..1",
    ]

    for star_range in star_ranges:
        current_page = 1
        keep_going = True
        while keep_going:
            base_url = "https://api.github.com/search/repositories"
            params = {
                "q": f" {base_query} {star_range}".strip(),
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

            if len(repos) < 100:
                keep_going = False

            if not repos:
                time.sleep(1)
                break

            repo_infos = []
            for repo in repos:
                #We consider only repositories that are true templates.
                full_name = repo["full_name"]
                details = requests.get(f"{BASE_URL}/repos/{full_name}", headers=headers)

                if full_name == "eunomia-bpf/libbpf-starter-template":
                    print("HERE")

                if details.status_code != 200:
                    print(f"⚠️ Failed to get {full_name}: {details.json().get('message')}")
                    continue

                repo_data = details.json()
                #template_info = repo_data.get("template_repository")

                if repo.get('is_template', True):# and template_info is None:
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

                    time.sleep(0.5)

            save_repo_info_to_csv(repo_infos, language=language)

            current_page += 1
            time.sleep(1)


if __name__ == "__main__":
    languages = ["JavaScript", "Java", "Typescript"]
    language = "JavaScript"
    search_github_templates_with_star_ranges(
        base_query="language:"+language+" in:name,description (template OR boilerplate OR starter OR skeleton) fork:false archived:false",
        token=config.GITHUB_TOKEN,
        language=language
    )