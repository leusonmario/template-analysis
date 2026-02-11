#!/usr/bin/env python3
"""
LLM-based project classification using OpenRouter.

Usage:
    python llm_judge_final.py --input repos_mined_all_info_with_readme_Python.csv --output llm_classified_Python.csv --model gpt-4o-mini

Requirements:
    pip install pandas langdetect requests tqdm
"""

import argparse
import pandas as pd
import requests
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import time
import csv
import os

DetectorFactory.seed = 0  # deterministic language detection

# ---------------------------------------------------------------------
def filter_projects(df):
    """Keep only projects with English READMEs."""
    def safe_detect(text):
        try:
            return detect(str(text))
        except Exception:
            return "unknown"

    df = df[df["readme_content"].notna()].copy()
    df["lang"] = df["readme_content"].apply(safe_detect)
    df = df[df["lang"] == "en"].copy()
    print(f"✅ {len(df)} English projects retained for classification.")
    return df


def classify_project(api_key, model, readme_text, description, topics=None):
    """Send one classification request to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "TemplateRepoClassifier",
        "Content-Type": "application/json",
    }

    topics_text = f"\nTopics: {topics}" if topics else ""

    # --- Unified prompt for Python + Java projects ---
    prompt = f"""
You are an expert in software engineering and open-source GitHub repositories.
Analyze the GitHub template repository below and classify its main associated domain.
For that, below we provide the description and README content of the project, if available.

Regarding the domain, consider the following categories:
- Automation/ DevOps: Templates that streamline development operations, CI/CD, and system deployment. Typical examples include Docker, GitHub Actions, Terraform, and other infrastructure-as-code or workflow-automation setups. These templates define reproducible environments, containerized services, and automated pipelines for testing, deployment, or monitoring.
- Communication/ Bots: Templates focused on conversational or messaging applications, such as Discord or Telegram bots. They often include preconfigured APIs, event-handling logic, and modular command structures to facilitate the rapid development of interactive communication agents or chat automation tools, and QA
- Data Apps/ Visualization: Templates for interactive data-driven applications and dashboards. These repositories emphasize data presentation and exploration, offering visual interfaces for analysis or model interpretation rather than backend infrastructure.
- Data Management: Templates that enable or extend frameworks for managing datasets, data pipelines, or versioning systems. This domain includes projects providing mechanisms for data storage, lineage tracking, reproducibility, and research data management. Such templates focus on structuring workflows and metadata handling rather than model training or inference.
- Education: Templates that provide standardized structures for academic projects, course material, assignments, hackathons, and example implementations. These repositories are common in university or open-science contexts, including course projects, tutorials, and coding competitions. Their main goal is to promote reproducibility, ease of learning, explaining how to use specific technologies or patterns, and/or sharing of educational material.
- Game Development: Templates aimed at the creation of games, simulations, or interactive media. These repositories typically include setups for libraries, IDEs, or custom rendering/game-loop frameworks. They define core game logic, event handling, asset management, and physics configurations to accelerate prototyping and experimentation.
- Infrastructure/ Cloud: Templates designed to support cloud-native architectures or serverless computing, such as AWS Lambda, Serverless Framework, or Spring Cloud projects. They define deployment configurations, resource provisioning, and event-driven execution models that enable scalable and resilient cloud applications.
- Machine Learning/ AI: Templates designed for developing, training, or deploying AI and machine learning models. This includes AI frameworks and ML pipelines. Repositories in this domain provide standardized experiment setups, model architectures, data loaders, and evaluation workflows for research or production.
- Mobile Development: Templates for creating native or cross-platform mobile apps using frameworks such as Flutter, React Native, Android Studio (Java/Kotlin), or Swift. They define core structures for UI, navigation, and API integration, often including authentication, storage, and deployment configurations for app distribution or testing.
- Plugin/ Extensibility Frameworks: Templates that provide the basic structure to extend existing ecosystems, such as creating new modules, plugins, or libraries. They serve as minimal scaffolds that define registration mechanisms, configuration files, and integration points for extending a host platform’s functionality.
- Regular Project: Templates representing general-purpose project structures that are not domain-specific. This includes CLI tools, package libraries, testing frameworks, and generic software setups. These repositories often serve as reusable scaffolds for organizing source code, tests, documentation, and continuous integration workflows following language-specific best practices.
- Research: Templates specifically focused on reproducible scientific studies, empirical experiments, academic paper codebases, benchmark evaluations, academic competitions (challenges), and initial investigations. These repositories often include data analysis scripts, experiment configurations, and result replication pipelines. Their main goal is to ensure scientific transparency, facilitate replication, and accelerate knowledge transfer within research communities.
- Robotics/ IoT: Templates focused on embedded, robotic, or sensor-based systems. These projects define control architectures, deployment environments, and simulation setups for robotic applications or distributed IoT networks.
- Web Development: Templates that support the creation of web-based systems and APIs/SaaS. They include popular web frameworks for backend, frontend, and/or full-stack applications. These repositories typically define structure (code/test/deploy), routing, authentication, and database configuration scaffolds for web services.

Description: {description}{topics_text}
README file: {readme_text}

Please:
1. Summarize what the project is about (1–2 sentences);
2. Understand the key characteristics of the project;
3. Classify it into one of the previous informed high-level domains. Focus on the main goal, not on secondary goals. If you can't associate the repository with one of the previous domains, you can provide a new one.
4. Finally, provide a reasoning about your choice. For that, you can highlight how the previous summarization related to the domain description. 

Return the answer in JSON format:
{{
  "category": "...",
  "reasoning": "..."
}}
"""

    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=90
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------
def already_processed(repo_url, output_path):
    """Check if this repo is already saved in the output CSV."""
    if not os.path.exists(output_path):
        return False
    try:
        processed = pd.read_csv(output_path, usecols=["repo_url"])
        return repo_url in set(processed["repo_url"])
    except Exception:
        return False


def main(input_path, output_path, model):
    # ⚠️ Define your key directly here
    api_key = "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"

    df = pd.read_csv(input_path)
    df = filter_projects(df)
    #df = df.sample(n=10, random_state=42).copy()  # adjust number as needed

    # Prepare output CSV (incremental mode)
    if not os.path.exists(output_path):
        with open(output_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["repo_url", "description", "topics", "llm_output"])
            writer.writeheader()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying projects"):
        repo_url = row["original_link_repository"]

        if repo_url == "https://github.com/AtticusZeller/fastapi_supabase_template" or repo_url == "https://github.com/Blinorot/pytorch_project_template" or repo_url == "https://github.com/grisuno/LazyOwn" or repo_url == "https://github.com/danielrosehill/System-Prompt-Library":

            if already_processed(repo_url, output_path):
                print(f"⏩ Skipping already classified: {repo_url}")
                continue

            description = row.get("repo_description", "")
            topics = row.get("topics", "")
            readme_text = row.get("readme_content", "")

            print(f"📘 Classifying: {repo_url}")
            llm_output = classify_project(api_key, model, readme_text, description, topics)

            # Append immediately (crash-safe)
            with open(output_path, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["repo_url", "description", "topics", "llm_output"])
                writer.writerow({
                    "repo_url": repo_url,
                    "description": description,
                    "topics": topics,
                    "llm_output": llm_output,
                })

            time.sleep(3)  # gentle rate limiting

    print("\n✅ All repositories processed incrementally and saved.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM classification of GitHub projects using OpenRouter.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenRouter model name.")
    args = parser.parse_args()
    main(args.input, args.output, args.model)
