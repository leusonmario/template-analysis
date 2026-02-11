#!/usr/bin/env python3
"""
LLM-based project classification using OpenRouter.

Usage:
    python llm_judge_final.py \
        --input repos_mined_all_info_with_readme_Python.csv \
        --output llm_classified_Python.csv \
        --model gpt-4o-mini \
        --focus focus_list.csv \
        --focus-col repo_url

python llm_judge_final.py --input repos_mined_all_info_with_readme_Python.csv --output llm_classified_selected_Python.csv --model gpt-4o-mini --focus selected_projects_classified_Python.csv --focus-col repo_url

Requirements:
    pip install pandas langdetect requests tqdm
"""

import argparse
import csv
import os
import time
from typing import Optional, Set

import pandas as pd
import requests
from langdetect import detect, DetectorFactory
from tqdm import tqdm

DetectorFactory.seed = 0  # deterministic language detection


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_url(u: str) -> str:
    """Lightweight normalization for matching URLs across CSVs."""
    if pd.isna(u) or u is None:
        return ""
    s = str(u).strip()
    if not s:
        return s
    s = s.rstrip("/")  # drop trailing slash
    return s


def load_focus_set(focus_csv_path: str, focus_col: Optional[str]) -> Set[str]:
    """Load the set of repo URLs to keep from a 'focus' CSV."""
    fdf = pd.read_csv(focus_csv_path, dtype=str)
    if focus_col is None:
        for cand in ["repo_url", "original_link_repository", "url", "link", "repository"]:
            if cand in fdf.columns:
                focus_col = cand
                break
    if not focus_col or focus_col not in fdf.columns:
        raise ValueError(
            f"Could not find a URL column in {focus_csv_path}. "
            "Use --focus-col to specify it (e.g., --focus-col repo_url)."
        )
    return set(fdf[focus_col].dropna().map(normalize_url).unique())


def safe_lang_detect(text) -> str:
    try:
        return detect(str(text))
    except Exception:
        return "unknown"


def filter_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only projects with English READMEs."""
    if "readme_content" not in df.columns:
        raise KeyError("Column 'readme_content' not found in input CSV.")
    df = df[df["readme_content"].notna()].copy()
    df["lang"] = df["readme_content"].apply(safe_lang_detect)
    df = df[df["lang"] == "en"].copy()
    print(f"✅ {len(df)} English projects retained for classification.")
    return df


def already_processed(repo_url: str, output_path: str) -> bool:
    """Check if this repo is already saved in the output CSV (by normalized URL)."""
    if not os.path.exists(output_path):
        return False
    try:
        # Use streaming csv reader to avoid loading entire file into memory.
        nurl = normalize_url(repo_url)
        with open(output_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return any(normalize_url(row.get("repo_url", "")) == nurl for row in reader)
    except Exception:
        return False


# ---------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------
import requests

def classify_project(api_key, model, readme_text, description, topics=None):
    """Send one classification request to DeepSeek."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    topics_text = f"\nTopics: {topics}" if topics else ""
    prompt = f"""
You are an expert in software engineering and open-source GitHub repositories.
Analyze the GitHub template repository below and classify its main associated domain.
We provide the description and README content of the project, if available.

Regarding the domain, consider the following categories:
- Automation/DevOps: Templates that streamline development operations, CI/CD, and system deployment. Typical examples include Docker, GitHub Actions, Terraform, and other infrastructure-as-code or workflow-automation setups. These templates define reproducible environments, containerized services, and automated pipelines for testing, deployment, or monitoring.
- Communication/Bots: Templates focused on conversational or messaging applications, such as Discord or Telegram bots. They often include preconfigured APIs, event-handling logic, and modular command structures to facilitate the rapid development of interactive communication agents or chat automation tools, and QA
- Data Apps/Visualization: Templates for interactive data-driven applications and dashboards. These repositories emphasize data presentation and exploration, offering visual interfaces for analysis or model interpretation rather than backend infrastructure.
- Data Management: Templates that enable or extend frameworks for managing datasets, data pipelines, or versioning systems. This domain includes projects providing mechanisms for data storage, lineage tracking, reproducibility, and research data management. Such templates focus on structuring workflows and metadata handling rather than model training or inference.
- Education: Templates that provide standardized structures for academic projects, course and book material, assignments, hackathons, showcases, and example/demonstration implementations. These repositories are common in university or open-science contexts, including course projects, tutorials, recommendations, and coding competitions. Their main goal is to promote reproducibility, ease of learning, explaining/demonstrating how to use specific technologies or patterns, and/or sharing of educational material.
- Game Development: Templates aimed at the creation of games, simulations, or interactive media. These repositories typically include setups for libraries, IDEs, or custom rendering/game-loop frameworks. They define core game logic, event handling, asset management, and physics configurations to accelerate prototyping and experimentation.
- Infrastructure/Cloud: Templates designed to support cloud-native architectures or serverless computing, such as AWS Lambda, Serverless Framework, or Spring Cloud projects. They define deployment configurations, resource provisioning, and event-driven execution models that enable scalable and resilient cloud applications.
- Machine Learning/AI: Templates designed for developing, training, or deploying AI and machine learning models. This includes AI frameworks and ML pipelines. Repositories in this domain provide standardized experiment setups, model architectures, data loaders, and evaluation workflows for research or production.
- Mobile Development: Templates for creating native or cross-platform mobile apps using frameworks such as Flutter, React Native, Android Studio (Java/Kotlin), or Swift. They define core structures for UI, navigation, and API integration, often including authentication, storage, and deployment configurations for app distribution or testing.
- Plugin/Extensibility Frameworks: Templates that provide the basic structure to extend existing ecosystems, such as creating new modules and plugins, or support the defintion/creaiton of new libraries. They serve as minimal scaffolds that define registration mechanisms, configuration files, and integration points for extending a host platform’s functionality.
- Regular Project: Templates representing general-purpose project structures that are not domain-specific. This includes CLI tools, package libraries, testing frameworks, and generic software setups. These repositories often serve as reusable scaffolds for organizing source code, tests, documentation, and continuous integration workflows following language-specific best practices.
- Research: Templates specifically focused on reproducible scientific studies, empirical experiments, academic paper codebases, benchmark evaluations, academic competitions (challenges), and initial investigations. These repositories often include data analysis scripts, experiment configurations, and result replication pipelines. Their main goal is to ensure scientific transparency, facilitate replication, and accelerate knowledge transfer within research communities.
- Robotics/IoT: Templates focused on embedded, robotic, or sensor-based systems. These projects define control architectures, deployment environments, and simulation setups for robotic applications or distributed IoT networks.
- Web Development: Templates that support the creation of web-based systems and APIs/SaaS. They include popular web frameworks for backend, frontend, and/or full-stack applications. These repositories typically define structure (code/test/deploy), routing, authentication, and database configuration scaffolds for web services.

Description: {description}{topics_text}
README file: {readme_text}

Please:
1) Summarize what the project is about (1–2 sentences);
2) Understand the key characteristics of the project;
3) Classify it into exactly one of the categories above (focus on the main goal; if none fits, you may propose a new one);
4) Provide a short reasoning linking the summary to the chosen category.
5) Report your confidence with your label, reporting a number between 1 and 5. Values close to 5 mean high confidence, while values close to 1 means less confidence. 

Return JSON only:
{{
  "category": "...",
  "reasoning": "..."
  "confidence": "..."
}}
"""

    data = {
        "model": model,  # e.g., "deepseek-chat" or "deepseek-coder"
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(input_path, output_path, model, focus_csv=None, focus_col=None, sample_n=None, sleep_s=0):
    #api_key = "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"
    api_key = "sk-0d643e7550db452e897642c943e2feb7"
    if not api_key:
        raise RuntimeError("Set your API key in the environment variable OPENROUTER_API_KEY.")

    df = pd.read_csv(input_path, dtype=str)
    if "original_link_repository" not in df.columns:
        raise KeyError("Column 'original_link_repository' not found in input CSV.")

    # Filter by English README
    df = filter_projects(df)

    # Optional: filter by focus CSV
    if focus_csv:
        focus_set = load_focus_set(focus_csv, focus_col)
        df["__norm_url"] = df["original_link_repository"].map(normalize_url)
        before = len(df)
        df = df[df["__norm_url"].isin(focus_set)].copy()
        df.drop(columns=["__norm_url"], inplace=True)
        print(f"🗂️ Focus filter: kept {len(df)}/{before} rows from {focus_csv}.")

    # Optional: sample for quick runs
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).copy()
        print(f"🎯 Sampling enabled: {len(df)} rows.")

    # Prepare output CSV (incremental)
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["repo_url", "description", "topics", "llm_output"])
            writer.writeheader()

    # Iterate and classify
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying projects"):
        repo_url = row.get("original_link_repository", "")
        if not repo_url or repo_url != "https://github.com/bgoonz/BGOONZ_BLOG_2.0":
            continue

        if already_processed(repo_url, output_path):
            print(f"⏩ Skipping already classified: {repo_url}")
            continue

        description = row.get("repo_description", "") or ""
        topics = row.get("topics", "") or ""
        readme_text = row.get("README", "") or ""

        print(f"📘 Classifying: {repo_url}")
        llm_output = classify_project(api_key, model, readme_text, description, topics)

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["repo_url", "description", "topics", "llm_output"])
            writer.writerow(
                {
                    "repo_url": repo_url,
                    "description": description,
                    "topics": topics,
                    "llm_output": llm_output,
                }
            )

        #time.sleep(max(0, sleep_s))  # gentle rate limiting

    print("\n✅ All repositories processed incrementally and saved.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM classification of GitHub projects using OpenRouter.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenRouter model name.")
    parser.add_argument("--focus", help="Optional CSV with URLs to process (subset filter).")
    parser.add_argument("--focus-col", help="Column name in --focus CSV containing URLs.")
    parser.add_argument("--sample-n", type=int, help="Optional sample size for quick runs.")
    parser.add_argument("--sleep-s", type=float, default=3.0, help="Sleep seconds between requests.")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        focus_csv=args.focus,
        focus_col=args.focus_col,
        sample_n=args.sample_n,
        sleep_s=args.sleep_s,
    )
