#!/usr/bin/env python3
"""
LLM-based project classification using OpenRouter.

Usage:
    python llm_as_judge.py --input repos_mined_all_info_with_readme_Java.csv --output llm_classified_random_java.csv --model gpt-4o-mini

Requirements:
    pip install pandas langdetect requests
"""

import argparse
import pandas as pd
import requests
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import time

DetectorFactory.seed = 0  # deterministic language detection

# ---------------------------------------------------------------------
def filter_projects(df):
    """Keep only projects with English descriptions."""
    def safe_detect(text):
        try:
            return detect(str(text))
        except Exception:
            return "unknown"

    df = df[df["readme_content"].notna()].copy()
    df["lang"] = df["readme_content"].apply(safe_detect)
    df = df[df["lang"] == "en"].copy()
    print(f"✅ {len(df)} English projects with descriptions retained.")
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
    prompt = f"""
You are an expert in software engineering and open-source GitHub repositories.
Analyze the GitHub template repositories below and classify its main domain and purpose.

Regarding the domain, consider the following categories:
- Regular Python Projects/ testing: Templates representing general-purpose Python project structures that are not domain-specific. This includes CLI tools, package libraries, testing frameworks, and pytest-based setups. These repositories often serve as reusable scaffolds for organizing source code, tests, documentation, and continuous integration workflows following Python best practices.
- Game Development: Templates aimed at the creation of games, simulations, or interactive media. These repositories typically include setups for Pygame, Godot, Unity (Python integrations), or custom rendering/game-loop frameworks. They define core game logic, event handling, asset management, and physics configurations to accelerate prototyping and experimentation.
- Web Development: Templates that support the creation of web-based systems and APIs. They include frameworks such as Django, Flask, and FastAPI for backend development, and React or Next.js for frontend or full-stack applications. These repositories typically define routing, authentication, and database configuration scaffolds for web services.
- Data Management: Templates that enable or extend frameworks for managing datasets, data pipelines, or versioning systems. This domain includes projects like DataLad, DVC, Airflow, or MLflow, which provide mechanisms for data storage, lineage tracking, reproducibility, and research data management. Such templates focus on structuring workflows and metadata handling rather than model training or inference.
- Machine Learning/AI: Templates designed for developing, training, or deploying AI and machine learning models. This includes frameworks such as PyTorch, TensorFlow, Hugging Face Transformers, and LangChain. Repositories in this domain provide standardized experiment setups, model architectures, data loaders, and evaluation pipelines for ML research or production.
- Automation/DevOps: Templates that streamline development operations, CI/CD, and system deployment. Typical examples are Docker, GitHub Actions, Terraform, and other infrastructure-as-code or workflow-automation setups. These templates define reproducible environments, containerized services, and automated pipelines for testing, deployment, or monitoring.
- Communication/Bots: Templates focused on conversational or messaging applications, such as Discord or Telegram bots. They often include preconfigured APIs, event-handling logic, and modular command structures to facilitate the rapid development of interactive communication agents or chat automation tools.
- Data Apps/ Visualization: Templates for interactive data-driven applications and dashboards, using frameworks such as Streamlit, Dash, or Plotly. These repositories emphasize data presentation and exploration, offering visual interfaces for analysis or model interpretation rather than backend infrastructure.
- Infrastructure/Cloud: Templates designed to support cloud-native architectures or serverless computing, such as AWS Lambda or Serverless Framework projects. They define deployment configurations, resource provisioning, and event-driven execution models that enable scalable and resilient cloud applications.
- Education/Research: Templates that provide standardized structures for academic projects, course material, or research experiments. These repositories are common in university or open-science contexts, including ML course projects, reproducible study templates, or academic paper codebases. Their main goal is to promote reproducibility and ease of learning.
- Robotics/ IoT: Templates focused on embedded, robotic, or sensor-based systems such as ROS, Swarm, or Railway. These projects define control architectures, deployment environments, and simulation setups for robotic applications or distributed IoT networks.
- Plugin/ Extensibility Frameworks: Templates that provide the basic structure to extend existing ecosystems, e.g., creating new modules or plugins for KiCAD, Colcon, or DataLad. They serve as minimal scaffolds that define registration mechanisms, configuration files, and integration points for extending a host platform’s functionality.

Repository URL: {readme_text}
Description: {description}{topics_text}

Please:
1. Summarize what the project is about (1–2 sentences);
2. Classify it into one of the previous informed high-level domains. If you can't associate the repository with one of the previous domains, you can provide a new one.
3. Finally, provide a reasoning about your choice. For that, you can highlight how the previous summarization related to the domain description. 

Return the answer in JSON format:
{{
  "summary": "...",
  "category": "...",
  "reasoning": "..."
}}
"""

    prompt_default = f"""
You are an expert in software engineering and open-source GitHub repositories.
Analyze the GitHub template repositories below and classify their main domain and purpose.

Regarding the domain, consider the following categories:
- Regular Projects/ testing: Templates representing general-purpose project structures that are not domain-specific. This includes CLI tools, package libraries, testing frameworks, and testing setups. These repositories often serve as reusable scaffolds for organizing source code, tests, documentation, and continuous integration workflows following the target programming language best practices.
- Game Development: Templates aimed at the creation of games, simulations, or interactive media. These repositories typically include setups for libraries, IDEs, or custom rendering/game-loop frameworks. They define core game logic, event handling, asset management, and physics configurations to accelerate prototyping and experimentation.
- Web Development: Templates that support the creation of web-based systems and APIs. They include web frameworks for backend, frontend, and/or full-stack applications. These repositories typically define routing, authentication, and database configuration scaffolds for web services.
- Data Management: Templates that enable or extend frameworks for managing datasets, data pipelines, or versioning systems. This domain includes projects providing mechanisms for data storage, lineage tracking, reproducibility, and research data management. Such templates focus on structuring workflows and metadata handling rather than model training or inference.
- Machine Learning/AI: Templates designed for developing, training, or deploying AI and machine learning models. This includes AI frameworks and ML models. Repositories in this domain provide standardized experiment setups, model architectures, data loaders, and evaluation pipelines for ML research or production.
- Automation/DevOps: Templates that streamline development operations, CI/CD, and system deployment. Typical examples are Docker, GitHub Actions, Terraform, and other infrastructure-as-code or workflow-automation setups. These templates define reproducible environments, containerized services, and automated pipelines for testing, deployment, or monitoring.
- Communication/Bots: Templates focused on conversational or messaging applications, such as Discord or Telegram bots. They often include preconfigured APIs, event-handling logic, and modular command structures to facilitate the rapid development of interactive communication agents or chat automation tools.
- Data Apps/ Visualization: Templates for interactive data-driven applications and dashboards. These repositories emphasize data presentation and exploration, offering visual interfaces for analysis or model interpretation rather than backend infrastructure.
- Infrastructure/Cloud: Templates designed to support cloud-native architectures or serverless computing, such as AWS Lambda or Serverless Framework projects. They define deployment configurations, resource provisioning, and event-driven execution models that enable scalable and resilient cloud applications.
- Education/Research: Templates that provide standardized structures for academic projects, course material, or research experiments. These repositories are common in university or open-science contexts, including ML course projects, reproducible study templates, or academic paper codebases. Their main goal is to promote reproducibility and ease of learning.
- Robotics/ IoT: Templates focused on embedded, robotic, or sensor-based systems. These projects define control architectures, deployment environments, and simulation setups for robotic applications or distributed IoT networks.
- Plugin/ Extensibility Frameworks: Templates that provide the basic structure to extend existing ecosystems, e.g., creating new modules, plugins or libraries. They serve as minimal scaffolds that define registration mechanisms, configuration files, and integration points for extending a host platform’s functionality.

Description: {description}{topics_text}
README file: {readme_text}

Please:
1. Summarize what the project is about (1–2 sentences);
2. Classify it into one of the previous informed high-level domains. If you can't associate the repository with one of the previous domains, you can provide a new one.
3. Finally, provide a reasoning about your choice. For that, you can highlight how the previous summarization related to the domain description. 

Return the answer in JSON format:
{{
  "summary": "...",
  "category": "...",
  "reasoning": "..."
}}
"""

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_default}],
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
        return output.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main(input_path, output_path, model):
    api_key = "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"
    df = pd.read_csv(input_path)
    df = filter_projects(df)
    #df = df.head(20).copy()  # process first 20 projects
    df = df.sample(n=10, random_state=42).copy()  # randomly select 100 projects

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying projects"):
        repo_url = row["readme_content"]
        description = row["repo_description"]
        topics = row.get("topics", "")
        output = classify_project(api_key, model, repo_url, description, topics)
        results.append({
            "repo_url": row["original_link_repository"],
            "description": description,
            "topics": topics,
            "llm_output": output,
        })
        time.sleep(2)  # gentle rate limiting

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"💾 Saved results to {output_path}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM classification of GitHub projects using OpenRouter.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenRouter model name.")
    args = parser.parse_args()
    main(args.input, args.output, args.model)
