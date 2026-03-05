# Template Repository Analysis

This repository contains the artifacts and scripts used to analyze **GitHub template repositories**, focusing on their domains, characteristics and activity, quality practices (guidelines), and common pitfalls. 
The goal of this project is to support empirical studies on how template repositories are created, maintained, and adopted across different ecosystems, covering TypeScript, Python, JavaScript, Java, and C#.

The repository includes scripts for **data collection, automated analysis (RQs), and adoption of LLMs for checking adherence of guidelines and pitfalls** in template repositories.

---

# Repository Structure

The repository is organized into several directories, each responsible for a specific step of the analysis pipeline.

```
template-analysis/
│
├── data_collection/
├── llm_evaluation/
├── rq1/
│   ├── llm_evaluation/
│   └── owner_evaluation/

├── rq2/
│   ├── correlation_complete/
│   └── regression_complete/
│
├── rq3/
│   ├── llm_assistant/
│   └── rq3_samples/
```

## data_collection/

This folder contains the **scripts and mined datasets used during the study**, including:

- GitHub template repositories analyzed in the study
- Scripts used based on GitHub API
- Metadata about repositories (e.g., language, creation date, activity)

For each target programming languages, a specific CSV is provided.
These datasets are used as input for subsequent mining and analysis steps.

---

## llm_evaluation/

This directory contains the initial analysis leveraging LLMs for classifying the domains associated with GitHub templates.
For that, two LLMs were evaluated based on sample of 100 templates manually classified by a human.

For each LLM, we report the results of the classification provided alongside with the manual evaluation (ground-truth).

---

## rq1/

The `rq1` folder includes the scripts used to **DeepSeek model to classify the domains** associated with the mined GitHub Templates.

These scripts are responsible for tasks such as:

- Calling DeepSeek to classify the domain associated with a given template
- Plot figures about the distribution of domains and ownership of templates

---

## rq2/

This folder contains the scripts regarding the occurrence of quality and maintenance issues faced by GitHub templates.


that leverage **Large Language Models (LLMs)** to assist in evaluating template repositories.

The scripts are responsible for:

- Call the SonarQube tool to collect a set of metrics, including bugs, code smelles, vulnerabilities, etc. 
- Run binomial regression analysis based on repository activity and characteristics

---

## rq3/

This folder contains scripts responsible for selecting a sample of templates for analysis aiming to identify guidelines and pitalls.

The scripts handle:

- Organizing and selecting projects based on the best, worst, and mixed classification.
- Leverage LLMs to check the adherence to recommended practices (guidelines and pitfalls)


---

# Requirements

Typical requirements include:

- Python 3.10+
- pandas
- requests
- scikit-learn
- Access to an LLM API (e.g., OpenAI or OpenRouter)

Install dependencies:

```bash
pip install -r requirements.txt