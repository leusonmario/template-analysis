#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

import config

# -----------------------------
# Configuration
# -----------------------------

MODEL = "gpt-5.2"   # direct OpenAI model
TEMPERATURE = 0.2
MAX_TOKENS = 2500

GLOBAL_OUT_DIR = Path("template_eval_outputs")
GLOBAL_CSV_PATH = GLOBAL_OUT_DIR / "results.csv"

DEDUPE = True

PROJECTS = [
    "https://github.com/NikolayIT/ASP.NET-Core-Template",
    "https://github.com/lgwk42/auth-template",
    "https://github.com/gpauloski/python-template",
    "https://github.com/The-Marcy-Lab-School/technical-prework-1",
    "https://github.com/thiago-roock/Microsservice-Application",
    "https://github.com/Jaxelr/VueSimpleTemplate",
    "https://github.com/SpongePowered/sponge-plugin-template",
    "https://github.com/MersadHabibi/template-nextjs-daisyui-reactquery",
    "https://github.com/delaurentis/planner",
    "https://github.com/Measurity/ModTemplateValheim",
    "https://github.com/platformsh-templates/django4",
    "https://github.com/bracesproul/monorepo-template",
    "https://github.com/BattlesnakeOfficial/starter-snake-javascript",
    "https://github.com/CodelyTV/java-ddd-example",
    "https://github.com/joaomlneto/travis-ci-tutorial-java",
    "https://github.com/nextjs/deploy-render",
]

# (GUIDELINES_AND_PITFALLS_TEXT unchanged)
# KEEP YOUR EXISTING GUIDELINES_AND_PITFALLS_TEXT HERE

GUIDELINES_AND_PITFALLS_TEXT = r"""
Act as a senior software engineer with extensive experience in GitHub repositories, specifically reviewing GitHub templates.

Now, you are asked to evaluate the given repository against a list of (A) guidelines and (B) pitfalls for creating and maintaining GitHub templates.

For EACH item (guideline or pitfall), output ONE JSON object keyed by its ID.

Definition:
- For a GUIDELINE: "adherence": "Yes" means the repository follows the guideline; "No" means it does not.
- For a PITFALL: "adherence": "Yes" means the repository is affected by the pitfall; "No" means it is not.

You MUST base your answers on evidence from public available content, mainly focusing on GitHub repository pages and reported content: 
internal or external) (README, .github/workflows, releases/tags, issues, repository description/topics, technologies used, resources, etc.).
Any information publicly available on the GitHub template repository. 

Output format (JSON only, no markdown, no extra text):
{
  "<ID>": {
    "adherence": "Yes" | "No",
    "reasoning": ["reasoning"],
    "confidence": <integer 0-5>
  },
  ...
}

Guidelines: 
G1: Adopt established software engineering practices: Templates that adopted established software engineering practices. In particular, they provided ready-to-use project setups and integrated common quality mechanisms, including, but not limited to, linting, testing at appropriate levels, CI/CD pipelines, hooks, automation tools, and bots, for different purposes, like dependency updates and code review. 

G2: Provide comprehensive/accessible documentation: Beyond explaining the project’s purpose, templates must guide users in understanding, customizing, and extending the provided structure to fit their specific needs. To this end, effective documentation can take multiple forms, including step-by-step guides, code snippets with explanatory descriptions, explicit technology and version requirements, links to external resources such as tutorials or videos, and creation of wikis. Templates may also provide documentation in different languages, supporting reuse while avoiding the need to create new templates to broadly support the same initial goal.

G3: Educate users on how to use the template feature: Although GitHub’s templating feature is not new, several templates provide inaccurate or incomplete instructions on how to properly use it. In particular, users are often instructed to clone the repository directly, rather than using the Use this template button. Templates should explicitly guide users toward the correct instantiation mechanism and update their documentation accordingly, especially when existing repositories are later converted into templates.

G4: Communicate evolution through a roadmap: GitHub repositories usually provide roadmaps to communicate upcoming changes and planned features. Templates can equally benefit from this practice, as a roadmap helps users understand how the template is expected to evolve and how it can be extended over time. This is particularly valuable for templates targeting the Education domain, where roadmaps can guide learners by informing tasks to be performed and expected outcomes.

G5: Organize templates as a reusable family: Templates can be designed as a reusable family, where new templates are derived from existing ones to preserve and propagate previously established good practices while introducing technology-specific variations. Practitioners may define a default template and subsequently use it as the basis for creating additional templates targeting different technologies, ecosystems, and programming languages. If you observe a template was based on previous ones, such guideline is applicable. Otherwise, it's just one single template without the need for having a family of templates.

G6: Align technology versions with releases: Templates are created to support the use of specific technologies, often alongside libraries or frameworks provided by organizations or individual developers. As these technologies evolve, templates should explicitly track the supported technology versions and associate them with corresponding template releases. Templates are expected to provide clear release information, for example, through versioned releases, allowing practitioners to identify compatible template versions while supporting reproducibility over time.

G7: Encourage community and social support: Collaborative software development inherently relies on community engagement through discussion, contribution, and shared problem-solving. For templates, this engagement can be affected by how practitioners consume, adapt, and extend the provided templates. Some initiatives, aimed at fostering active communities around templates, are dedicated communication channels (e.g., Discord servers) to support users. Similarly, some templates include calls for sponsorship, helping sustain long-term maintenance in ways comparable to traditional GitHub repositories.

Pitfalls:
P1: Non-templates: Some repositories, labeled as templates, were originally designed as full implementations, such as end-to-end applications, libraries, or frameworks. While some templates may provide useful examples of how to use specific technologies, non-templates often lack the scaffolding, guidance, and extensibility expected from a template. 

P2: Unclear or inactive repository status: Several inactive templates fail to clearly communicate their current status to practitioners. To avoid misleading users, template providers should explicitly signal inactivity, for example, by archiving the repository or clearly documenting its status. In some cases, where inactivity was mentioned in the README, such information is less visible and may be overlooked compared to GitHub’s native archival mechanisms.

P3: Multiple templates in a single repository: GitHub repositories are typically designed as single entities with a clear and focused goal, and templates are no exception. However, multiple variants of a template, often targeting different programming languages or ecosystems, are grouped within the same repository, distributed across folders, or even separate branches. Such a decision increases complexity and can negatively impact discoverability, maintenance, and reuse. 

P4: Mixing technology and template concerns: Repositories hosting technologies on GitHub typically serve as primary points for implementation, documentation, issue tracking, and releases. Templates, however, address a different concern: providing reusable scaffolding rather than serving as the primary implementation artifact. Technology implementations and their corresponding templates should not be colocated within the same repository. Such a pattern can be confusing for practitioners and may hinder long-term evolution by mixing product-level changes with template-specific concerns. Instead, practitioners should advertise their templates as part of their documentation.

P5: Empty templates and missing documentation: While templates are expected to provide reusable scaffolding, some templates are empty repositories; no files are available, only the structure of folders. Some templates lack basic documentation, including a repository description or a README file. Such templates provide insufficient guidance and offer limited support for practitioners aiming to understand, adapt, or extend them. 

""".strip()


# -----------------------------
# Helpers
# -----------------------------

def sanitize_project_name(url: str) -> str:
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"{owner}__{repo}".replace(".", "_")
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").strip("_")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def ensure_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Top-level output is not a JSON object.")
    for k, v in obj.items():
        if not isinstance(v, dict):
            raise ValueError(f"Value for key {k} is not an object.")
        if "adherence" not in v or "reasoning" not in v or "confidence" not in v:
            raise ValueError(f"Missing fields for key {k}. Got: {list(v.keys())}")
        if v["adherence"] not in ("Yes", "No"):
            raise ValueError(f'Invalid adherence for {k}: {v["adherence"]}')
        if not isinstance(v["reasoning"], list) or not all(isinstance(x, str) for x in v["reasoning"]):
            raise ValueError(f"Invalid reasoning list for {k}.")
        if not isinstance(v["confidence"], int) or not (0 <= v["confidence"] <= 5):
            raise ValueError(f"Invalid confidence for {k}: {v['confidence']}")
    return obj


@dataclass
class LLMResult:
    project_url: str
    project_name: str
    parsed: Dict[str, Any]
    raw_text: str
    usage: Optional[Dict[str, Any]] = None


def call_gpt(project_url: str, client: OpenAI) -> Tuple[str, Optional[Dict[str, Any]]]:

    user_prompt = (
        f"{GUIDELINES_AND_PITFALLS_TEXT}\n\n"
        f"Project: {project_url}\n\n"
        "You MUST actively search and inspect the GitHub repository pages "
        "(README, workflows, releases, issues, topics, etc.) before answering. "
        "If you cannot retrieve content, explicitly state so.\n\n"
    )

    response = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "system",
                "content": (
                    "You must output ONLY valid JSON (a single object). "
                    "No markdown, no commentary, no code fences. "
                    "Follow the requested schema exactly."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    text = response.output_text
    usage = response.usage.model_dump() if response.usage else None
    return text, usage


def retry_call(project_url: str, client: OpenAI, retries: int = 3, backoff_s: float = 2.0) -> LLMResult:
    project_name = sanitize_project_name(project_url)

    for attempt in range(1, retries + 1):
        try:
            raw_text, usage = call_gpt(project_url, client)
            parsed = ensure_schema(extract_json_object(raw_text))
            return LLMResult(
                project_url=project_url,
                project_name=project_name,
                parsed=parsed,
                raw_text=raw_text,
                usage=usage,
            )
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff_s * attempt)
            else:
                raise RuntimeError(f"Failed for {project_url} after {retries} attempts: {e}") from e

def flatten_to_rows(result: LLMResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item_id, obj in result.parsed.items():
        rows.append(
            {
                "model": MODEL,
                "project_name": result.project_name,
                "project_url": result.project_url,
                "item_id": item_id,
                "item_type": "guideline" if item_id.startswith("G") else ("pitfall" if item_id.startswith("P") else "unknown"),
                "adherence": obj["adherence"],
                "confidence": obj["confidence"],
                "reasoning": " | ".join(obj["reasoning"]),
            }
        )
    return rows


def load_existing_pairs(csv_path: Path) -> Set[Tuple[str, str]]:
    existing: Set[Tuple[str, str]] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return existing

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return existing
        for row in reader:
            m = (row.get("model") or "").strip()
            u = (row.get("project_url") or "").strip()
            if m and u:
                existing.add((m, u))
    return existing


def append_rows(csv_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    client = OpenAI(api_key=config.GPT_KEY)

    model_folder = sanitize_model_name(MODEL)
    model_out_dir = GLOBAL_OUT_DIR / model_folder
    json_dir = model_out_dir / "json"
    raw_dir = model_out_dir / "raw"

    GLOBAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "project_name", "project_url", "item_id", "item_type", "adherence", "confidence", "reasoning"]
    existing_pairs = load_existing_pairs(GLOBAL_CSV_PATH) if DEDUPE else set()

    for i, project_url in enumerate(PROJECTS, start=1):
        if DEDUPE and (MODEL, project_url) in existing_pairs:
            print(f"[{i}/{len(PROJECTS)}] Skipping: {project_url}")
            continue

        print(f"[{i}/{len(PROJECTS)}] Evaluating {project_url} with {MODEL} ...")
        res = retry_call(project_url, client=client)

        json_path = json_dir / f"{res.project_name}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(res.parsed, f, ensure_ascii=False, indent=2)

        raw_path = raw_dir / f"{res.project_name}.txt"
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(res.raw_text.strip() + "\n")

        rows = flatten_to_rows(res)
        append_rows(GLOBAL_CSV_PATH, rows, fieldnames)

        if DEDUPE:
            existing_pairs.add((MODEL, project_url))

        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    main()