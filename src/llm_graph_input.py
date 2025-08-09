import os
import json
import requests
import yaml

from .config import PROMPT_PREFIX_GRAPH

OLLAMA_API = "http://localhost:11434/api/chat"

def load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load YAML from {path}: {e}")
        return {}

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON from {path}: {e}")
        return {}

def read_file_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def gather_docs_and_configs(repo_path):
    """
    Collects .md, .toml, .yml, .yaml, .ini files.
    TODO (striding): chunk large files with overlap in main.py before LLM.
    """
    files_data = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".md", ".toml", ".yml", ".yaml", ".ini")):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                files_data[rel_path] = read_file_safe(full_path)
    return files_data

def analyze_with_llm(
    model="mistral",
    graph_path=None,
    coverage_path=None,
    bug_reports=None,
    repo_path=None
):
    """
    Run graph-based LLM analysis with optional extra repo context (docs/config files).
    Preferred minimum inputs: graph_path (+ coverage, bug reports).
    """
    if not graph_path:
        raise ValueError("graph_path is required")

    prefix = load_yaml(PROMPT_PREFIX_GRAPH)
    graph_data = load_json_safe(graph_path)
    coverage_data = load_json_safe(coverage_path) if coverage_path else {}

    prompt_parts = [
        prefix.get("task_intro", ""),
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        prefix.get("instructions", ""),
        prefix.get("output_format", ""),
        # TODO (striding): chunk the graph JSON in main.py if very large.
        "\nGraph-based Representation:\n" + json.dumps(graph_data, indent=2),
        "\nCoverage Information:\n" + json.dumps(coverage_data, indent=2),
        "\nBug Reports:\n" + json.dumps(bug_reports or [], indent=2),
    ]

    # Optionally append docs/configs if repo_path provided
    if repo_path:
        docs_configs = gather_docs_and_configs(repo_path)
        if docs_configs:
            prompt_parts.append("\nDocumentation and Config Files:")
            # TODO (striding): chunk these too in main.py when large.
            for rel_path, content in docs_configs.items():
                prompt_parts.append(f"\n--- {rel_path} ---\n{content}")

    prompt = "\n\n".join(prompt_parts)

    response = requests.post(OLLAMA_API, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    content = response.json()["message"]["content"]
    print(content)
    return content
