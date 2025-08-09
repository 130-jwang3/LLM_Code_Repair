import os
import json
import requests
import yaml

from .config import PROMPT_PREFIX_TEXT

OLLAMA_API = "http://localhost:11434/api/chat"

def load_prompt_sections(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load prompt YAML: {e}")
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

def gather_project_context(repo_path):
    """
    Collects non-code context files.
    Currently: .md, .toml, .yml, .yaml, .ini
    Returns a dict {rel_path: content}.

    TODO (striding): When inputs are large, chunk these contents with overlap
    in main.py before sending to the LLM.
    """
    context_files = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".md", ".toml", ".yml", ".yaml", ".ini")):
                full_path = os.path.join(root, file)
                context_files[os.path.relpath(full_path, repo_path)] = read_file_safe(full_path)
    return context_files

def load_coverage(coverage_path):
    if coverage_path and os.path.exists(coverage_path):
        try:
            with open(coverage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load coverage JSON: {e}")
            return {}
    return {}

def analyze_with_llm(
    model="mistral",
    *,
    text_path=None,
    repo_path=None,
    coverage_path=None,
    bug_reports=None
):
    """
    Run text-based LLM analysis.

    Preferred usage: pass text_path (bundle from scripts/code_to_text.py).
    Optionally also pass repo_path to include docs/configs (.md/.toml/.yml/.ini).
    """
    if not text_path and not repo_path:
        raise ValueError("Provide at least text_path (preferred) or repo_path.")

    prefix = load_prompt_sections(PROMPT_PREFIX_TEXT)
    coverage_data = load_coverage(coverage_path)

    prompt_parts = [
        prefix.get("task_intro", ""),
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        prefix.get("instructions", ""),
        prefix.get("output_format", ""),
        "\nCoverage Information:\n" + json.dumps(coverage_data, indent=2),
        "\nBug Reports:\n" + json.dumps(bug_reports or [], indent=2),
    ]

    # Prefer the single-file text bundle (Python files)
    if text_path:
        bundle = load_json_safe(text_path)
        file_tree = bundle.get("file_tree", "")
        files = bundle.get("files", [])

        prompt_parts.append("\nRepository File Tree (Python files):\n" + file_tree)

        # TODO (striding): Split these file contents into overlapping chunks in main.py
        prompt_parts.append("\nProject Python Files (from bundle):")
        for entry in files:
            path = entry.get("path", "")
            content = entry.get("content", "")
            prompt_parts.append(f"\n--- {path} ---\n{content}")

    # Optionally add non-code docs/configs from repo (if provided)
    if repo_path:
        docs = gather_project_context(repo_path)
        if docs:
            prompt_parts.append("\nDocumentation & Config Files:")
            # TODO (striding): Also chunk these if large
            for rel_path, content in docs.items():
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
