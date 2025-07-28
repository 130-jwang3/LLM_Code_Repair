from config import SAMPLE_GRAPH, SAMPLE_COVERAGE, SAMPLE_BUG_REPORT, PROMPT_PREFIX_GRAPH
import json, requests, yaml

OLLAMA_API = "http://localhost:11434/api/chat"

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON from {path}: {e}")
        return {}

def load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load YAML from {path}: {e}")
        return {}

def analyze_with_llm(model="mistral"):
    graph = load_json(SAMPLE_GRAPH)
    coverage = load_json(SAMPLE_COVERAGE)
    bug_report = load_json(SAMPLE_BUG_REPORT)
    prefix = load_yaml(PROMPT_PREFIX_GRAPH)

    prompt = "\n\n".join([
        prefix.get("task_intro", ""),
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        prefix.get("instructions", ""),
        "\nGraph-based Representation:\n" + json.dumps(graph, indent=2),
        "\nCoverage Information:\n" + json.dumps(coverage, indent=2),
        "\nBug Report:\n" + json.dumps(bug_report, indent=2)
    ])

    response = requests.post(OLLAMA_API, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    print(response.json()["message"]["content"])
