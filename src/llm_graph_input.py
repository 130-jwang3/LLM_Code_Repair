from config import SAMPLE_GRAPH, SAMPLE_COVERAGE, SAMPLE_BUG_REPORT, README_PATH
import json, requests

OLLAMA_API = "http://localhost:11434/api/chat"

def analyze_with_llm(model="mistral"):
    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def safe_read(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    graph = load_json(SAMPLE_GRAPH)
    coverage = load_json(SAMPLE_COVERAGE)
    bug_report = load_json(SAMPLE_BUG_REPORT)
    readme = safe_read(README_PATH)

    prompt = (
        "Context from README:\n" + readme + "\n\n"
        "Bug Report:\n" + json.dumps(bug_report, indent=2) + "\n\n"
        "Coverage:\n" + json.dumps(coverage, indent=2) + "\n\n"
        "AST/Graph:\n" + json.dumps(graph, indent=2)
    )

    response = requests.post(OLLAMA_API, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    print(response.json()["message"]["content"])
