from config import SAMPLE_CODE, SAMPLE_BUG_REPORT, README_PATH
import json, requests

OLLAMA_API = "http://localhost:11434/api/chat"

def analyze_with_llm(model="mistral"):
    def safe_read(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    code = safe_read(SAMPLE_CODE)
    readme = safe_read(README_PATH)
    bug_report = safe_read(SAMPLE_BUG_REPORT)

    prompt = (
        "Context from README:\n" + readme + "\n\n"
        "Bug Report:\n" + json.dumps(bug_report, indent=2) + "\n\n"
        "Analyze and fix the following code:\n\n" + code
    )

    response = requests.post(OLLAMA_API, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    print(response.json()["message"]["content"])
