from config import SAMPLE_CODE, SAMPLE_BUG_REPORT, README_PATH, PROMPT_PREFIX_TEXT
import json, requests, yaml

OLLAMA_API = "http://localhost:11434/api/chat"

def load_prompt_sections(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load prompt YAML: {e}")
        return {}

def safe_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def analyze_with_llm(model="mistral"):
    # Load files
    code = safe_read(SAMPLE_CODE)
    readme = safe_read(README_PATH)
    bug_report = safe_read(SAMPLE_BUG_REPORT)

    # Load structured prompt
    prefix = load_prompt_sections(PROMPT_PREFIX_TEXT)

    # Compose final prompt
    prompt = "\n\n".join([
        prefix.get("task_intro", ""),
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        prefix.get("instructions", ""),
        "\nContext from README:\n" + readme,
        "\nBug Report:\n" + json.dumps(bug_report, indent=2),
        "\nCode to analyze:\n" + code
    ])

    # Call the LLM
    response = requests.post(OLLAMA_API, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    print(response.json()["message"]["content"])
