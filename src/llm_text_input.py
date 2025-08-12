import os
import json
import yaml

from .config import PROMPT_PREFIX_TEXT
from .llm_common import post_ollama, extract_first_json
from .schemas import is_detection, is_repair

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
    model: str = "mistral",
    *,
    # NEW: pass both original and mutated bundles
    text_path_orig: str | None = None,
    text_path_mut: str | None = None,
    # Back-compat (if someone passes just text_path, treat it as original)
    text_path: str | None = None,
    coverage_path: str | None = None,
    bug_reports=None,
):
    """
    Run TEXT-mode analysis with ORIGINAL + MUTATED context.

    Expected LLM output (JSON-only):
      - detection: {"findings":[{"file": "...", "line_spans": [[s,e],...], "confidence": 0.x}, ...]}
      - (optional) repair: {"diff": "<unified diff touching the findings>"}
    """
    prefix = load_prompt_sections(PROMPT_PREFIX_TEXT)
    coverage_data = load_coverage(coverage_path)

    # Load ORIGINAL (prefer text_path_orig; fall back to legacy text_path)
    orig_bundle = load_json_safe(text_path_orig or text_path) if (text_path_orig or text_path) else {}
    orig_tree = orig_bundle.get("file_tree", "")
    orig_files = orig_bundle.get("files", [])

    # Load MUTATED
    mut_bundle = load_json_safe(text_path_mut) if text_path_mut else {}
    mut_tree = mut_bundle.get("file_tree", "")
    mut_files = mut_bundle.get("files", [])

    # Build prompt
    parts = [
        prefix.get("task_intro", ""),
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        # Make clear the task is DETECT on mutated vs original, then optionally REPAIR:
        prefix.get("instructions", ""),
        prefix.get("output_format", ""),
        "\nCOVERAGE (context only):\n" + json.dumps(coverage_data, indent=2),
        "\nBUG REPORTS (context only):\n" + json.dumps(bug_reports or [], indent=2),
        "\n=== ORIGINAL REPO (TEXT) ===",
        "FILE TREE:\n" + (orig_tree or "<empty>"),
        "\nFILES:\n" + "\n".join(
            f"\n--- {f.get('path')} ---\n{f.get('content','')}" for f in orig_files
        ),
        "\n=== MUTATED REPO (TEXT) ===",
        "FILE TREE:\n" + (mut_tree or "<empty>"),
        "\nFILES:\n" + "\n".join(
            f"\n--- {f.get('path')} ---\n{f.get('content','')}" for f in mut_files
        ),
        "\nReturn a single JSON object ONLY (no markdown).",
    ]
    prompt = "\n".join(parts)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix.get("system", "Respond with JSON only.")},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.95},
    }

    code, resp, err = post_ollama(OLLAMA_API, payload, timeout_s=120, retries=4)
    if not resp:
        raise RuntimeError(f"LLM call failed (text mode): {err or code}")

    content = (resp.get("message") or {}).get("content", "")
    parsed = extract_first_json(content)

    result = {"raw": content}
    if parsed:
        ok_det, why_det = is_detection(parsed)
        ok_rep, why_rep = is_repair(parsed)
        if ok_det:
            result["detection"] = parsed
        if ok_rep:
            result["repair"] = parsed
        if not ok_det and not ok_rep:
            result["_noncompliant_json"] = {"json": parsed, "reason": {"det": why_det, "rep": why_rep}}
    else:
        result["_noncompliant_text"] = content[:5000]
    return result
