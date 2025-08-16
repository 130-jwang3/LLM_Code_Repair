import os
import json
import time
import re
from typing import Optional

import requests
from .runlog import RunLogger

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")


def chat_or_generate(
    *,
    model: str,
    system_text: str,
    user_text: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    timeout_s: int = 240,          # was 120
    retries: int = 5,              # was 3
    num_ctx: int | None = None,
    enforce_json: bool = False,    # request JSON mode if supported
    logger: Optional[RunLogger] = None,
    log_tag: str = "llm"
):
    """
    Try /api/chat first; if unsupported, fall back to /api/generate.
    If enforce_json=True, we pass options.format="json" and auto-fallback if the model rejects it.
    Returns: (content_text|None, error|None)
    """
    def _opts(json_mode: bool = False):
        o = {"temperature": temperature, "top_p": top_p}
        if num_ctx:
            o["num_ctx"] = num_ctx
        if json_mode:
            o["format"] = "json"
        return o

    # 1) chat
    chat_url = f"{OLLAMA_BASE}/api/chat"
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "options": _opts(enforce_json),
    }
    if logger:
        logger.write_json(f"{log_tag}_chat_request", {"url": chat_url, "payload_head": str(chat_payload)[:1000]})

    code, resp, err = post_ollama(chat_url, chat_payload, timeout_s=timeout_s, retries=retries)
    if (not resp or not isinstance(resp, dict)) and enforce_json:
        # retry once without JSON mode
        chat_payload["options"] = _opts(False)
        code, resp, err = post_ollama(chat_url, chat_payload, timeout_s=timeout_s, retries=retries)
    if logger:
        logger.write_json(
            f"{log_tag}_chat_response",
            {"status": code, "error": err, "resp_keys": list(resp.keys()) if isinstance(resp, dict) else None}
        )

    if resp and isinstance(resp, dict):
        content = (resp.get("message") or {}).get("content")
        if content:
            return content, None

    # 2) generate fallback
    gen_url = f"{OLLAMA_BASE}/api/generate"
    gen_payload = {
        "model": model,
        "prompt": f"{system_text}\n\n{user_text}",
        "stream": False,
        "options": _opts(enforce_json),
    }
    if logger:
        logger.write_json(f"{log_tag}_generate_request", {"url": gen_url, "payload_head": str(gen_payload)[:1000]})

    code, resp, err2 = post_ollama(gen_url, gen_payload, timeout_s=timeout_s, retries=retries)
    if (not resp or not isinstance(resp, dict)) and enforce_json:
        gen_payload["options"] = _opts(False)
        code, resp, err2 = post_ollama(gen_url, gen_payload, timeout_s=timeout_s, retries=retries)
    if logger:
        logger.write_json(
            f"{log_tag}_generate_response",
            {"status": code, "error": err2, "resp_keys": list(resp.keys()) if isinstance(resp, dict) else None}
        )

    if resp and isinstance(resp, dict):
        content = resp.get("response")
        if content:
            return content, None

    return None, err2 or err or "no response"


def post_ollama(url, payload, timeout_s=120, retries=3, backoff=0.8):
    """
    Robust POST with simple retries/backoff.
    Returns: (status_code|None, response_json|None, error_message|None)
    """
    err = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            if r.ok:
                try:
                    return r.status_code, r.json(), None
                except Exception as e:
                    return r.status_code, None, f"json decode error: {e}"
            err = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as e:
            err = str(e)
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    return None, None, err


# -----------------------
# JSON extraction helpers
# -----------------------

def _find_balanced_block(s: str, start_char: str, end_char: str, start_pos: int) -> int:
    n = len(s)
    stack = 0
    in_str = False
    esc = False
    i = start_pos
    if i >= n or s[i] != start_char:
        return -1
    while i < n:
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == start_char:
                stack += 1
            elif c == end_char:
                stack -= 1
                if stack == 0:
                    return i
        i += 1
    return -1


def _coerce_json_like(s: str) -> str:
    # strip code fences
    s = re.sub(r"^```(?:json)?\s*|```$", "", s.strip(), flags=re.I | re.M)
    # remove // and # comments
    s = re.sub(r"(?m)//.*?$", "", s)
    s = re.sub(r"(?m)^\s*#.*?$", "", s)
    # remove /* ... */ comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    # replace smart quotes
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    # keys in single quotes -> double
    s = re.sub(r"(?m)'([A-Za-z0-9_]+)'\s*:", r'"\1":', s)
    # Python booleans/null -> JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # kill trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # "1-2" -> [1,2]
    s = re.sub(r'"(\d+)\s*-\s*(\d+)"', r'[\1,\2]', s)
    # remove bare ellipses
    s = re.sub(r"\.\.\.", "", s)
    return s


def _try_load_json(fragment: str):
    try:
        return json.loads(fragment)
    except Exception:
        try:
            return json.loads(_coerce_json_like(fragment))
        except Exception:
            return None


def extract_first_json(text: str):
    if not text:
        return None
    # fenced block first
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        candidate = m.group(1).strip()
        obj = _try_load_json(candidate)
        if obj is not None:
            return obj
    # object
    start = text.find("{")
    while start != -1:
        end = _find_balanced_block(text, "{", "}", start)
        if end != -1:
            fragment = text[start:end + 1]
            obj = _try_load_json(fragment)
            if obj is not None:
                return obj
        start = text.find("{", start + 1)
    # array
    start = text.find("[")
    while start != -1:
        end = _find_balanced_block(text, "[", "]", start)
        if end != -1:
            fragment = text[start:end + 1]
            obj = _try_load_json(fragment)
            if obj is not None:
                return obj
        start = text.find("[", start + 1)
    return None
