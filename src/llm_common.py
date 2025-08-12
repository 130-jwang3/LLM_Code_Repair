# src/llm_common.py
import json, re, time, requests
from typing import Any, Optional, Tuple

# crude but effective: find first top-level JSON object
JSON_BLOCK = re.compile(r"\{(?:[^{}]|(?R))*\}", re.S)

def extract_first_json(text: str) -> Optional[dict]:
    # try whole
    try:
        return json.loads(text)
    except Exception:
        pass
    # try first balanced block
    m = JSON_BLOCK.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def post_ollama(url: str, payload: dict, *, timeout_s=60, retries=3, backoff=1.6) -> Tuple[int, Any, str]:
    last_err = ""
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            if r.status_code == 200:
                return r.status_code, r.json(), ""
            last_err = f"HTTP {r.status_code}: {r.text[:400]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff ** i)
    return 0, None, last_err
