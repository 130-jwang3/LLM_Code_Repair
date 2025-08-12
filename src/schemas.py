# src/schemas.py
from typing import Any, Dict, List, Tuple

DETECTION_KEYS = {"findings"}  # list[Finding]
Localization = Dict[str, Any]
Repair = Dict[str, Any]

LOCALIZATION_KEYS = {"file", "line_spans"}  # optional: "confidence"
REPAIR_KEYS = {"diff"}  # optional: "notes"

def is_localization(obj: dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    missing = LOCALIZATION_KEYS - set(obj.keys())
    if missing:
        return False, f"missing keys: {missing}"
    if not isinstance(obj["file"], str):
        return False, "file must be str"
    spans = obj["line_spans"]
    if not isinstance(spans, list) or not all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(x, int) for x in p) and p[0] <= p[1]
        for p in spans
    ):
        return False, "line_spans must be [[start,end],...] with ints"
    return True, ""

def is_repair(obj: dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    missing = REPAIR_KEYS - set(obj.keys())
    if missing:
        return False, f"missing keys: {missing}"
    if not isinstance(obj["diff"], str) or not obj["diff"].strip():
        return False, "diff must be non-empty string (unified diff)"
    return True, ""

def is_detection(obj: dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict): return False, "not a dict"
    if "findings" not in obj or not isinstance(obj["findings"], list):
        return False, "missing findings list"
    for f in obj["findings"]:
        if not isinstance(f, dict): return False, "finding not dict"
        if "file" not in f or not isinstance(f["file"], str): return False, "finding.file"
        spans = f.get("line_spans", [])
        if not isinstance(spans, list) or not all(
            isinstance(p, list) and len(p)==2 and all(isinstance(x,int) for x in p) and p[0] <= p[1]
            for p in spans
        ):
            return False, "finding.line_spans invalid"
    return True, ""
