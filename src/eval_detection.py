# src/eval_detection.py
import os, json
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def _normalize_line_spans(raw: Any) -> List[List[int]]:
    """Accepts [[s,e], ...], [{'start':s,'end':e}], or nested lists/dicts."""
    def _add(out, a, b):
        try:
            ia, ib = int(a), int(b)
            if ia <= ib:
                out.append([ia, ib])
        except Exception:
            pass

    spans: List[List[int]] = []
    if isinstance(raw, dict):
        raw = (raw.get("line_spans") or raw.get("spans") or raw.get("ranges")
               or raw.get("lines") or raw.get("lineRanges") or [])

    if not isinstance(raw, (list, tuple)):
        return spans

    for item in raw:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and isinstance(item[0], (int, str)) and isinstance(item[1], (int, str)):
                _add(spans, item[0], item[1]); continue
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                    _add(spans, sub[0], sub[1])
            continue
        if isinstance(item, dict):
            s = (item.get("start") or item.get("s") or item.get("from") or
                 item.get("line_start") or item.get("lineStart"))
            e = (item.get("end")   or item.get("e") or item.get("to")   or
                 item.get("line_end")   or item.get("lineEnd"))
            if s is not None and e is not None:
                _add(spans, s, e)
    return spans

def _merge_spans(spans: List[List[int]]) -> List[List[int]]:
    if not spans:
        return []
    spans = sorted((int(s), int(e)) for s, e in spans)
    out: List[List[int]] = []
    for s, e in spans:
        if not out or s > out[-1][1] + 1:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def _overlap(a: List[int], b: List[int]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union

def _load_mutations(mutants_dir: str) -> Dict[str, List[List[int]]]:
    """Return GT map: { file_path: [[s,e], ...] } from data/mutated/.../mutated_files.json."""
    path = os.path.join(mutants_dir, "mutated_files.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept both top-level list and dict variants.
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("files", "mutants", "items", "records"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            # fallback: direct {file: spans}
            if all(isinstance(v, list) for v in data.values()):
                return {k: _merge_spans(_normalize_line_spans(v)) for k, v in data.items()}
            records = []
    else:
        records = []

    gt: Dict[str, List[List[int]]] = defaultdict(list)
    for rec in records:
        if not isinstance(rec, dict):
            continue
        f = rec.get("file") or rec.get("path") or rec.get("relpath") or rec.get("filename")
        spans = _normalize_line_spans(rec.get("line_spans") or rec.get("spans") or
                                      rec.get("ranges") or rec.get("lines") or [])
        # also accept edit-style entries
        for ed in (rec.get("mutations") or rec.get("edits") or []):
            if isinstance(ed, dict):
                s = ed.get("start") or ed.get("line_start")
                e = ed.get("end")   or ed.get("line_end")
                if s is not None and e is not None:
                    try: spans.append([int(s), int(e)])
                    except Exception: pass
        if f and spans:
            gt[f].extend(spans)

    for k in list(gt.keys()):
        gt[k] = _merge_spans(gt[k])
    return gt

def evaluate_detection(detection_json: Any, mutants_dir: str, iou_thresh: float = 0.2):
    gt = _load_mutations(mutants_dir)

    # Normalize predictions
    pred: Dict[str, List[List[int]]] = defaultdict(list)
    findings = []
    if isinstance(detection_json, dict):
        findings = detection_json.get("findings") or detection_json.get("detections") or []
    elif isinstance(detection_json, list):
        findings = detection_json

    for item in findings:
        if not isinstance(item, dict):
            continue
        f = item.get("file") or item.get("path") or item.get("filename")
        spans = _normalize_line_spans(item.get("line_spans") or item.get("spans") or
                                      item.get("ranges") or item.get("lines") or [])
        if f and spans:
            pred[f].extend(spans)

    tp = fp = fn = 0
    for fpath, gts in gt.items():
        used = [False] * len(gts)
        for p in pred.get(fpath, []):
            hit = False
            for i, g in enumerate(gts):
                if not used[i] and _overlap(p, g) >= iou_thresh:
                    used[i] = True
                    tp += 1
                    hit = True
                    break
            if not hit:
                fp += 1
        fn += used.count(False)

    # predictions for files with no GT are false positives
    for fpath, preds in pred.items():
        if fpath not in gt:
            fp += len(preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
