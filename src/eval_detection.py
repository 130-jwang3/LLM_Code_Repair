# src/eval_detection.py
import os, json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ---------- small helpers ----------

def _normalize_line_spans(raw: Any) -> List[List[int]]:
    """Accepts [[s,e], ...], dicts, or nested lists; returns [[s,e], ...] (ints)."""
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
                 item.get("line_start") or item.get("lineStart") or item.get("lineno"))
            e = (item.get("end")   or item.get("e") or item.get("to")   or
                 item.get("line_end")   or item.get("lineEnd")   or item.get("end_lineno") or s)
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

def _normpath(p: str | None) -> str | None:
    if not p:
        return None
    # normalize and unify separators; case-insensitive on Windows
    p2 = os.path.normpath(p).replace("\\", "/")
    # lower to avoid case mismatches on Windows
    return p2.lower()

def _pick_file_key(rec: dict, mutants_dir: str) -> str | None:
    """
    Prefer rel_path; fallback to file/path/relpath/filename; else derive from dst_path.
    Always return normalized forward-slash + lower case.
    """
    for k in ("rel_path", "file", "path", "relpath", "filename"):
        if rec.get(k):
            return _normpath(rec[k])

    dst = rec.get("dst_path") or rec.get("dest_path")
    if dst:
        try:
            rel = os.path.relpath(dst, mutants_dir)
        except Exception:
            rel = dst
        return _normpath(rel)

    src = rec.get("src_path")
    if src:
        # last-resort: use src relative to raw repo (still works if model also used same rel)
        return _normpath(os.path.basename(src)) if not mutants_dir else _normpath(src)

    return None

# ---------- load GT from mutated_files.json ----------

def _load_mutations(mutants_dir: str) -> Dict[str, List[List[int]]]:
    """
    Return GT map: { file_rel_path: [[s,e], ...] }.
    Supports your mutator format:
      - top-level list of records
      - each record may have 'rel_path' and 'mutations': [{'lineno', 'end_lineno', ...}, ...]
      - ignore 'copied' entries; use only records where action == 'mutated' (if present)
    Also supports older dict/list variants.
    """
    path = os.path.join(mutants_dir, "mutated_files.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to list of records
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("files", "mutants", "items", "records"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            # fallback: direct {file: spans-like}
            if all(isinstance(v, list) for v in data.values()):
                return { _normpath(k) or k : _merge_spans(_normalize_line_spans(v)) for k, v in data.items() }
            records = []
    else:
        records = []

    gt: Dict[str, List[List[int]]] = defaultdict(list)
    for rec in records:
        if not isinstance(rec, dict):
            continue
        # skip non-mutated files when an action is present and not "mutated"
        if "action" in rec and rec.get("action") != "mutated":
            continue

        f = _pick_file_key(rec, mutants_dir)
        if not f:
            continue

        spans: List[List[int]] = []

        # accept direct span arrays on the record
        spans += _normalize_line_spans(rec.get("line_spans") or rec.get("spans") or
                                       rec.get("ranges") or rec.get("lines") or [])

        # accept mutation/edit style entries
        for ed in (rec.get("mutations") or rec.get("edits") or []):
            if not isinstance(ed, dict):
                continue
            s = (ed.get("start") or ed.get("line_start") or ed.get("lineno"))
            e = (ed.get("end")   or ed.get("line_end")   or ed.get("end_lineno") or s)
            if s is not None and e is not None:
                try:
                    spans.append([int(s), int(e)])
                except Exception:
                    pass

        if spans:
            gt[f].extend(spans)

    # merge and clean
    for k in list(gt.keys()):
        gt[k] = _merge_spans(gt[k])

    return gt

# ---------- main evaluation ----------

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
        f = _normpath(f)
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
