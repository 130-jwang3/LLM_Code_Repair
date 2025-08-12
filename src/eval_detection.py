import os, json
from typing import List, Dict, Any, Tuple

def _overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def load_mutations(mutants_dir: str) -> Dict[str, List[Tuple[int,int]]]:
    """Returns {rel_path: [(s,e),...]} for mutated lines."""
    path = os.path.join(mutants_dir, "mutated_files.json")
    if not os.path.isfile(path): return {}
    data = json.load(open(path, "r", encoding="utf-8"))
    m: Dict[str, List[Tuple[int,int]]] = {}
    for e in data:
        if e.get("action")=="mutated" and e.get("ok") and e.get("rel_path","").endswith(".py"):
            spans=[]
            for mm in e.get("mutations", []):
                s = mm.get("lineno"); e2 = mm.get("end_lineno") or s
                if s: spans.append((int(s), int(e2)))
            if spans: m[e["rel_path"]] = spans
    return m

def evaluate_detection(detection_json: Dict[str, Any], mutants_dir: str) -> Dict[str, Any]:
    """Compute precision/recall/F1 by span overlap per file."""
    gt = load_mutations(mutants_dir)
    pred = detection_json.get("findings", []) if isinstance(detection_json, dict) else []

    # build per-file predicted spans
    pd: Dict[str, List[Tuple[int,int]]] = {}
    for f in pred:
        path = f.get("file")
        # normalize to repo-relative path used in mutated_files.json
        if path and (path in gt or True):
            spans = [(int(s), int(e)) for s,e in f.get("line_spans", []) if isinstance(s,int) and isinstance(e,int)]
            if spans:
                pd.setdefault(path, []).extend(spans)

    # count matches by any-overlap
    tp = fp = fn = 0
    # For each predicted span, if it overlaps any GT span in same file, count TP and mark GT span as used once
    used = {k: [False]*len(v) for k,v in gt.items()}
    for path, spans in pd.items():
        gsp = gt.get(path, [])
        for s,e in spans:
            hit = False
            for i, (gs, ge) in enumerate(gsp):
                if not used.get(path, [])[i] and _overlap((s,e),(gs,ge)):
                    used[path][i] = True
                    tp += 1; hit = True; break
            if not hit:
                fp += 1

    # FN = GT spans not matched
    for path, gsp in gt.items():
        for i in range(len(gsp)):
            if not used[path][i]:
                fn += 1

    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
