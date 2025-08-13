import os, json
from collections import defaultdict

def _load_mutations(mutants_dir):
    path = os.path.join(mutants_dir, "mutated_files.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path, "r", encoding="utf-8"))
    # expected shape: { "files": [{"path": "...", "mutations": [{"start":..,"end":..}, ...]}], ... }
    gt = defaultdict(list)
    for f in data.get("files", []):
        p = f.get("path")
        for m in f.get("mutations", []):
            s, e = int(m.get("start", 0)), int(m.get("end", 0))
            if p and s and e and s <= e:
                gt[p].append([s, e])
    return gt

def _overlap(a, b):
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union

def evaluate_detection(detection_json, mutants_dir, iou_thresh=0.2):
    gt = _load_mutations(mutants_dir)
    pred = defaultdict(list)
    for item in detection_json.get("findings", []):
        f = item.get("file")
        for s, e in item.get("line_spans", []):
            try:
                s, e = int(s), int(e)
            except Exception:
                continue
            if f and s <= e:
                pred[f].append([s, e])

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

    # count predictions on files with no GT as FP
    for fpath, preds in pred.items():
        if fpath not in gt:
            fp += len(preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
