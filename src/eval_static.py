# src/eval_static.py
import os, json, ast
from typing import Dict, Any, List, Tuple

def load_mutation_log(mutants_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(mutants_dir, "mutated_files.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def _span_overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def _guess_patch_spans_from_diff_text(diff_text: str) -> Dict[str, List[Tuple[int,int]]]:
    """
    Parse unified diff hunks → { "path.py": [(new_start,new_end), ...], ... }
    """
    spans: Dict[str, List[Tuple[int,int]]] = {}
    cur: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/") or line.startswith("+++ "):
            cur = line.split("+++ ",1)[1].strip()
            if cur.startswith("b/"):
                cur = cur[2:]
            spans.setdefault(cur, [])
        elif line.startswith("@@"):
            # @@ -old_start,old_count +new_start,new_count @@
            try:
                hunk = line.split("@@")[1].strip()
                parts = hunk.split()
                plus = [p for p in parts if p.startswith("+")][0][1:]
                if "," in plus:
                    ns, nc = plus.split(",",1); ns, nc = int(ns), int(nc)
                else:
                    ns, nc = int(plus), 1
                start = ns
                end = ns + max(nc-1, 0)
                if cur:
                    spans[cur].append((start, end))
            except Exception:
                pass
    return spans

def evaluate_patch_against_mutations(
    sandbox_repo: str,
    mutants_dir: str,
    predicted_diff: str
) -> Dict[str, Any]:
    """
    Static evaluation:
      - syntax_ok rate for mutated files after patch
      - localization overlap (diff hunks vs mutation spans)
      - mutation-reversal heuristics (Eq/NotEq, Add/Sub, Mult/Div, And/Or)
    """
    mut_log = load_mutation_log(mutants_dir)
    diff_spans = _guess_patch_spans_from_diff_text(predicted_diff)

    total_mut_files = 0
    touched_mut_files = 0
    syntax_ok_files = 0
    reversal_hits = 0
    reversal_total = 0

    REV_PAIRS = {
        "Eq→NotEq": ("Eq", "NotEq"), "NotEq→Eq": ("NotEq", "Eq"),
        "Add→Sub": ("Add", "Sub"),   "Sub→Add": ("Sub", "Add"),
        "Mult→Div": ("Mult", "Div"), "Div→Mult": ("Div", "Mult"),
        "And→Or": ("And", "Or"),     "Or→And": ("Or", "And"),
    }
    token_map = {
        "Eq": "==", "NotEq": "!=", "Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
        "And": " and ", "Or": " or ",
    }

    for entry in mut_log:
        if entry.get("action") != "mutated" or not entry.get("ok"):
            continue
        rel = entry.get("rel_path")
        if not rel or not rel.endswith(".py"):
            continue

        total_mut_files += 1
        abs_path = os.path.join(sandbox_repo, rel)
        code = _read(abs_path)

        if _ast_ok(code):
            syntax_ok_files += 1

        # localization overlap
        mut_spans = []
        for m in entry.get("mutations", []):
            s = m.get("lineno"); e = m.get("end_lineno") or s
            if s:
                mut_spans.append((int(s), int(e)))
        diff_file_spans = diff_spans.get(rel, [])
        if diff_file_spans and mut_spans:
            if any(_span_overlap(a, b) for a in mut_spans for b in diff_file_spans):
                touched_mut_files += 1

        # reversal heuristic
        lines = code.splitlines()
        for m in entry.get("mutations", []):
            change = m.get("change", "")
            if change in REV_PAIRS and m.get("lineno"):
                reversal_total += 1
                want_left, _ = REV_PAIRS[change]  # we want the "original" operator restored
                wanted_token = token_map.get(want_left)
                ln = int(m["lineno"])
                line = lines[ln-1] if 1 <= ln <= len(lines) else ""
                if wanted_token and (wanted_token in line):
                    reversal_hits += 1

    return {
        "total_mut_files": total_mut_files,
        "touched_mut_files": touched_mut_files,
        "touched_mut_files_rate": (touched_mut_files / total_mut_files) if total_mut_files else 0.0,
        "syntax_ok_files": syntax_ok_files,
        "syntax_ok_rate": (syntax_ok_files / total_mut_files) if total_mut_files else 0.0,
        "reversal_hits": reversal_hits,
        "reversal_total": reversal_total,
        "reversal_rate": (reversal_hits / reversal_total) if reversal_total else 0.0,
    }
