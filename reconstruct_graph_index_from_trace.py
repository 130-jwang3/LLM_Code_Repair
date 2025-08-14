# tools/reconstruct_graph_index_from_trace.py
import json, re, sys, os
from collections import defaultdict

def coerce_json_like(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*|```$", "", s.strip(), flags=re.I | re.M)
    s = re.sub(r"(?m)//.*?$", "", s)                 # // comments
    s = re.sub(r"(?m)^\s*#.*?$", "", s)              # # comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)      # /* ... */ comments
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    s = re.sub(r"(?m)'([A-Za-z0-9_]+)'\s*:", r'"\1":', s)   # 'key': -> "key":
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)             # trailing commas
    s = re.sub(r'"(\d+)\s*-\s*(\d+)"', r'[\1,\2]', s)# "1-2" -> [1,2]
    s = re.sub(r"\.\.\.", "", s)                     # ellipses
    return s

def try_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(coerce_json_like(s))
        except Exception:
            return None

def find_balanced(text: str, start_char: str, end_char: str, start_pos: int) -> int:
    n = len(text); stack = 0; in_str = False; esc = False; i = start_pos
    if i >= n or text[i] != start_char: return -1
    while i < n:
        c = text[i]
        if in_str:
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': in_str = False
        else:
            if c == '"': in_str = True
            elif c == start_char: stack += 1
            elif c == end_char:
                stack -= 1
                if stack == 0: return i
        i += 1
    return -1

def extract_first_json(text: str):
    if not text: return None
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        obj = try_load_json(m.group(1).strip())
        if obj is not None: return obj
    i = text.find("{")
    while i != -1:
        j = find_balanced(text, "{", "}", i)
        if j != -1:
            obj = try_load_json(text[i:j+1])
            if obj is not None: return obj
        i = text.find("{", i+1)
    i = text.find("[")
    while i != -1:
        j = find_balanced(text, "[", "]", i)
        if j != -1:
            obj = try_load_json(text[i:j+1])
            if obj is not None: return obj
        i = text.find("[", i+1)
    return None

def merge_spans(spans):
    if not spans: return []
    spans = sorted((int(s), int(e)) for s, e in spans)
    out = []
    for s, e in spans:
        if not out or s > out[-1][1] + 1:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def parse_file_and_lines(user_text: str):
    file_path = None; s_line = None; e_line = None
    for line in (user_text or "").splitlines():
        m1 = re.match(r"^\s*FILE:\s*(.+?)\s*$", line)
        if m1: file_path = m1.group(1).strip()
        m2 = re.match(r"^\s*ORIGINAL GRAPH CHUNK LINES:\s*(\d+)\s*-\s*(\d+)\s*$", line)
        if m2:
            s_line = int(m2.group(1)); e_line = int(m2.group(2))
    return file_path, s_line, e_line

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/reconstruct_graph_index_from_trace.py <trace.jsonl> <output_json_path>")
        sys.exit(1)

    trace_path = sys.argv[1]
    out_path   = sys.argv[2]

    summary_map = defaultdict(lambda: {
        "symbols": [], "intents": [], "invariants": [], "risky_spans": []
    })

    attempts = hits = 0
    files_seen = set()

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Only use graph indexing records (prefix like gindex_*)
            pref = rec.get("prefix") or rec.get("tag") or rec.get("name") or ""
            if not isinstance(pref, str) or not pref.startswith("gindex_"):
                continue

            attempts += 1
            user_text = rec.get("user_text") or rec.get("user") or rec.get("prompt") or ""
            resp_text = rec.get("response_text") or rec.get("response") or rec.get("reply") or ""

            file_path, s_line, e_line = parse_file_and_lines(user_text)
            if not file_path:
                continue

            parsed = extract_first_json(resp_text)
            if not (isinstance(parsed, dict) and parsed.get("file")):
                continue
            if parsed.get("file") not in (None, "", file_path):
                continue

            hits += 1
            files_seen.add(file_path)

            slot = summary_map[file_path]
            for key in ("symbols", "intents", "invariants"):
                vals = parsed.get(key) or []
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, str) and v and v not in slot[key]:
                            slot[key].append(v)

            # risky_spans mapping: chunk-local -> file-global using s_line
            for se in (parsed.get("risky_spans") or []):
                if isinstance(se, list) and len(se) >= 2:
                    try:
                        loc_s = int(se[0]); loc_e = int(se[1])
                    except Exception:
                        continue
                    reason = se[2] if len(se) >= 3 and isinstance(se[2], str) else ""
                    if isinstance(s_line, int):
                        gs = s_line + (loc_s - 1)
                        ge = s_line + (loc_e - 1)
                    else:
                        gs = loc_s; ge = loc_e
                    if gs <= ge:
                        slot["risky_spans"].append([gs, ge, reason])

    # compact/merge & trim
    for fpath, slot in summary_map.items():
        slot["symbols"]    = slot["symbols"][:50]
        slot["intents"]    = slot["intents"][:50]
        slot["invariants"] = slot["invariants"][:50]
        merged = merge_spans([[s, e] for s, e, _ in slot["risky_spans"]])
        slot["risky_spans"] = [[s, e, ""]] * 0  # clear
        slot["risky_spans"] = [[s, e, ""] for s, e in merged[:50]]

    stats = {
        "orig_chunks": attempts,
        "attempts": attempts,
        "hits": hits,
        "files": len(files_seen),
        "duration_sec": 0.0
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump({"__ok__": True, "summary_map": dict(summary_map), "stats": stats}, out, indent=2)
    print(f"Wrote cache: {out_path}")
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()
