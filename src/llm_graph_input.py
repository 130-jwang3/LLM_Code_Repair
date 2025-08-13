# src/llm_graph_input.py
import os
import json
import yaml
import time
from pathlib import Path
from typing import Optional, Any, Dict, List

from .runlog import RunLogger
from .config import PROMPT_PREFIX_TEXT as PROMPT_PREFIX  # reuse same yaml for simplicity
from .llm_common import chat_or_generate, extract_first_json
from .input_splitter import split_ast


# ---------- helpers (shared logic) ----------

def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _load_json(path: Optional[str]) -> dict:
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:120]

def _fit(text: Optional[str], max_chars: int) -> str:
    # No truncation: always return full text
    return text or ""

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

def _normalize_line_spans(raw: Any) -> List[List[int]]:
    """
    Accepts [[s,e], ...], [{'start':s,'end':e}], or nested lists.
    Returns clean [[s,e], ...] with ints.
    """
    def _add(out, a, b):
        try:
            ia, ib = int(a), int(b)
            if ia <= ib:
                out.append([ia, ib])
        except Exception:
            pass

    spans: List[List[int]] = []
    if isinstance(raw, dict):
        raw = raw.get("line_spans") or raw.get("spans") or raw.get("ranges") or raw.get("lines") or []

    if not isinstance(raw, (list, tuple)):
        return spans

    for item in raw:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and isinstance(item[0], (int, str)) and isinstance(item[1], (int, str)):
                _add(spans, item[0], item[1]); continue
            flat = False
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                    _add(spans, sub[0], sub[1]); flat = True
            if flat:
                continue
        if isinstance(item, dict):
            s = item.get("start") or item.get("s") or item.get("from") or item.get("line_start") or item.get("lineStart")
            e = item.get("end")   or item.get("e") or item.get("to")   or item.get("line_end")   or item.get("lineEnd")
            if s is not None and e is not None:
                _add(spans, s, e)
    return spans


# ---------- Phase 1: Index ORIGINAL graph ----------

def _summarize_original_graph(
    *,
    model: str,
    graph_path_orig: str,
    chunk_size: int,
    coverage: dict,
    bug_reports: list,
    debug_dir: Optional[str],
    verbose: bool,
    max_chunks: Optional[int],
    logger: Optional[RunLogger],
):
    """
    Build per-file summaries of ORIGINAL graph chunks.
    Returns (summary_map, stats, cache_path)
      summary_map: { file: {"symbols":[], "intents":[], "invariants":[], "risky_spans":[[s,e,why], ...]} }
    """
    t0 = time.perf_counter()
    cache_path = None
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(debug_dir) / "original_graph_summaries.json")
        if os.path.exists(cache_path):
            try:
                cached = _load_json(cache_path)
                if isinstance(cached, dict) and cached.get("__ok__"):
                    if verbose: print(f"[GRAPH] using cached original summaries: {cache_path}")
                    return cached.get("summary_map", {}), cached.get("stats", {}), cache_path
            except Exception:
                pass

    prefix = _load_yaml(PROMPT_PREFIX)
    orig_chunks = split_ast(graph_json_path=graph_path_orig, chunk_size=chunk_size) or []

    system_lines = [
        prefix.get("task_intro", "") or "You are an expert software engineer.",
        "Build a compact structural understanding of the ORIGINAL codebase **from graph nodes**.",
        "Each chunk contains a list of nodes for a single file.",
        "Return JSON ONLY with the shape:",
        """{
  "file":"<path>",
  "section": <int>,
  "start_line": <int>,
  "end_line": <int>,
  "symbols": ["Key classes/functions/entities mentioned by these nodes"],
  "intents": ["What these nodes collectively intend to do"],
  "invariants": ["Important invariants/pre/post-conditions in these nodes"],
  "risky_spans": [[start,end,"why"], ...] // file-global lines likely to break
}""",
    ]
    system_text = "\n".join([s for s in system_lines if s]).strip()
    cov_hint = f"[coverage_present={bool(coverage)}]"
    bug_hint = f"[bug_reports={len(bug_reports)}]"

    attempts = hits = 0
    file_aggr: Dict[str, Dict[str, Any]] = {}

    for x in orig_chunks:
        if not (isinstance(x, dict) and "file" in x):
            continue
        file_path = x["file"]
        section   = x.get("section")
        s_line    = x.get("start_line")
        e_line    = x.get("end_line")
        content   = x.get("content", "")

        if not (section and content):
            continue

        if max_chunks is not None and attempts >= max_chunks:
            break

        user_lines = [
            f"FILE: {file_path}",
            f"ORIGINAL GRAPH CHUNK LINES: {s_line}-{e_line}",
            cov_hint, bug_hint,
            "=== GRAPH NODES (compact text) ===",
            _fit(content, 3500),
            "Return JSON only."
        ]
        user_text = "\n".join([u for u in user_lines if u])

        attempts += 1
        if verbose and attempts % 25 == 1:
            print(f"[GRAPH-INDEX] calling LLM… attempt={attempts}")

        out, err = chat_or_generate(
            model=model,
            system_text=system_text,
            user_text=user_text,
            temperature=0.2,
            top_p=0.95,
            num_ctx=chunk_size,
        )
        if logger and hits < 5:
            logger.dump_pair(
                prefix=f"gindex_{_sanitize_name(file_path)}_sec{section}",
                system_text=system_text, user_text=user_text, response_text=out or err or ""
            )
        if err or not out:
            if verbose:
                print(f"[GRAPH-INDEX] LLM error/empty content on attempt {attempts}: {err}")
            continue

        parsed = extract_first_json(out) or {}
        if not (isinstance(parsed, dict) and parsed.get("file")):
            continue
        if parsed.get("file") not in (None, "", file_path):
            continue

        hits += 1
        rec = file_aggr.setdefault(file_path, {
            "symbols": [], "intents": [], "invariants": [], "risky_spans": []
        })
        for k in ("symbols", "intents", "invariants"):
            vals = parsed.get(k) or []
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v and v not in rec[k]:
                        rec[k].append(v)

        for se in (parsed.get("risky_spans") or []):
            if (isinstance(se, list) and len(se) >= 2 and
                isinstance(se[0], (int, float)) and isinstance(se[1], (int, float))):
                loc_s = int(se[0]); loc_e = int(se[1])
                reason = se[2] if len(se) >= 3 and isinstance(se[2], str) else ""
                # These are chunk-local; map to file-global via chunk start
                if isinstance(s_line, int):
                    gs = s_line + (loc_s - 1)
                    ge = s_line + (loc_e - 1)
                else:
                    gs = loc_s; ge = loc_e
                if gs <= ge:
                    rec["risky_spans"].append([gs, ge, reason])

    # compact / merge
    for f, rec in file_aggr.items():
        rec["symbols"]    = rec["symbols"][:50]
        rec["intents"]    = rec["intents"][:50]
        rec["invariants"] = rec["invariants"][:50]
        merged = _merge_spans([[s, e] for s, e, _ in rec["risky_spans"]])
        rec["risky_spans"] = [[s, e, ""] for s, e in merged[:50]]

    stats = {
        "orig_chunks": sum(1 for x in orig_chunks if isinstance(x, dict) and "file" in x),
        "attempts": attempts, "hits": hits,
        "files": len(file_aggr),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }

    if debug_dir and cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"__ok__": True, "summary_map": file_aggr, "stats": stats}, f, indent=2)

    if verbose:
        print(f"[GRAPH-INDEX] {stats}")

    return file_aggr, stats, cache_path


# ---------- Phase 2: Detect on MUTATED graph ----------

def analyze_with_llm(
    *,
    model: str = "mistral",
    graph_path_orig: str,
    graph_path_mut: str,
    coverage_path: Optional[str] = None,
    bug_reports: Optional[list] = None,
    chunk_size: int = 4000,
    verbose: bool = False,
    max_chunks: Optional[int] = None,
    debug_dir: Optional[str] = None,
    logger: Optional[RunLogger] = None,
):
    """
    Phase 1: summarize ORIGINAL graph by file (cached).
    Phase 2: detect mutations from MUTATED graph chunks, using those summaries.
    Returns: {"detection": {...}, "stats": {...}, "orig_summary_path": <str|None>}
    """
    t0 = time.perf_counter()

    coverage = _load_json(coverage_path) if coverage_path else {}
    bug_reports = bug_reports or []

    # Phase 1
    orig_summary_map, idx_stats, cache_path = _summarize_original_graph(
        model=model,
        graph_path_orig=graph_path_orig,
        chunk_size=chunk_size,
        coverage=coverage,
        bug_reports=bug_reports,
        debug_dir=debug_dir,
        verbose=verbose,
        max_chunks=None,
        logger=logger,
    )

    # Phase 2: split both graphs to align by file
    orig_chunks = split_ast(graph_json_path=graph_path_orig, chunk_size=chunk_size) or []
    mut_chunks  = split_ast(graph_json_path=graph_path_mut,  chunk_size=chunk_size) or []

    def _index(chunks):
        idx: Dict[str, List[Dict[str, Any]]] = {}
        for x in chunks:
            if isinstance(x, dict) and "file" in x:
                idx.setdefault(x["file"], []).append(x)
        for k in idx:
            idx[k].sort(key=lambda d: (d.get("start_line") or 10**12, d.get("section") or 0))
        return idx

    idx_orig = _index(orig_chunks)
    idx_mut  = _index(mut_chunks)

    files_count = len(idx_mut)
    chunks_mutated = sum(len(v) for v in idx_mut.values())

    if verbose:
        print(f"[GRAPH] files_mutated={files_count}  mutated_chunks={chunks_mutated}  "
              f"orig_chunks_total={len(orig_chunks)}  mut_chunks_total={len(mut_chunks)}")

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    prefix = _load_yaml(PROMPT_PREFIX)
    system_lines = [
        prefix.get("task_intro", "") or "You are an expert software engineer and bug localization analyst.",
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        (
            (prefix.get("instructions", "") or "") +
            "\n\nYou will receive ORIGINAL file summary (from graph) and a MUTATED graph chunk for the same file.\n"
            "Detect only mutated regions within the provided MUTATED chunk lines.\n"
            "Return ONLY JSON with this exact shape:\n"
            '{"findings":[{"file":"<path>","line_spans":[[start,end],...],"confidence":0.0}]}\n'
            "Where line_spans is an array of [start,end] pairs (1-based, inclusive)."
        ),
        prefix.get("output_format", ""),
    ]
    system_text = "\n".join([s for s in system_lines if s]).strip()

    cov_hint = f"[coverage_present={bool(coverage)}]" if coverage else ""
    bug_hint = f"[bug_reports={len(bug_reports)}]" if bug_reports else ""

    all_findings: List[Dict[str, Any]] = []
    attempts = detections = 0
    skipped_no_lines = 0
    saved_samples = 0

    for file_path, sections in idx_mut.items():
        orig_list = idx_orig.get(file_path, [])

        sm = orig_summary_map.get(file_path, {})
        sm_text = json.dumps({
            "symbols": sm.get("symbols", [])[:20],
            "intents": sm.get("intents", [])[:20],
            "invariants": sm.get("invariants", [])[:20],
            "risky_spans": sm.get("risky_spans", [])[:20],
        }, ensure_ascii=False)
        sm_text = _fit(sm_text, 1800)

        # prepare a tiny raw-orig text (optional)
        orig_compact = ""
        if orig_list:
            # pick the first overlapping chunk later; for now keep first chunk’s compact content
            orig_compact = orig_list[0].get("content", "")

        for mut_sec in sections:
            if max_chunks is not None and attempts >= max_chunks:
                break

            m_s = mut_sec.get("start_line")
            m_e = mut_sec.get("end_line")
            if not m_s or not m_e:
                skipped_no_lines += 1
                continue

            # find an ORIGINAL chunk that overlaps for a bit of raw context
            raw_overlap = ""
            for oc in orig_list:
                o_s = oc.get("start_line"); o_e = oc.get("end_line")
                if o_s and o_e and not (o_e < m_s or m_e < o_s):
                    raw_overlap = oc.get("content", ""); break

            user_lines = [
                f"FILE: {file_path}",
                f"MUTATED CHUNK LINES (global): {m_s}-{m_e}",
                cov_hint, bug_hint,
                "=== ORIGINAL FILE SUMMARY (from Phase-1, graph-based) ===",
                sm_text,
                "=== ORIGINAL GRAPH CHUNK (compact) ===",
                _fit(raw_overlap or orig_compact, 1200),
                "=== MUTATED GRAPH CHUNK (compact) ===",
                _fit(mut_sec.get("content", ""), 2000),
                "Return only JSON as specified.",
            ]
            user_text = "\n".join([u for u in user_lines if u is not None])

            attempts += 1
            if verbose and attempts % 25 == 1:
                print(f"[GRAPH] calling LLM… attempt={attempts}")

            out, err = chat_or_generate(
                model=model,
                system_text=system_text,
                user_text=user_text,
                temperature=0.2,
                top_p=0.95,
                num_ctx=chunk_size,
            )

            if logger and detections < 5:
                logger.dump_pair(
                    prefix=f"gdetect_{_sanitize_name(file_path)}_{m_s}_{m_e}",
                    system_text=system_text, user_text=user_text, response_text=out or err or ""
                )
            if err or not out:
                if verbose:
                    print(f"[GRAPH] LLM error/empty content on attempt {attempts}: {err}")
                continue

            parsed = extract_first_json(out) or {}
            if not (isinstance(parsed, dict) and "findings" in parsed):
                if verbose:
                    print(f"[GRAPH] no JSON findings on attempt {attempts}")
                if debug_dir and saved_samples < 5:
                    sample = {
                        "phase": "detect",
                        "file": file_path,
                        "mutated_lines": [m_s, m_e],
                        "system_text_head": system_text[:400],
                        "user_text_head": user_text[:800],
                        "model_reply_head": out[:1200],
                        "parsed": None,
                    }
                    with open(Path(debug_dir) / f"g_sample_nojson_{saved_samples + 1}.json", "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2)
                    saved_samples += 1
                continue

            spans: List[List[int]] = []
            for ffind in parsed.get("findings", []):
                fpath = ffind.get("file") or file_path
                if fpath != file_path:
                    continue
                norm = _normalize_line_spans(ffind.get("line_spans", []))
                for s, e in norm:
                    gs = m_s + (s - 1)
                    ge = m_s + (e - 1)
                    if gs <= ge:
                        spans.append([gs, ge])

            if spans:
                detections += 1
                all_findings.append({"file": file_path, "line_spans": spans})

                if debug_dir and saved_samples < 5:
                    sample = {
                        "phase": "detect",
                        "file": file_path,
                        "mutated_lines": [m_s, m_e],
                        "model_reply_head": out[:1200],
                        "parsed": {"file": file_path, "line_spans": spans[:5]},
                    }
                    with open(Path(debug_dir) / f"g_sample_hit_{saved_samples + 1}.json", "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2)
                    saved_samples += 1

        if max_chunks is not None and attempts >= max_chunks:
            break

    # Merge per file
    merged: Dict[str, List[List[int]]] = {}
    for f in all_findings:
        merged.setdefault(f["file"], []).extend(f["line_spans"])
    detection = {"findings": [{"file": fp, "line_spans": _merge_spans(sp)} for fp, sp in merged.items()]}

    dt = time.perf_counter() - t0
    stats = {
        "index_files": idx_stats.get("files", 0),
        "index_attempts": idx_stats.get("attempts", 0),
        "index_hits": idx_stats.get("hits", 0),
        "index_duration_sec": idx_stats.get("duration_sec", 0.0),

        "files": files_count,
        "chunks_mutated": chunks_mutated,
        "chunk_attempts": attempts,
        "chunks_with_detections": detections,
        "chunks_skipped_no_lineinfo": skipped_no_lines,
        "total_duration_sec": round(dt, 2),
    }

    if verbose:
        print(f"[GRAPH] {stats}")

    if debug_dir:
        with open(Path(debug_dir) / "graph_run_summary.json", "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "detection": detection}, f, indent=2)

    return {"detection": detection, "stats": stats, "orig_summary_path": cache_path}
