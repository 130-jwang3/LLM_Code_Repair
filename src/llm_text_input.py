import os
import json
import yaml
import time
from pathlib import Path
from typing import Optional
from .runlog import RunLogger

from .config import PROMPT_PREFIX_TEXT
from .llm_common import chat_or_generate, extract_first_json
from .input_splitter import split_text


def _normalize_line_spans(raw):
    """
    Returns a clean list of [start, end] (ints, 1-based, inclusive) from many shapes.
    """
    def _add(out, a, b):
        try:
            ia, ib = int(a), int(b)
            if ia <= ib:
                out.append([ia, ib])
        except Exception:
            pass

    spans = []
    if isinstance(raw, dict):
        raw = (raw.get("line_spans") or raw.get("lineSpans") or
               raw.get("spans") or raw.get("ranges") or raw.get("lines") or [])

    if not isinstance(raw, (list, tuple)):
        return spans

    for item in raw:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and isinstance(item[0], (int, str)) and isinstance(item[1], (int, str)):
                _add(spans, item[0], item[1]); continue
            flattened = False
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                    _add(spans, sub[0], sub[1]); flattened = True
            if flattened:
                continue
        if isinstance(item, dict):
            start = (item.get("start") or item.get("s") or item.get("from") or
                     item.get("line_start") or item.get("lineStart"))
            end   = (item.get("end")   or item.get("e") or item.get("to")   or
                     item.get("line_end")   or item.get("lineEnd"))
            if start is not None and end is not None:
                _add(spans, start, end)
    return spans


def _merge_spans(spans):
    if not spans:
        return []
    spans = sorted((int(s), int(e)) for s, e in spans)
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:120]


def _fit_chars(text: str, max_chars: int) -> str:
    # No truncation; upstream chunker handles size.
    return text or ""


def _as_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    try:
        if isinstance(value, (list, tuple)):
            nums = []
            for v in value:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return float(max(nums)) if nums else float(default)
        return float(value)
    except Exception:
        return float(default)


def _coerce_to_findings(parsed, default_file: str):
    """
    Normalize many shapes into:
      {"findings":[{"file": "<path>", "line_spans": [[s,e],...], "confidence": 0.0}]}
    """
    fixed = []

    def _pack(spans_raw, fpath, conf_raw):
        spans = _normalize_line_spans(spans_raw)
        conf = _as_float(conf_raw, 0.0)
        if conf < 0.0: conf = 0.0
        if conf > 1.0: conf = 1.0
        fixed.append({"file": fpath or default_file, "line_spans": spans, "confidence": conf})

    if isinstance(parsed, list):
        _pack(parsed, default_file, 0.0)
    elif isinstance(parsed, dict):
        if "findings" in parsed and isinstance(parsed["findings"], list):
            for f in parsed["findings"]:
                if isinstance(f, dict):
                    fpath = f.get("file") or f.get("path") or default_file
                    spans_raw = (
                        f.get("line_spans") or f.get("lineSpans") or
                        f.get("spans") or f.get("ranges") or f.get("lines") or
                        f.get("mutations") or f.get("risky_spans") or []
                    )
                    _pack(spans_raw, fpath, f.get("confidence"))
                else:
                    _pack(f, default_file, 0.0)
        else:
            fpath = parsed.get("file") or parsed.get("path") or default_file
            spans_raw = (
                parsed.get("line_spans") or parsed.get("lineSpans") or
                parsed.get("spans") or parsed.get("ranges") or parsed.get("lines") or
                parsed.get("mutations") or parsed.get("risky_spans") or []
            )
            _pack(spans_raw, fpath, parsed.get("confidence"))

    if not fixed:
        fixed = [{"file": default_file, "line_spans": [], "confidence": 0.0}]
    return {"findings": fixed}


# -------------------------
# Phase 1: ORIGINAL indexing
# -------------------------

def _summarize_original_repo(
    *,
    model: str,
    text_path_orig: str,
    chunk_size: int,
    coverage: dict,
    bug_reports: list,
    debug_dir: str | None,
    verbose: bool,
    max_chunks: int | None,
    logger: Optional[RunLogger] = None,
):
    t0 = time.perf_counter()
    cache_path = None
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(debug_dir) / "original_summaries.json")
        if os.path.exists(cache_path):
            try:
                cached = _load_json(cache_path)
                if isinstance(cached, dict) and cached.get("__ok__"):
                    if verbose:
                        print(f"[TEXT] using cached original summaries: {cache_path}")
                    return cached.get("summary_map", {}), cached.get("stats", {}), cache_path
            except Exception:
                pass

    prefix = _load_yaml(PROMPT_PREFIX_TEXT)
    orig_chunks = split_text(text_path_orig, chunk_size) or []

    system_lines = [
        prefix.get("task_intro", "") or "You are an expert software engineer.",
        "First, build a compact structural understanding of the ORIGINAL codebase.",
        "For *each* chunk you receive, summarize ONLY what that chunk defines or implies.",
        "Return JSON ONLY with the shape:",
        """{
  "file":"<path>",
  "section": <int>,
  "start_line": <int>,
  "end_line": <int>,
  "symbols": ["Class/func names or key identifiers"],
  "intents": ["What the code intends to do, high-level"],
  "invariants": ["Important invariants/pre/post-conditions"],
  "risky_spans": [[s,e,"why"], ...]  // lines likely to break if behavior changes
}""",
    ]
    system_text = "\n".join([s for s in system_lines if s]).strip()
    cov_hint = f"[coverage_present={bool(coverage)}]"
    bug_hint = f"[bug_reports={len(bug_reports)}]"

    attempts = hits = 0
    file_aggr = {}

    for x in orig_chunks:
        if not (isinstance(x, dict) and "file" in x):
            continue
        file_path = x["file"]
        section = x.get("section")
        s_line = x.get("start_line")
        e_line = x.get("end_line")
        content = x.get("content", "")

        if not (section and s_line and e_line and content):
            continue

        if max_chunks is not None and attempts >= max_chunks:
            break

        user_lines = [
            f"FILE: {file_path}",
            f"ORIGINAL CHUNK LINES: {s_line}-{e_line}",
            cov_hint, bug_hint,
            "=== ORIGINAL CHUNK ===",
            _fit_chars(content, 3500),
            "Return JSON only."
        ]
        user_text = "\n".join([u for u in user_lines if u])

        attempts += 1
        if verbose and attempts % 25 == 1:
            print(f"[INDEX] calling LLM… attempt={attempts}")

        content_out, err = chat_or_generate(
            model=model,
            system_text=system_text,
            user_text=user_text,
            temperature=0.0,   # safer for schema
            top_p=0.95,
            num_ctx=chunk_size,
            timeout_s=240,
            retries=5,
            enforce_json=True, # request strict JSON
        )
        if logger:
            logger.dump_pair(
                prefix=f"index_{_sanitize_name(file_path)}_sec{section}",
                system_text=system_text,
                user_text=user_text,
                response_text=content_out or err or "",
            )
        if err or not content_out:
            if verbose:
                print(f"[INDEX] LLM error/empty content on attempt {attempts}: {err}")
            continue

        parsed = extract_first_json(content_out) or {}
        if not (isinstance(parsed, dict) and parsed.get("file")):
            continue
        if parsed.get("file") not in (None, "", file_path):
            continue

        hits += 1
        rec = file_aggr.setdefault(file_path, {
            "symbols": [],
            "intents": [],
            "invariants": [],
            "risky_spans": [],
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
                gs = s_line + (loc_s - 1)
                ge = s_line + (loc_e - 1)
                if gs <= ge:
                    rec["risky_spans"].append([gs, ge, reason])

    for f, rec in file_aggr.items():
        rec["symbols"] = rec["symbols"][:50]
        rec["intents"] = rec["intents"][:50]
        rec["invariants"] = rec["invariants"][:50]
        spans_only = [[s, e] for s, e, _ in rec["risky_spans"]]
        merged = _merge_spans(spans_only)
        rec["risky_spans"] = [[s, e, ""] for s, e in merged[:50]]

    stats = {
        "orig_chunks": sum(1 for x in orig_chunks if isinstance(x, dict) and "file" in x),
        "attempts": attempts,
        "hits": hits,
        "files": len(file_aggr),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }

    if debug_dir and cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"__ok__": True, "summary_map": file_aggr, "stats": stats}, f, indent=2)

    if verbose:
        print(f"[INDEX] {stats}")

    return file_aggr, stats, cache_path


# -------------------------
# Phase 2: MUTATED detection
# -------------------------

def analyze_with_llm(
    * ,
    model: str = "mistral",
    text_path_orig: str,
    text_path_mut: str,
    coverage_path: str | None = None,
    bug_reports: list | None = None,
    chunk_size: int = 4000,
    verbose: bool = False,
    max_chunks: int | None = None,
    debug_dir: str | None = None,
    logger: Optional[RunLogger] = None,
):
    t0 = time.perf_counter()

    coverage = _load_json(coverage_path) if coverage_path and os.path.exists(coverage_path) else {}
    bug_reports = bug_reports or []

    orig_summary_map, idx_stats, cache_path = _summarize_original_repo(
        model=model,
        text_path_orig=text_path_orig,
        chunk_size=chunk_size,
        coverage=coverage,
        bug_reports=bug_reports,
        debug_dir=debug_dir,
        verbose=verbose,
        max_chunks=None,
        logger=logger,
    )

    orig_chunks = split_text(text_path_orig, chunk_size) or []
    mut_chunks  = split_text(text_path_mut,  chunk_size) or []

    def _index(chunks):
        idx = {}
        for x in chunks:
            if isinstance(x, dict) and "file" in x:
                idx.setdefault(x["file"], []).append(x)
        for k in idx:
            idx[k].sort(key=lambda d: (d.get("start_line") or 10 ** 12, d.get("section") or 0))
        return idx

    idx_orig = _index(orig_chunks)
    idx_mut  = _index(mut_chunks)

    files_count = len(idx_mut)
    chunks_mutated = sum(len(v) for v in idx_mut.values())

    if verbose:
        print(f"[TEXT] files_mutated={files_count}  mutated_chunks={chunks_mutated}  "
              f"orig_chunks_total={len(orig_chunks)}  mut_chunks_total={len(mut_chunks)}")

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    prefix = _load_yaml(PROMPT_PREFIX_TEXT)
    system_lines = [
        prefix.get("task_intro", "") or "You are an expert software engineer and bug localization analyst.",
        prefix.get("goal", ""),
        prefix.get("structure_description", ""),
        (prefix.get("instructions", "") or "") + "\n\n"
        "You will receive ORIGINAL summary for a file and a MUTATED chunk.\n"
        "Detect only mutated regions within the provided MUTATED chunk lines.\n"
        "Return ONLY JSON with this exact shape:\n"
        '{"findings":[{"file":"<path>","line_spans":[[start,end],...],"confidence":0.0}]}\n'
        "Where line_spans is an array of pairs [start,end] (integers, 1-based, inclusive).\n"
        "Do NOT return dicts or nested arrays inside line_spans. If nothing found, return:\n"
        '{"findings":[{"file":"<path>","line_spans":[], "confidence":0.0}]}.\n',
        prefix.get("output_format", ""),
    ]
    system_text = "\n".join([s for s in system_lines if s]).strip()

    cov_hint = f"[coverage_present={bool(coverage)}]" if coverage else ""
    bug_hint = f"[bug_reports={len(bug_reports)}]" if bug_reports else ""

    all_findings = []
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
        sm_text = _fit_chars(sm_text, 1800)

        for mut_sec in sections:
            if max_chunks is not None and attempts >= max_chunks:
                break

            m_s = mut_sec.get("start_line")
            m_e = mut_sec.get("end_line")
            if not m_s or not m_e:
                skipped_no_lines += 1
                continue

            orig_overlap = ""
            for oc in orig_list:
                o_s = oc.get("start_line"); o_e = oc.get("end_line")
                if o_s and o_e and not (o_e < m_s or m_e < o_s):
                    orig_overlap = oc.get("content", "")
                    break

            user_lines = [
                f"FILE: {file_path}",
                f"MUTATED CHUNK LINES (global): {m_s}-{m_e}",
                cov_hint,
                bug_hint,
                "=== ORIGINAL FILE SUMMARY (from Phase-1) ===",
                sm_text,
                "=== ORIGINAL CHUNK (raw context) ===",
                _fit_chars(orig_overlap, 1500),
                "=== MUTATED CHUNK (to analyze) ===",
                _fit_chars(mut_sec.get("content", ""), 2500),
                "Return only JSON as specified.",
            ]
            user_text = "\n".join([u for u in user_lines if u is not None])

            attempts += 1
            if verbose and attempts % 25 == 1:
                print(f"[TEXT] calling LLM… attempt={attempts}")

            content_out, err = chat_or_generate(
                model=model,
                system_text=system_text,
                user_text=user_text,
                temperature=0.0,   # reduce drift
                top_p=0.95,
                num_ctx=chunk_size,
                timeout_s=240,
                retries=5,
                enforce_json=True, # request strict JSON
            )

            if logger:
                logger.dump_pair(
                    prefix=f"detect_{_sanitize_name(file_path)}_lines_{m_s}_{m_e}",
                    system_text=system_text,
                    user_text=user_text,
                    response_text=content_out or err or "",
                )
            if err or not content_out:
                if verbose:
                    print(f"[TEXT] LLM error/empty content on attempt {attempts}: {err}")
                continue

            parsed = extract_first_json(content_out) or {}
            if not isinstance(parsed, (list, dict)):
                skeleton = (
                    f'Only JSON. No prose.\n'
                    f'Return exactly one of:\n'
                    f'{{"findings":[{{"file":"{file_path}","line_spans":[[1,1]],"confidence":0.0}}]}} '
                    f'OR {{"findings":[{{"file":"{file_path}","line_spans":[],"confidence":0.0}}]}}'
                )
                out2, _err2 = chat_or_generate(
                    model=model,
                    system_text=system_text,
                    user_text=skeleton,
                    temperature=0.0,
                    top_p=0.3,
                    num_ctx=1024,
                    timeout_s=60,
                    retries=2,
                    enforce_json=True,
                )
                parsed = extract_first_json(out2 or "") or {}

            coerced = _coerce_to_findings(parsed, file_path)
            if not (isinstance(coerced, dict) and "findings" in coerced):
                if verbose:
                    print(f"[TEXT] no JSON findings on attempt {attempts}")
                if debug_dir and saved_samples < 5:
                    sample = {
                        "phase": "detect",
                        "file": file_path,
                        "mutated_lines": [m_s, m_e],
                        "system_text_head": system_text[:400],
                        "user_text_head": user_text[:800],
                        "model_reply_head": (content_out or "")[:1200],
                        "parsed": parsed,
                    }
                    with open(Path(debug_dir) / f"sample_nojson_{saved_samples + 1}.json", "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2)
                    saved_samples += 1
                continue

            spans = []
            for ffind in coerced.get("findings", []):
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
                        "model_reply_head": (content_out or "")[:1200],
                        "parsed": {"file": file_path, "line_spans": spans[:5]},
                    }
                    with open(Path(debug_dir) / f"sample_hit_{saved_samples + 1}.json", "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2)
                    saved_samples += 1

        if max_chunks is not None and attempts >= max_chunks:
            break

    merged = {}
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
        print(f"[TEXT] {stats}")

    if debug_dir:
        with open(Path(debug_dir) / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "detection": detection}, f, indent=2)

    return {
        "detection": detection,
        "stats": stats,
        "orig_summary_path": cache_path,
    }
