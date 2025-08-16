# src/input_splitter.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


# ---------------------------
# Text bundle splitter (files -> code chunks)
# ---------------------------

def split_text(input_path: str, chunk_size: int) -> List[Dict[str, Any]]:
    """
    Split repo_text_bundle.json into chunked code sections with global line spans.

    Returns a list of dicts:
      - first entry: {"file_tree": "..."}
      - subsequent: {
            "file": "<repo-relative path>",
            "section": <1..N>,
            "content": "<code subsequence>",
            "start_line": <int>, "end_line": <int>,
            "num_lines": <int>
        }
    """
    separated_text: List[Dict[str, Any]] = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            src = json.load(f)

        # explicit telemetry helps
        print(f"[split_text] loading: {os.path.abspath(input_path)}  chunk_size={chunk_size}")
        n_files = len(src.get("files", []))
        print(f"[split_text] bundle files[]: {n_files}")

        # build splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=50,
        )

        # keep your original first entry
        separated_text.append({"file_tree": src.get("file_tree", "")})

        # produce chunks; add line spans so caller can map to file-global lines
        for i in src.get("files", []):
            path = i.get("path")
            content = i.get("content", "") or ""
            if not path or not content:
                continue

            chunks = text_splitter.split_text(content)
            if not isinstance(chunks, list):
                print(f"[split_text] WARN: splitter returned non-list for {path}")
                continue

            scan_pos = 0
            for j, sec in enumerate(chunks, start=1):
                try:
                    start_off = content.find(sec, scan_pos)
                    if start_off == -1:
                        start_off = content.find(sec)
                    end_off = start_off + len(sec) if start_off != -1 else None

                    if start_off is None or start_off == -1 or end_off is None:
                        start_line = 1
                        end_line = content.count("\n") + 1
                    else:
                        start_line = content.count("\n", 0, start_off) + 1
                        end_line = content.count("\n", 0, end_off) + 1
                        scan_pos = end_off
                except Exception as e:
                    print(f"[split_text] WARN: line map failed for {path} sec#{j}: {e}")
                    start_line = 1
                    end_line = content.count("\n") + 1

                separated_text.append({
                    "file": path,
                    "section": j,
                    "content": sec,
                    "start_line": start_line,
                    "end_line": end_line,
                    "num_lines": sec.count("\n") + 1
                })

    except FileNotFoundError:
        print(f"Error: {input_path} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON")
    except Exception as e:
        print(f"[split_text] ERROR: {e}")

    file_chunks = sum(1 for x in separated_text if isinstance(x, dict) and "file" in x)
    print(f"[split_text] produced entries: total={len(separated_text)} file_chunks={file_chunks}")

    return separated_text


# ---------------------------
# Graph bundle splitter (nodes -> graph chunks)
# ---------------------------

def _tok_len(enc, s: str) -> int:
    return len(enc.encode(s or ""))

def _node_text(n: Dict[str, Any], code_max: int = 400) -> str:
    """
    Compact per-node textual representation for the LLM prompt.
    """
    typ = n.get("type") or n.get("label") or "Node"
    name = n.get("qualified_name") or n.get("name") or ""
    s = n.get("start_line")
    e = n.get("end_line")
    hdr = f"[{typ}] {name} L{s}-{e}".strip()
    code = (n.get("code") or n.get("summary") or "").strip()
    if code_max and code:
        if len(code) > code_max:
            code = code[: code_max // 2] + "\n...\n" + code[-(code_max // 2):]
    return f"{hdr}\n{code}\n---\n"


def split_ast(graph_json_path: str, chunk_size: int) -> List[Dict[str, Any]]:
    """
    Split graph.json (with {"nodes": [...], "edges": [...]}) into LLM-sized chunks
    per source file. Each chunk includes a compact textual `content` for prompting
    plus the raw `nodes` list for downstream use.

    Returns list of dicts:
      {
        "file": "<repo-relative path>",
        "section": <1..N>,
        "nodes": [ {node}, ... ],
        "content": "<compact text for these nodes>",
        "start_line": <min start>, "end_line": <max end>
      }
    """
    results: List[Dict[str, Any]] = []
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            g = json.load(f)

        all_nodes = g.get("nodes") or []
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for n in all_nodes:
            file_key = n.get("module") or n.get("path")
            if not file_key:
                continue
            by_file.setdefault(file_key, []).append(n)

        enc = tiktoken.get_encoding("cl100k_base")

        for file_path, nodes in by_file.items():
            nodes.sort(key=lambda x: (x.get("start_line") or 10**12, x.get("end_line") or 10**12))

            current_nodes: List[Dict[str, Any]] = []
            current_text_parts: List[str] = []
            current_tokens = 0
            section = 0

            def _flush():
                nonlocal current_nodes, current_text_parts, current_tokens, section
                if not current_nodes:
                    return
                section += 1
                span_lines = [(n.get("start_line"), n.get("end_line"))
                              for n in current_nodes if isinstance(n.get("start_line"), int) and isinstance(n.get("end_line"), int)]
                if span_lines:
                    s_min = min(s for s, _ in span_lines)
                    e_max = max(e for _, e in span_lines)
                else:
                    s_min = e_max = None

                results.append({
                    "file": file_path,
                    "section": section,
                    "nodes": current_nodes,
                    "content": "".join(current_text_parts),
                    "start_line": s_min,
                    "end_line": e_max,
                })
                current_nodes = []
                current_text_parts = []
                current_tokens = 0

            for n in nodes:
                piece = _node_text(n, code_max=400)
                piece_tokens = _tok_len(enc, piece)
                if current_tokens == 0 and piece_tokens >= chunk_size:
                    current_nodes = [n]
                    current_text_parts = [piece]
                    current_tokens = piece_tokens
                    _flush()
                    continue

                if current_tokens + piece_tokens > chunk_size and current_nodes:
                    _flush()

                current_nodes.append(n)
                current_text_parts.append(piece)
                current_tokens += piece_tokens

            _flush()

        print(f"[split_ast] files={len(by_file)} chunks={len(results)}")

    except FileNotFoundError:
        print(f"Error: {graph_json_path} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON ")

    return results
