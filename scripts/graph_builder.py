# src/graph_builder.py
import os
import json
import hashlib
from itertools import count
from collections import defaultdict
from typing import Optional, Dict, Any

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from .entity_extractor import (
    is_class, is_function, is_import, is_call, get_name, get_code,
    is_assignment, is_decorator, is_docstring,
    is_if, is_for, is_while, is_try, is_with,
    get_signature, get_docstring_text, extract_called_name
)

# ---------------------------
#  Tree-sitter setup
# ---------------------------
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# ---------------------------
#  Small helpers
# ---------------------------

def stable_id(qname_or_name: str) -> str:
    base = (qname_or_name or "").encode("utf-8", errors="ignore")
    return hashlib.sha1(base).hexdigest()[:16]

def code_sha256(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()

def make_summary(code: Optional[str], max_chars: int = 400) -> Optional[str]:
    if not code:
        return None
    lines = [ln for ln in code.strip().splitlines()]
    if not lines:
        return None
    head = lines[:2]
    tail = lines[-2:] if len(lines) > 2 else []
    snippet = "\n".join(head + (["..."] if tail and len(lines) > 2 else []) + tail)
    return snippet[:max_chars]

def sloc(code: Optional[str]) -> int:
    return 0 if not code else len(code.splitlines())

def _ts_lines(ts_node):
    # Tree-sitter rows are 0-based; convert to 1-based inclusive
    s = ts_node.start_point[0] + 1
    e = ts_node.end_point[0] + 1
    return s, e


class CodeGraphBuilder:
    def __init__(self, root_dir: Optional[str] = None):
        """
        If root_dir is provided, all node['path'] and node['module'] will be
        repo-relative paths under this root. Otherwise absolute paths are used.
        """
        self.root_dir = os.path.abspath(root_dir) if root_dir else None

        self.graph: Dict[str, list[dict]] = {"nodes": [], "edges": []}
        self.node_ids = count()
        self.symbol_table: Dict[str, int] = {}   # qualified_name -> node_id
        self.file_sources: Dict[str, bytes] = {}
        self.ast_trees = {}

        # stride helpers
        self.parents: Dict[int, int] = {}        # child_id -> parent_id (via CONTAINS)
        self.node_by_id: Dict[int, Dict[str, Any]] = {}  # id -> node

    # -----------------
    # Graph helpers
    # -----------------
    def add_node(self, label: str, **attrs) -> int:
        node_id = next(self.node_ids)
        node_data: Dict[str, Any] = {"id": node_id, "label": label, "type": attrs.pop("type", label)}
        node_data.update(attrs)

        # compute sid, code_sha, summary, sloc, path/module placeholders
        qname = node_data.get("qualified_name") or node_data.get("name") or str(node_id)
        node_data["sid"] = stable_id(qname)
        code = node_data.get("code")
        node_data["code_sha"] = code_sha256(code)
        node_data["summary"] = make_summary(code)
        node_data["sloc"] = sloc(code)

        node_data.setdefault("module", None)     # repo-relative path of file
        node_data.setdefault("path", None)       # same as module; kept for clarity
        node_data.setdefault("path_abs", None)   # absolute path
        node_data.setdefault("start_line", None)
        node_data.setdefault("end_line", None)
        node_data.setdefault("parent_id", None)

        self.graph["nodes"].append(node_data)
        self.node_by_id[node_id] = node_data
        return node_id

    def add_edge(self, src: int, dst: int, edge_type: str) -> None:
        edge = {
            "source": src,
            "target": dst,
            "type": edge_type,
            # stable references for striding
            "source_sid": self.node_by_id.get(src, {}).get("sid"),
            "target_sid": self.node_by_id.get(dst, {}).get("sid"),
        }
        self.graph["edges"].append(edge)
        if edge_type == "CONTAINS":
            self.parents[dst] = src
            # set parent breadcrumb
            if dst in self.node_by_id:
                self.node_by_id[dst]["parent_id"] = src

    # -----------------
    # Pass 1 — register defs & statements
    # -----------------
    def first_pass(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            source_code = f.read()

        tree = parser.parse(source_code)
        self.file_sources[file_path] = source_code
        self.ast_trees[file_path] = tree

        rel_path = os.path.relpath(file_path, self.root_dir) if self.root_dir else os.path.abspath(file_path)
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        total_lines = source_code.decode("utf-8", "ignore").count("\n") + 1

        module_id = self.add_node(
            "Module",
            type="Module",
            name=module_name,
            path=rel_path,
            path_abs=os.path.abspath(file_path),
            qualified_name=module_name,               # treat module as top-level qname
            code=source_code.decode("utf-8", errors="ignore"),
            start_line=1,
            end_line=total_lines,
            module=rel_path,
        )
        self.symbol_table[module_name] = module_id

        self._register_defs(tree.root_node, source_code, module_id, module_name)
        self._propagate_module(module_id)

    def _propagate_module(self, module_id: int) -> None:
        """Fill `module` (repo-relative path) for the subtree and ensure breadcrumbs exist."""
        mod_path = self.node_by_id[module_id].get("path")  # repo-relative or abs
        stack = [module_id]
        # Build quick child index of CONTAINS edges for speed
        children = defaultdict(list)
        for e in self.graph["edges"]:
            if e["type"] == "CONTAINS":
                children[e["source"]].append(e["target"])

        while stack:
            cur = stack.pop()
            self.node_by_id[cur]["module"] = mod_path
            self.node_by_id[cur].setdefault("path", mod_path)
            for ch in children.get(cur, []):
                # make sure parent_id exists
                if self.node_by_id[ch].get("parent_id") is None:
                    self.node_by_id[ch]["parent_id"] = cur
                # also propagate path/module if missing
                self.node_by_id[ch].setdefault("module", mod_path)
                self.node_by_id[ch].setdefault("path", mod_path)
                stack.append(ch)

    def _register_defs(self, node, source_code: bytes, parent_id: int, scope_name: str) -> None:
        # Classes
        if is_class(node):
            s, e = _ts_lines(node)
            name = get_name(node, source_code)
            qname = f"{scope_name}.{name}"
            class_id = self.add_node(
                "Class", type="Class",
                name=name,
                qualified_name=qname,
                code=get_code(node, source_code),
                signature=None,
                docstring=get_docstring_text(node, source_code),
                start_line=s, end_line=e,
            )
            self.symbol_table[qname] = class_id
            self.add_edge(parent_id, class_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, class_id, qname)
            return

        # Functions / Methods
        if is_function(node):
            s, e = _ts_lines(node)
            name = get_name(node, source_code)
            qname = f"{scope_name}.{name}"
            parent_label = self.node_by_id[parent_id]["label"]
            label = "Method" if parent_label == "Class" else "Function"
            func_id = self.add_node(
                label, type=label,
                name=name,
                qualified_name=qname,
                code=get_code(node, source_code),
                signature=get_signature(node, source_code),
                docstring=get_docstring_text(node, source_code),
                start_line=s, end_line=e,
            )
            self.symbol_table[qname] = func_id
            self.add_edge(parent_id, func_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, func_id, qname)
            return

        # Control flow
        if is_if(node):
            s, e = _ts_lines(node)
            cf_id = self.add_node("If", type="If", code=get_code(node, source_code), start_line=s, end_line=e)
            self.add_edge(parent_id, cf_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, cf_id, scope_name)
            return
        if is_for(node):
            s, e = _ts_lines(node)
            cf_id = self.add_node("For", type="For", code=get_code(node, source_code), start_line=s, end_line=e)
            self.add_edge(parent_id, cf_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, cf_id, scope_name)
            return
        if is_while(node):
            s, e = _ts_lines(node)
            cf_id = self.add_node("While", type="While", code=get_code(node, source_code), start_line=s, end_line=e)
            self.add_edge(parent_id, cf_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, cf_id, scope_name)
            return
        if is_try(node):
            s, e = _ts_lines(node)
            cf_id = self.add_node("Try", type="Try", code=get_code(node, source_code), start_line=s, end_line=e)
            self.add_edge(parent_id, cf_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, cf_id, scope_name)
            return
        if is_with(node):
            s, e = _ts_lines(node)
            cf_id = self.add_node("With", type="With", code=get_code(node, source_code), start_line=s, end_line=e)
            self.add_edge(parent_id, cf_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, cf_id, scope_name)
            return

        # Other constructs
        if is_assignment(node):
            s, e = _ts_lines(node)
            assign_id = self.add_node("Assignment", type="Assignment", code=get_code(node, source_code),
                                      start_line=s, end_line=e)
            self.add_edge(parent_id, assign_id, "CONTAINS")
            return
        if is_decorator(node):
            s, e = _ts_lines(node)
            deco_id = self.add_node("Decorator", type="Decorator", code=get_code(node, source_code),
                                    start_line=s, end_line=e)
            self.add_edge(parent_id, deco_id, "CONTAINS")
            return
        if is_docstring(node, source_code):
            s, e = _ts_lines(node)
            doc_id = self.add_node("Docstring", type="Docstring", code=get_code(node, source_code),
                                   start_line=s, end_line=e)
            self.add_edge(parent_id, doc_id, "CONTAINS")
            return

        # Default — keep walking
        for child in node.children:
            self._register_defs(child, source_code, parent_id, scope_name)

    # -----------------
    # Pass 2 — resolve imports & calls
    # -----------------
    def second_pass(self) -> None:
        for file_path, tree in self.ast_trees.items():
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module_id = self.symbol_table[module_name]
            self._resolve_edges(tree.root_node, self.file_sources[file_path], module_id, module_name)

    def _resolve_edges(self, node, source_code: bytes, parent_id: int, scope_name: str) -> None:
        if is_import(node):
            s, e = _ts_lines(node)
            import_text = get_code(node, source_code).strip()
            import_id = self.add_node("Import", type="Import", code=import_text, start_line=s, end_line=e)
            self.add_edge(parent_id, import_id, "IMPORTS")
            return

        if is_call(node):
            call_name = extract_called_name(node, source_code)  # dotted if possible
            func_name = call_name or get_code(node, source_code).strip().split("(")[0]
            qname_same_scope = f"{scope_name}.{func_name}"
            qname_module_level = func_name

            if qname_same_scope in self.symbol_table:
                callee_id = self.symbol_table[qname_same_scope]
                self.add_edge(parent_id, callee_id, "CALLS")
            elif qname_module_level in self.symbol_table:
                callee_id = self.symbol_table[qname_module_level]
                self.add_edge(parent_id, callee_id, "CALLS")
            else:
                # Create resolvable external node with a qname and sid
                ext_qname = func_name
                ext_id = self.add_node("ExternalFunction", type="ExternalFunction",
                                       name=func_name, qualified_name=ext_qname)
                self.add_edge(parent_id, ext_id, "CALLS")
            return

        for child in node.children:
            self._resolve_edges(child, source_code, parent_id, scope_name)

    # -----------------
    # Save
    # -----------------
    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, indent=2)

    def save_stride_index(self, output_path_ndjson: str) -> None:
        """Optional: emit NDJSON where each line is a node or edge record for streaming."""
        os.makedirs(os.path.dirname(output_path_ndjson), exist_ok=True)
        with open(output_path_ndjson, "w", encoding="utf-8") as f:
            for n in self.graph["nodes"]:
                rec = {"type": "node"}
                rec.update(n)
                f.write(json.dumps(rec) + "\n")
            for e in self.graph["edges"]:
                rec = {"type": "edge"}
                rec.update(e)
                f.write(json.dumps(rec) + "\n")
