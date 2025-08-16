import os
import sys
import json
import ast
import astor
import shutil
from typing import Dict, List, Any

# === Multiple Mutators with detailed tracking ===

class MultiMutator(ast.NodeTransformer):
    """
    Applies a set of simple mutations and records each change with type, description, and location.
    """
    def __init__(self):
        super().__init__()
        self.mutations: List[Dict[str, Any]] = []

    # --- helpers ---
    def _add_mut(self, node: ast.AST, node_type: str, change: str):
        # lineno/col_offset exist for parsed nodes; guard with getattr for robustness
        self.mutations.append({
            "node": node_type,
            "change": change,
            "lineno": getattr(node, "lineno", None),
            "col": getattr(node, "col_offset", None),
            "end_lineno": getattr(node, "end_lineno", None),
            "end_col": getattr(node, "end_col_offset", None),
        })

    # --- visitors ---
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            if isinstance(op, ast.Eq):
                node.ops[i] = ast.NotEq()
                self._add_mut(node, "Compare", "Eq→NotEq")
            elif isinstance(op, ast.NotEq):
                node.ops[i] = ast.Eq()
                self._add_mut(node, "Compare", "NotEq→Eq")
            elif isinstance(op, ast.Gt):
                node.ops[i] = ast.Lt()
                self._add_mut(node, "Compare", "Gt→Lt")
            elif isinstance(op, ast.Lt):
                node.ops[i] = ast.Gt()
                self._add_mut(node, "Compare", "Lt→Gt")
        return node

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            self._add_mut(node, "Constant", f"{node.value}→{not node.value}")
            return ast.copy_location(ast.Constant(value=not node.value), node)
        return node

    def visit_NameConstant(self, node: ast.NameConstant):  # fallback for very old versions
        if isinstance(node.value, bool):
            self._add_mut(node, "NameConstant", f"{node.value}→{not node.value}")
            return ast.copy_location(ast.NameConstant(value=not node.value), node)
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self._add_mut(node, "BinOp", "Add→Sub")
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
            self._add_mut(node, "BinOp", "Sub→Add")
        elif isinstance(node.op, ast.Mult):
            node.op = ast.Div()
            self._add_mut(node, "BinOp", "Mult→Div")
        elif isinstance(node.op, ast.Div):
            node.op = ast.Mult()
            self._add_mut(node, "BinOp", "Div→Mult")
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            node.op = ast.Or()
            self._add_mut(node, "BoolOp", "And→Or")
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()
            self._add_mut(node, "BoolOp", "Or→And")
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            self._add_mut(node, "UnaryOp", "Remove Not")
            return node.operand  # remove not
        else:
            self._add_mut(node, "UnaryOp", "Add Not")
            return ast.UnaryOp(op=ast.Not(), operand=node)


# === Mutation and File Logic ===

def mutate_file(src_path: str, dst_path: str) -> Dict[str, Any]:
    """
    Mutate a single file and return a JSON-serializable record describing the changes.
    """
    record: Dict[str, Any] = {
        "src_path": src_path,
        "dst_path": dst_path,
        "ok": False,
        "error": None,
        "mutation_count": 0,
        "mutations_by_type": {},
        "mutations": []
    }

    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        record["error"] = f"read_error: {e}"
        return record

    try:
        tree = ast.parse(code)
    except Exception as e:
        record["error"] = f"parse_error: {e}"
        return record

    mutator = MultiMutator()
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)

    # If no mutations occurred, just copy original code through
    try:
        mutated_code = astor.to_source(mutated_tree)
    except Exception as e:
        record["error"] = f"unparse_error: {e}"
        return record

    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(mutated_code)
    except Exception as e:
        record["error"] = f"write_error: {e}"
        return record

    # Summarize mutation info
    record["mutations"] = mutator.mutations
    record["mutation_count"] = len(mutator.mutations)

    summary: Dict[str, int] = {}
    for m in mutator.mutations:
        mtype = m.get("node", "Unknown")
        summary[mtype] = summary.get(mtype, 0) + 1
    record["mutations_by_type"] = summary

    record["ok"] = True
    return record


def generate_faulty_mutant_code(input_dir: str, output_dir: str) -> None:
    """
    Walk the repo, mutate non-test .py files, copy everything else.
    Produce JSON log with one entry per processed file (mutated or copied).
    """
    if os.path.exists(output_dir):
        print(f"Removing previous output directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except PermissionError:
            print("PermissionError while deleting. Retrying with ignore_errors=True...")
            shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)

    log_entries: List[Dict[str, Any]] = []
    json_log_path = os.path.join(output_dir, "mutated_files.json")

    for root, dirs, files in os.walk(input_dir):
        # Skip .git
        parts = root.split(os.sep)
        if '.git' in parts:
            continue

        rel_dir = os.path.relpath(root, input_dir)
        dst_root = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
        os.makedirs(dst_root, exist_ok=True)

        is_test_dir = any(p.lower() in ("tests", "test") for p in parts)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)

            if file.endswith('.py') and not is_test_dir:
                rec = mutate_file(src_file, dst_file)
                # add a stable, repo-relative path
                rec["rel_path"] = os.path.relpath(src_file, input_dir)
                rec["is_test_dir"] = False
                rec["action"] = "mutated" if rec["ok"] else "skipped"
                log_entries.append(rec)
            else:
                # copy everything else (including tests) without mutation
                try:
                    shutil.copy2(src_file, dst_file)
                    log_entries.append({
                        "src_path": src_file,
                        "dst_path": dst_file,
                        "rel_path": os.path.relpath(src_file, input_dir),
                        "ok": True,
                        "error": None,
                        "mutation_count": 0,
                        "mutations_by_type": {},
                        "mutations": [],
                        "is_test_dir": is_test_dir,
                        "action": "copied"
                    })
                except Exception as e:
                    log_entries.append({
                        "src_path": src_file,
                        "dst_path": dst_file,
                        "rel_path": os.path.relpath(src_file, input_dir),
                        "ok": False,
                        "error": f"copy_error: {e}",
                        "mutation_count": 0,
                        "mutations_by_type": {},
                        "mutations": [],
                        "is_test_dir": is_test_dir,
                        "action": "copy_failed"
                    })

    # Write JSON log
    with open(json_log_path, "w", encoding="utf-8") as jf:
        json.dump(log_entries, jf, indent=2)

    # Print a short summary
    mutated = sum(1 for e in log_entries if e.get("action") == "mutated" and e.get("ok"))
    copied = sum(1 for e in log_entries if e.get("action") == "copied")
    skipped = sum(1 for e in log_entries if e.get("action") in ("skipped", "copy_failed"))

    print(f"\n✅ Mutation complete. Mutants saved to: {output_dir}")
    print(f"📝 JSON log saved: {json_log_path}")
    print(f"   Files mutated: {mutated} | copied: {copied} | skipped/failed: {skipped}")


if __name__ == "__main__":
    # python scripts/generate_faulty_mutate.py <input_dir> <output_dir>
    if len(sys.argv) == 3:
        generate_faulty_mutant_code(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python generate_faulty_mutate.py <input_dir> <output_dir>")
