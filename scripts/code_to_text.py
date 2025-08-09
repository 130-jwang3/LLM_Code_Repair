# scripts/build_text.py
import os
import json
from datetime import datetime
from typing import Iterable, Dict, List, Tuple

PY_ONLY = (".py",)
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", ".mypy_cache", ".pytest_cache", "node_modules"}

def _iter_paths(root: str) -> Iterable[Tuple[str, List[str], List[str]]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        yield dirpath, sorted(dirnames), sorted(filenames)

def _render_tree(root: str, include_exts: Tuple[str, ...]) -> str:
    lines = [os.path.basename(os.path.abspath(root)) or root]
    root_abs = os.path.abspath(root)

    for dirpath, dirnames, filenames in _iter_paths(root):
        rel_dir = os.path.relpath(dirpath, root_abs)
        if rel_dir == ".":
            prefix = ""
        else:
            depth = len(rel_dir.split(os.sep))
            prefix = "    " * (depth - 1) + "└── "

        if rel_dir != ".":
            lines.append(f"{prefix}{os.path.basename(dirpath)}/")

        depth_indent = "    " * (0 if rel_dir == "." else len(rel_dir.split(os.sep)))
        for fn in filenames:
            if include_exts and not fn.endswith(include_exts):
                continue
            lines.append(f"{depth_indent}├── {fn}")

    return "\n".join(lines)

def _read_file(path: str, max_bytes: int | None) -> str:
    try:
        if max_bytes is not None and os.path.getsize(path) > max_bytes:
            with open(path, "rb") as f:
                data = f.read(max_bytes)
            return data.decode("utf-8", errors="ignore")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _collect_py_files(root: str, max_bytes_per_file: int | None) -> List[Dict]:
    bundle = []
    root_abs = os.path.abspath(root)
    for dirpath, _, filenames in _iter_paths(root):
        for fn in filenames:
            if not fn.endswith(PY_ONLY):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, root_abs)
            content = _read_file(full_path, max_bytes_per_file)
            bundle.append({
                "path": rel_path,
                "size_bytes": os.path.getsize(full_path) if os.path.exists(full_path) else None,
                "content": content,
            })
    bundle.sort(key=lambda x: x["path"])
    return bundle

def build_text_repo(
    input_dir: str,
    output_file: str,
    max_bytes_per_file: int | None = None
) -> None:
    """
    Build a single JSON for ONLY Python files:
      - summary (counts, timestamp)
      - file_tree (ascii; py files only)
      - files: [{path, size_bytes, content}]
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    file_tree = _render_tree(input_dir, PY_ONLY)
    files = _collect_py_files(input_dir, max_bytes_per_file)

    data = {
        "summary": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "root": os.path.abspath(input_dir),
            "file_count": len(files),
            "extensions": PY_ONLY,
        },
        "file_tree": file_tree,
        "files": files,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
