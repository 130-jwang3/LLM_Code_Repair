# src/sandbox_patch.py
import os, shutil, tempfile, subprocess
from typing import Tuple, Dict, Any

def _copy_repo_to_tmp(src_repo: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="repair_sbx_")
    dst = os.path.join(tmpdir, "repo")
    shutil.copytree(src_repo, dst, dirs_exist_ok=True)
    return dst

def apply_unified_diff_in_sandbox(
    src_repo: str,
    diff_text: str,
    *,
    use_git: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Copy repo to a temp directory and apply the unified diff there.
    Returns (ok, report). Source repo is never modified.
    """
    sbx_repo = _copy_repo_to_tmp(src_repo)
    report: Dict[str, Any] = {
        "sandbox": sbx_repo, "method": "git-apply" if use_git else "in-memory",
        "stdout": "", "stderr": "", "rej_files": []
    }

    if use_git:
        # Init git if needed so apply has context
        try:
            r = subprocess.run(["git", "-C", sbx_repo, "rev-parse"], capture_output=True, text=True)
            if r.returncode != 0:
                subprocess.run(["git", "-C", sbx_repo, "init"], check=True)
                subprocess.run(["git", "-C", sbx_repo, "add", "-A"], check=True)
                subprocess.run(["git", "-C", sbx_repo, "commit", "-m", "sbx init"], check=True)
        except Exception as e:
            report["stderr"] = f"git init failed: {e}"

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".patch") as tf:
            tf.write(diff_text)
            patch_path = tf.name

        proc = subprocess.run(
            ["git", "-C", sbx_repo, "apply", "--reject", patch_path],
            capture_output=True, text=True
        )
        report["stdout"] = (proc.stdout or "")
        report["stderr"] = (proc.stderr or "")

        # collect any .rej files
        rej_files = []
        for root, _, files in os.walk(sbx_repo):
            for f in files:
                if f.endswith(".rej"):
                    rej_files.append(os.path.join(root, f))
        report["rej_files"] = rej_files
        ok = (proc.returncode == 0) and (not rej_files)
        return ok, report

    report["method"] = "not-implemented"
    return False, report
