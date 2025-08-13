# src/runlog.py
import os, json, time, threading

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)[:180]

class RunLogger:
    """Append-only JSONL + plaintext dumps per step."""
    def __init__(self, root: str, run_name: str | None = None):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.jsonl = os.path.join(self.root, "trace.jsonl")
        self.lock = threading.Lock()
        meta = {"run": run_name or time.strftime("%Y%m%d_%H%M%S"), "started_at": time.time()}
        with open(os.path.join(self.root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def write(self, kind: str, **data):
        rec = {"ts": time.time(), "kind": kind}
        rec.update(data)
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock, open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def dump_pair(self, prefix: str, system_text: str | None, user_text: str | None, response_text: str | None):
        prefix = _safe_name(prefix)
        sp = os.path.join(self.root, f"{prefix}.system.txt")
        up = os.path.join(self.root, f"{prefix}.user.txt")
        rp = os.path.join(self.root, f"{prefix}.response.txt")
        for p, txt in ((sp, system_text), (up, user_text), (rp, response_text)):
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt or "")
        self.write("dump_paths", prefix=prefix, system_path=sp, user_path=up, response_path=rp,
                   system_len=len(system_text or ""), user_len=len(user_text or ""), response_len=len(response_text or ""))
