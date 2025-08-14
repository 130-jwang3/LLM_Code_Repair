# main.py
import os
import sys
import json
from datetime import datetime

# === Step imports ===
from scripts.clone_repo import clone_repo_if_needed
from scripts.code_to_text import build_text_repo
from scripts.generate_faulty_mutate import generate_faulty_mutant_code
from scripts.coverage_analysis import analyze_test_coverage
from scripts.code_to_graph import process_directory as build_graph

from src.llm_text_input import analyze_with_llm as run_text_llm
from src.llm_graph_input import analyze_with_llm as run_graph_llm

from src.metrics import calculate_patch_accuracy, calculate_f1
from src.sandbox_patch import apply_unified_diff_in_sandbox
from src.eval_static import evaluate_patch_against_mutations
from src.eval_detection import evaluate_detection

from src.runlog import RunLogger
import logging

# === CONFIG ===
REPO_URL = "PyGithub/PyGithub"
REPO_NAME = REPO_URL.split("/")[-1]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

RAW_REPO = os.path.join(DATA_DIR, "raw", REPO_NAME)

# ORIGINAL artifacts
GRAPH_PATH = os.path.join(DATA_DIR, "graphs", "graph.json")
TEXT_BUNDLE_PATH = os.path.join(DATA_DIR, "text", "repo_text_bundle.json")

# MUTATED artifacts
MUTANTS_DIR = os.path.join(DATA_DIR, "mutated", f"{REPO_NAME}_mutants")
GRAPH_MUT_PATH = os.path.join(DATA_DIR, "graphs", "graph_MUT.json")
TEXT_BUNDLE_MUT_PATH = os.path.join(DATA_DIR, "text", "repo_text_bundle_MUT.json")

COVERAGE_DIR = os.path.join(DATA_DIR, "coverage")
COVERAGE_FILE = os.path.join(COVERAGE_DIR, "coverage.json")
ISSUES_DIR = os.path.join(DATA_DIR, "issues")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

BUG_REPORT_FILES = [
    os.path.join(ISSUES_DIR, "pygithub_issues_1_closed.json"),
    os.path.join(ISSUES_DIR, "pygithub_issues_2_closed.json"),
    os.path.join(ISSUES_DIR, "pygithub_issues_open.json"),
]

# Logging / run dirs
os.makedirs(REPORT_DIR, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(REPORT_DIR, f"run_{STAMP}.log")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

RUN_DIR = os.path.join(REPORT_DIR, f"run_{STAMP}")
os.makedirs(RUN_DIR, exist_ok=True)
RUN_LOG = RunLogger(root=RUN_DIR, run_name=STAMP)

# Env flags
RESUME = os.getenv("RESUME", "0") == "1"       # skip recomputing artifacts if they exist
SKIP_LLM = os.getenv("SKIP_LLM", "0") == "1"   # reuse last LLM detection output from debug dir
RUN_TAG = os.getenv("RUN_TAG")                 # stable debug dir tag across runs

def load_bug_reports():
    reports = []
    for file_path in BUG_REPORT_FILES:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reports.extend(json.load(f))
            except Exception as e:
                print(f"[WARN] Failed to load bug report file {file_path}: {e}")
        else:
            print(f"[WARN] Missing bug report file: {file_path}")
    return reports

def _maybe(text, fn, *args, **kwargs):
    print(text)
    return fn(*args, **kwargs)

def main(mode, model="mistral"):
    RUN_LOG.write("start", mode=mode, model=model, repo=REPO_URL, pid=os.getpid())
    if mode not in ("text", "graph"):
        print("[ERROR] Invalid mode. Use 'text' or 'graph'.")
        sys.exit(1)

    # chunk size heuristic
    chunk_size = 8000  # default
    m = model.lower()
    if "mistral" in m:
        chunk_size = 30000  # Ollama mistral shows 32k ctx; be conservative
    elif "deepseek-coder" in m:
        chunk_size = 14000
    elif "gemma2" in m:
        chunk_size = 6000

    # Stable debug dir if RUN_TAG set
    if RUN_TAG:
        dbg_text = os.path.join(REPORT_DIR, f"debug_text_{RUN_TAG}")
        dbg_graph = os.path.join(REPORT_DIR, f"debug_graph_{RUN_TAG}")
    else:
        dbg_text = os.path.join(REPORT_DIR, f"debug_text_run_{STAMP}")
        dbg_graph = os.path.join(REPORT_DIR, f"debug_graph_run_{STAMP}")

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=== [1] PREPROCESS ===")
    if not (RESUME and os.path.exists(RAW_REPO)):
        clone_repo_if_needed(REPO_URL, RAW_REPO)
    else:
        print(f"[RESUME] Using existing raw repo: {RAW_REPO}")
    RUN_LOG.write("clone_done", path=RAW_REPO)

    bug_reports = load_bug_reports()
    print(f"[LOG] Loaded {len(bug_reports)} bug reports")
    RUN_LOG.write("bugs_loaded", count=len(bug_reports))

    # Coverage
    if not (RESUME and os.path.exists(COVERAGE_FILE)):
        analyze_test_coverage(source_dir=RAW_REPO, output_dir=COVERAGE_DIR, repo_path=RAW_REPO)
    else:
        print(f"[RESUME] Using existing coverage: {COVERAGE_FILE}")
    RUN_LOG.write("coverage_done", path=COVERAGE_FILE)

    # ORIGINAL artifacts
    print("\n=== [1.1] BUILD ORIGINAL ARTIFACTS ===")
    if not (RESUME and os.path.exists(TEXT_BUNDLE_PATH)):
        _maybe("[build_text_repo]", build_text_repo, input_dir=RAW_REPO, output_file=TEXT_BUNDLE_PATH)
    else:
        print(f"[RESUME] Using existing text bundle: {TEXT_BUNDLE_PATH}")

    if not (RESUME and os.path.exists(GRAPH_PATH)):
        _maybe("[build_graph orig]", build_graph, RAW_REPO, GRAPH_PATH)
    else:
        print(f"[RESUME] Using existing graph: {GRAPH_PATH}")
    RUN_LOG.write("built_original_artifacts", text=TEXT_BUNDLE_PATH, graph=GRAPH_PATH)

    print("\n=== [2] MUTATION GENERATION ===")
    if not (RESUME and os.path.exists(MUTANTS_DIR) and os.path.exists(os.path.join(MUTANTS_DIR, "mutated_files.json"))):
        generate_faulty_mutant_code(RAW_REPO, MUTANTS_DIR)
    else:
        print(f"[RESUME] Using existing mutants dir: {MUTANTS_DIR}")
    RUN_LOG.write("mutants_generated", dir=MUTANTS_DIR)

    # MUTATED artifacts
    print("\n=== [2.1] BUILD MUTATED ARTIFACTS ===")
    if not (RESUME and os.path.exists(TEXT_BUNDLE_MUT_PATH)):
        _maybe("[build_text_repo MUT]", build_text_repo, input_dir=MUTANTS_DIR, output_file=TEXT_BUNDLE_MUT_PATH)
    else:
        print(f"[RESUME] Using existing mutated text bundle: {TEXT_BUNDLE_MUT_PATH}")

    if not (RESUME and os.path.exists(GRAPH_MUT_PATH)):
        _maybe("[build_graph MUT]", build_graph, MUTANTS_DIR, GRAPH_MUT_PATH)
    else:
        print(f"[RESUME] Using existing mutated graph: {GRAPH_MUT_PATH}")
    RUN_LOG.write("built_mutated_artifacts", text=TEXT_BUNDLE_MUT_PATH, graph=GRAPH_MUT_PATH)

    print("\n=== [3] LLM ANALYSIS ===")
    llm_result = None

    if SKIP_LLM:
        # load previous detection from the stable debug dir
        if mode == "text":
            sum_path = os.path.join(dbg_text, "run_summary.json")
        else:
            sum_path = os.path.join(dbg_graph, "graph_run_summary.json")

        if os.path.exists(sum_path):
            with open(sum_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            llm_result = {"detection": prev.get("detection"), "stats": prev.get("stats")}
            print(f"[RESUME] Loaded previous LLM detection from: {sum_path}")
        else:
            print(f"[WARN] SKIP_LLM=1 set but no prior summary at {sum_path}; running LLM anyway.")
            SKIP_LLM_local = False
        # fallthrough; if not found, we’ll run LLM below

    if llm_result is None:
        if mode == "text":
            os.makedirs(dbg_text, exist_ok=True)
            llm_result = run_text_llm(
                model=model,
                text_path_orig=TEXT_BUNDLE_PATH,
                text_path_mut=TEXT_BUNDLE_MUT_PATH,
                coverage_path=COVERAGE_FILE,
                bug_reports=bug_reports,
                chunk_size=chunk_size,
                verbose=True,
                max_chunks=None,
                debug_dir=dbg_text,   # <— stable if RUN_TAG set, enables index cache reuse
                logger=RUN_LOG,
            )
            stats = (llm_result or {}).get("stats", {})
            print(f"[TEXT] attempts={stats.get('chunk_attempts')} hits={stats.get('chunks_with_detections')} "
                  f"files={stats.get('files')} chunks={stats.get('chunks_mutated')} "
                  f"skipped={stats.get('chunks_skipped_no_lineinfo')} "
                  f"duration={stats.get('total_duration_sec')}s")
            print(f"[TEXT] debug artifacts: {dbg_text}")
            RUN_LOG.write("text_llm_done", stats=stats)
        else:  # graph
            os.makedirs(dbg_graph, exist_ok=True)
            llm_result = run_graph_llm(
                model=model,
                graph_path_orig=GRAPH_PATH,
                graph_path_mut=GRAPH_MUT_PATH,
                coverage_path=COVERAGE_FILE,
                bug_reports=bug_reports,
                chunk_size=chunk_size,
                verbose=True,
                max_chunks=None,
                debug_dir=dbg_graph,  # <— stable if RUN_TAG set, enables index cache reuse
                logger=RUN_LOG,
            )
            stats = (llm_result or {}).get("stats", {})
            print(f"[GRAPH] attempts={stats.get('chunk_attempts')} "
                  f"hits={stats.get('chunks_with_detections')} "
                  f"files={stats.get('files')} chunks={stats.get('chunks_mutated')} "
                  f"skipped={stats.get('chunks_skipped_no_lineinfo')} "
                  f"duration={stats.get('total_duration_sec')}s")
            print(f"[GRAPH] debug artifacts: {dbg_graph}")
            RUN_LOG.write("graph_llm_done", stats=stats)

    # === Detection metrics vs mutated_files.json ===
    print("\n=== [4] DETECTION METRICS (vs mutated_files.json) ===")
    detection_json = llm_result.get("detection") if isinstance(llm_result, dict) else None
    det_metrics = evaluate_detection(detection_json or {}, MUTANTS_DIR)
    print("Detection:", det_metrics)
    RUN_LOG.write("detection_metrics", **det_metrics)

    # === Optional static repair scoring if a unified diff was returned ===
    print("\n=== [5] OPTIONAL STATIC REPAIR EVAL ===")
    patch_application = None
    static_metrics = None
    repair_obj = llm_result.get("repair") if isinstance(llm_result, dict) else None
    if isinstance(repair_obj, dict) and "diff" in repair_obj:
        ok, rep = apply_unified_diff_in_sandbox(RAW_REPO, repair_obj["diff"])
        patch_application = {"ok": ok, **rep}
        if ok:
            static_metrics = evaluate_patch_against_mutations(
                sandbox_repo=rep["sandbox"],
                mutants_dir=MUTANTS_DIR,
                predicted_diff=repair_obj["diff"],
            )
            print("Static repair metrics:", static_metrics)
        else:
            print("Sandbox patch failed; see patch_application for details.")

    # Placeholders (optional legacy)
    print("\n=== [6] LEGACY PLACEHOLDER METRICS ===")
    predictions = ["patch1"] if static_metrics else []
    labels = ["patch1"] if static_metrics else []
    patch_acc = calculate_patch_accuracy(predictions, labels) if predictions else 0.0
    f1_score = calculate_f1(predictions, labels) if predictions else 0.0

    # === Report dump ===
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "repo": REPO_URL,
        "analysis_mode": mode,
        "llm_output": llm_result,
        "detection_metrics": det_metrics,
        "patch_application": patch_application,
        "static_metrics": static_metrics,
        "bug_report_count": len(bug_reports),
        "patch_accuracy": patch_acc,
        "f1_score": f1_score,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{mode}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    RUN_LOG.write("report_written", path=report_path, mode=mode)
    print(f"\n {mode.capitalize()} analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [text|graph] [model_name]")
        sys.exit(1)
    mode_arg = sys.argv[1].lower()
    model_arg = sys.argv[2] if len(sys.argv) >= 3 else "mistral"
    main(mode=mode_arg, model=model_arg)
