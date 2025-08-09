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

# === CONFIG ===
REPO_URL = "PyGithub/PyGithub"
REPO_NAME = REPO_URL.split("/")[-1]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_REPO = os.path.join(DATA_DIR, "raw", REPO_NAME)
GRAPH_PATH = os.path.join(DATA_DIR, "graphs", "graph.json")
TEXT_BUNDLE_PATH = os.path.join(DATA_DIR, "text", "repo_text_bundle.json")
MUTANTS_DIR = os.path.join(DATA_DIR, "mutated", f"{REPO_NAME}_mutants")
COVERAGE_DIR = os.path.join(DATA_DIR, "coverage")
COVERAGE_FILE = os.path.join(COVERAGE_DIR, "coverage.json")
ISSUES_DIR = os.path.join(DATA_DIR, "issues")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# Pre-existing bug report files
BUG_REPORT_FILES = [
    os.path.join(ISSUES_DIR, "pygithub_issues_1_closed.json"),
    os.path.join(ISSUES_DIR, "pygithub_issue_e_closed.json"),
    os.path.join(ISSUES_DIR, "pygithub_issues_open.json")
]

def load_bug_reports():
    reports = []
    for file_path in BUG_REPORT_FILES:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reports.extend(json.load(f))
            except Exception as e:
                print(f"⚠️ Failed to load bug report file {file_path}: {e}")
        else:
            print(f"⚠️ Missing bug report file: {file_path}")
    return reports

def main(mode, model="mistral"):
    if mode not in ("text", "graph"):
        print("❌ Invalid mode. Use 'text' or 'graph'.")
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=== [1] PREPROCESS ===")
    clone_repo_if_needed(REPO_URL, RAW_REPO)

    bug_reports = load_bug_reports()
    print(f"📄 Loaded {len(bug_reports)} bug reports from pre-existing JSON files")

    analyze_test_coverage(
        source_dir=RAW_REPO,
        output_dir=COVERAGE_DIR,
        repo_path=RAW_REPO
    )

    if mode == "graph":
        build_graph(RAW_REPO, GRAPH_PATH)
    elif mode == "text":
        os.makedirs(os.path.dirname(TEXT_BUNDLE_PATH), exist_ok=True)
        build_text_repo(
            input_dir=RAW_REPO,
            output_file=TEXT_BUNDLE_PATH,
            # max_bytes_per_file=None  # optional cap
        )

    print("\n=== [2] MUTATION GENERATION ===")
    generate_faulty_mutant_code(RAW_REPO, MUTANTS_DIR)

    print("\n=== [3] LLM ANALYSIS ===")

    # TODO: Implement striding/chunking for large inputs before passing to LLMs
    #   - Both text and graph modes will need max token-aware chunking with overlap
    #   - Striding should happen here so that both modes share the same logic

    if mode == "text":
        llm_result = run_text_llm(
            model=model,
            text_path=TEXT_BUNDLE_PATH,
            repo_path=RAW_REPO,
            coverage_path=COVERAGE_FILE,
            bug_reports=bug_reports
        )
    elif mode == "graph":
        llm_result = run_graph_llm(
            model=model,
            graph_path=GRAPH_PATH,
            repo_path=RAW_REPO,
            coverage_path=COVERAGE_FILE,
            bug_reports=bug_reports
        )

    print("\n=== [4] CODE REPAIR ===")
    predictions = ["patch1", "patch2"]
    labels = ["patch1", "patch2"]

    print("\n=== [5] METRICS ===")
    patch_acc = calculate_patch_accuracy(predictions, labels)
    f1_score = calculate_f1(predictions, labels)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "repo": REPO_URL,
        "analysis_mode": mode,
        "patch_accuracy": patch_acc,
        "f1_score": f1_score,
        "llm_output": llm_result,
        "bug_report_count": len(bug_reports)
    }

    report_path = os.path.join(REPORT_DIR, f"{mode}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✅ {mode.capitalize()} analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [text|graph] [model_name]")
        sys.exit(1)

    mode_arg = sys.argv[1].lower()
    model_arg = sys.argv[2] if len(sys.argv) >= 3 else "mistral"
    main(mode=mode_arg, model=model_arg)
