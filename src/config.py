import os


DATA_PATH = "data/"
RAW_CODE_PATH = DATA_PATH + "raw/"
MUTATED_CODE_PATH = DATA_PATH + "mutated/"
GRAPH_PATH = DATA_PATH + "graphs/"
COVERAGE_PATH = DATA_PATH + "coverage/"
BUG_REPORTS_PATH = DATA_PATH + "issues/"
README_PATH = RAW_CODE_PATH + "README.md"

# Sample files
SAMPLE_CODE = MUTATED_CODE_PATH + "sample.py"
SAMPLE_GRAPH = GRAPH_PATH + "sample_graph.json"
SAMPLE_COVERAGE = COVERAGE_PATH + "sample_coverage.json"
SAMPLE_BUG_REPORT = BUG_REPORTS_PATH + "sample_bug.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PREFIX_TEXT = os.path.join(BASE_DIR, "..", "prompts", "prefix_text.yaml")
PROMPT_PREFIX_GRAPH = os.path.join(BASE_DIR, "..", "prompts", "prefix_graph.yaml")

MODEL_PATH = "models/"
RESULTS_PATH = "results/"