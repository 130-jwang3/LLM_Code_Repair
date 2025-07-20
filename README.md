# Program Repair with LLMs: Text, Graph, and Coverage-Based Analysis

## Overview

This project explores automated software bug detection and program repair using Large Language Models (LLMs). We compare two main input modalities:

1. **Plain Code Text** — Providing source code directly to LLMs.
2. **Graph-Enhanced Representation** — Feeding LLMs with a combination of:
   - Code property graphs (CPG), abstract syntax trees (AST), or other graph structures derived from code.
   - **Test coverage information** (e.g., which parts of the code are executed by tests).
   - **Log and bug report data** associated with code or its graph.

Our aim is to evaluate if and how these richer, multi-modal graph-based representations improve bug localization and automated repair capabilities of LLMs compared to plain code text alone.

---

## Directory Structure

```markdown
program-repair-llm/
├── data/
│   ├── bug_reports/        # Extracted or original bug reports and logs
│   ├── coverage/           # Code test coverage reports/matrices
│   ├── graphs/             # Graph representations: ASTs, CPGs, etc.
│   ├── processed/          # Preprocessed/cleaned datasets for experiments
│   └── raw/                # Original datasets and sources (e.g. CodRep)
├── models/                 # Model checkpoints, pretrained or finetuned
├── notebooks/              # Jupyter notebooks for EDA, prototyping, experiments
├── results/
│   ├── logs/               # Experiment logs and outputs
│   └── reports/            # Generated reports and result summaries
├── scripts/                # Data processing and analysis scripts
│   ├── __init__.py
│   ├── bug_report_extract.py
│   ├── code_to_graph.py
│   ├── coverage_analysis.py
│   └── generate_faulty_code.py
├── src/                    # Core logic and experiment modules
│   ├── __init__.py
│   ├── config.py
│   ├── llm_graph_input.py
│   ├── llm_text_input.py
│   ├── metrics.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.8 or higher

- All Python dependencies specified in requirements.txt:

```bash
pip install -r requirements.txt
```

- GitHub CLI (gh) for automatic repository downloading and cloning:

- Install from the [GitHub CLI installation guide](https://docs.github.com/en/github-cli/github-cli/quickstart).

- After installation, authenticate using:

```bash
gh auth login
```

