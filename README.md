
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
│   ├── coverage/            # Code test coverage reports/matrices
│   ├── graphs/              # Graph representations: ASTs, CPGs, etc.
│   ├── issues/              # Extracted bug reports and logs
│   ├── mutated/              # Mutated (faulty) code versions for experiments
│   ├── raw/                  # Original datasets and sources
│   └── text/                 # Bundled plain-text code JSONs
├── models/                   # Model checkpoints
├── notebooks/                # Jupyter notebooks for EDA, prototyping
├── prompts/                  # Prompt templates for LLM modes
│   ├── prefix_graph.yaml
│   └── prefix_text.yaml
├── reports/                  # Generated analysis reports
├── results/                  # Experiment results and logs
├── scripts/                  # Data processing and analysis scripts
│   ├── __init__.py
│   ├── clone_repo.py
│   ├── code_to_graph.py
│   ├── code_to_text.py
│   ├── coverage_analysis.py
│   ├── entity_extractor.py
│   ├── generate_faulty_mutate.py
│   ├── graph_builder.py
│   └── issue_extract.py
├── src/                      # Core logic and experiment modules
│   ├── __init__.py
│   ├── config.py
│   ├── llm_graph_input.py
│   ├── llm_text_input.py
│   ├── metrics.py
│   └── utils.py
├── .gitignore
├── main.py                   # Entry point — run with different modes
├── README.md
└── requirements.txt
````

---

## Requirements

* Python 3.8 or higher
* All Python dependencies specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

* GitHub CLI (`gh`) for automatic repository downloading and cloning:

  Install from the [GitHub CLI installation guide](https://docs.github.com/en/github-cli/github-cli/quickstart).

  After installation, authenticate using:

```bash
gh auth login
```

---

## 🧠 Running LLMs Locally with Ollama (Windows)

We use [Ollama](https://ollama.com) to run models like **Mistral** and **DeepSeek-Coder** locally.

### ✅ Requirements

* Windows 10/11 (x86\_64)
* At least 16GB RAM
* A GPU with 8GB+ VRAM (e.g. RTX 3060 Ti recommended)
* WSL not required

### 🔧 Step 1: Install Ollama

Download the installer from the official website:

👉 [https://ollama.com](https://ollama.com)

Click **Download for Windows**, then run the `.exe` installer.

After installation, the Ollama service starts automatically and listens at:

```
http://localhost:11434
```

You can test it by running the following in PowerShell or Command Prompt:

```powershell
ollama --help
```

---

### 📥 Step 2: Pull and Run Models

#### Mistral

```powershell
ollama pull mistral
ollama run mistral
```

#### DeepSeek-Coder

```powershell
ollama pull deepseek-coder
ollama run deepseek-coder
```

---

### Running Program Repair Checks

Once your environment is set up, simply run the `main.py` script with the appropriate `--mode` argument to trigger an LLM-based analysis:

```bash
python main.py --mode text   # Plain code text mode
python main.py --mode graph  # Graph-based mode
```

* **`--mode text`** → Bundles Python source files into a single JSON (with file tree + code) and sends them to the LLM along with coverage and bug reports.
* **`--mode graph`** → Sends the generated graph representation (AST/CPG) along with coverage and bug reports.

The script automatically handles which LLM prompt to use based on the mode.
Coverage and bug report files are read from the configured paths in `config.py`.

---

## Retrieving Issues

GitHub CLI is used for retrieving issues.
After logging in as shown above, clone the repository and run:

```bash
gh issue list --state open --json author,body,comments,createdAt,number,state,title,url --limit 900  >pygithub_issues_open.json
gh issue list --state closed --json author,body,comments,createdAt,number,state,title,url --limit 900  >pygithub_issues_closed.json
```

**Note:** GitHub imposes a secondary limit of 900 results per minute.
If you have more than 900 issues, add a date range to the search query to split them into multiple requests. Example:

```bash
gh issue list --state closed --search "created:<=2019-10-24" --json author,body,comments,createdAt,number,state,title,url --limit 900 >pygithub_issues_2_closed.json
```

