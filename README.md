

# Program Repair with LLMs: Text, Graph, and Coverage-Based Analysis

## Overview

This project explores automated bug localization and program repair with LLMs using two input modalities:

1. **Plain Code Text** — repo bundled into a JSON (file tree + code).
2. **Graph View** — AST/CPG-like graph chunks built from the code.

Both modes can combine **test coverage** and **bug reports** to guide localization and repairs.

---

## Repository layout

```text
program-repair-llm/
├─ data/
│  ├─ coverage/                  # coverage.json
│  ├─ graphs/
│  │  ├─ graph_ORIG.json         # built once from original code
│  │  └─ graph_MUT.json          # built once from mutated code
│  ├─ issues/                    # GitHub issues exports (json)
│  ├─ mutated/                   # mutated (faulty) working tree
│  ├─ raw/                       # original checkout / raw inputs
│  └─ text/
│     ├─ repo_text_bundle_ORIG.json
│     └─ repo_text_bundle_MUT.json
├─ prompts/
│  ├─ prefix_graph.yaml          # graph prompt
│  └─ prefix_text.yaml           # text prompt
├─ reports/
│  ├─ *.log                      # run logs (nohup/stdout/stderr)
│  ├─ debug_text_run_<RUN_TAG>/  # cached text LLM artifacts
│  │  ├─ original_summaries.json
│  │  └─ run_summary.json
│  └─ debug_graph_run_<RUN_TAG>/ # cached graph LLM artifacts
│     ├─ original_graph_summaries.json
│     └─ graph_run_summary.json
├─ results/
│  ├─ text_analysis_report_<YYYYMMDD_HHMMSS>.json
│  └─ graph_analysis_report_<YYYYMMDD_HHMMSS>.json
├─ scripts/                      # data prep & utilities
│  ├─ code_to_text.py
│  ├─ code_to_graph.py
│  ├─ coverage_analysis.py
│  ├─ issue_extract.py
│  └─ generate_faulty_mutate.py
├─ src/
│  ├─ config.py                  # paths & prompt config
│  ├─ input_splitter.py          # split_text / split_ast
│  ├─ llm_common.py              # Ollama client + JSON extraction
│  ├─ llm_text_input.py          # text pipeline (index + detect)
│  ├─ llm_graph_input.py         # graph pipeline (index + detect)
│  ├─ metrics.py                 # evaluation helpers
│  └─ runlog.py                  # lightweight tracing
├─ main.py                       # entry point (positional args)
├─ requirements.txt
└─ README.md
```

---

## Requirements

* Python 3.10+ (3.8+ works but tested primarily on 3.10/3.11)
* Dependencies:

```bash
pip install -r requirements.txt
```

* (Optional) **GitHub CLI** to export issues:

```bash
gh auth login
```

---

## 🧠 Local LLMs with Ollama (Windows/Linux/macOS)

* Install Ollama: [https://ollama.com](https://ollama.com)
* Service runs at `http://localhost:11434` by default.

Pull models:

```bash
ollama pull mistral
ollama pull deepseek-coder
```

> You can override the host with `OLLAMA_BASE` (e.g. `http://127.0.0.1:11434`).

---

## Running the pipelines

`main.py` uses **positional args**:

```
python -u main.py <mode> <model>
```

* `<mode>`: `text` or `graph`
* `<model>`: e.g. `mistral`, `deepseek-coder`

### Environment flags you’ll actually use

* `RUN_TAG` — labels `reports/debug_*_run_<RUN_TAG>/…` (cache dir).
* `RESUME=1` — reuse existing prebuilt artifacts (text/graph bundles, coverage, issues) and cached LLM summaries if present.
* `SKIP_LLM=0` — run the model (set to `1` to skip).
* `PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8` — cleaner logs.

### Git Bash / Linux (with `nohup`)

**Graph + DeepSeek-Coder:**

```bash
nohup env PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  RUN_TAG=run_$(date +%Y%m%d_%H%M%S) RESUME=1 SKIP_LLM=0 \
  python -u main.py graph deepseek-coder \
  > reports/graph_deepseek_coder.log 2>&1 &
```

**Text + Mistral:**

```bash
nohup env PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  RUN_TAG=run_$(date +%Y%m%d_%H%M%S) RESUME=1 SKIP_LLM=0 \
  python -u main.py text mistral \
  > reports/text_mistral.log 2>&1 &
```

### PowerShell (no `nohup`)

```powershell
$env:PYTHONUNBUFFERED="1"
$env:PYTHONIOENCODING="utf-8"
$env:RUN_TAG="run_20250815_120000"
$env:RESUME="1"
$env:SKIP_LLM="0"
Start-Process -NoNewWindow `
  -FilePath python `
  -ArgumentList "-u","main.py","graph","deepseek-coder" `
  -RedirectStandardOutput "reports\graph_deepseek_coder.log" `
  -RedirectStandardError  "reports\graph_deepseek_coder.log"
```

---

## What gets cached & how to resume without re-indexing

**Index (Phase-1) cache files:**

* **Text:** `reports/debug_text_run_<RUN_TAG>/original_summaries.json`
* **Graph:** `reports/debug_graph_run_<RUN_TAG>/original_graph_summaries.json`

**To avoid re-indexing:**

* Easiest: **reuse the same `RUN_TAG`** and run with `RESUME=1`.
* If you changed `RUN_TAG`, copy the prior `original_*_summaries.json`
  into the new debug dir before running:

  * Text → `reports/debug_text_run_<NEW_TAG>/original_summaries.json`
  * Graph → `reports/debug_graph_run_<NEW_TAG>/original_graph_summaries.json`

The pipeline already reuses built artifacts:

* `data/text/repo_text_bundle_ORIG.json` / `_MUT.json`
* `data/graphs/graph_ORIG.json` / `_MUT.json`
* `data/coverage/coverage.json`
* `data/issues/*.json`

---

## Outputs

* **Logs:** `reports/*.log`
  (use `tail -f reports/<file>.log` in Git Bash to watch progress)

* **Per-run summaries:**

  * Text → `reports/debug_text_run_<RUN_TAG>/run_summary.json`
  * Graph → `reports/debug_graph_run_<RUN_TAG>/graph_run_summary.json`

* **Final analysis reports (for tables/figures):**

  * Text → `results/text_analysis_report_<YYYYMMDD_HHMMSS>.json`
  * Graph → `results/graph_analysis_report_<YYYYMMDD_HHMMSS>.json`

> Your earlier uploads (e.g., `graph_analysis_report_20250814_174447.json`,
> `text_analysis_report_20250815_175336.json`) match this naming.

---

## Exporting issues (optional)

```bash
# Open issues
gh issue list --state open \
  --json author,body,comments,createdAt,number,state,title,url \
  --limit 900 > data/issues/issues_open.json

# Closed issues
gh issue list --state closed \
  --json author,body,comments,createdAt,number,state,title,url \
  --limit 900 > data/issues/issues_closed.json
```

If you exceed 900, split by date:

```bash
gh issue list --state closed \
  --search "created:<=2019-10-24" \
  --json author,body,comments,createdAt,number,state,title,url \
  --limit 900 > data/issues/issues_closed_part2.json
```
