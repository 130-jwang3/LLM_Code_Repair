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

## 🧠 Running LLMs Locally with Ollama (Windows)

We use [Ollama](https://ollama.com) to run models like **Mistral** and **DeepSeek-Coder** locally.

### ✅ Requirements

- Windows 10/11 (x86_64)
- At least 16GB RAM
- A GPU with 8GB+ VRAM (e.g. RTX 3060 Ti recommended)
- WSL not required

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


## Retrieving issues 

Github CLI was used for retrieving issues. 
After logging in as shown above, clone the directory of the github to retrieve issues from.
Go into the directory and use these commands: 

``` bash
gh issue list --state open --json author,body,comments,createdAt,number,state,title,url --limit 900  >pygithub_issues_open.json
gh issue list --state closed --json author,body,comments,createdAt,number,state,title,url --limit 900  >pygithub_issues_closed.json
```

Note that there are primary and secondary limits on Github API requests. The limiting factor here is the secondary limit of 900 per minute.
To work around this limit if there are more than 900 issues available, a search option can be added to the command with a date range to ensure no more than 900 are requested at a time.
For example: 
``` bash
 gh issue list --state closed --search "created:<=2019-10-24" --json author,body,comments,createdAt,number,state,title,url --limit 900  >pygithub_issues_2_closed.json
```