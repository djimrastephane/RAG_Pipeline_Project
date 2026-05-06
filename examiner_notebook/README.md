# Examiner Walkthrough — RAG Pipeline

This folder contains an interactive Jupyter notebook that guides you through the retrieval-augmented generation (RAG) pipeline described in the thesis.

---

## What you need (and what you do not)

**You do NOT need the original NHS Grampian PDF files.** All pipeline outputs are pre-computed and included in the project folder.

The notebook requires the following directories to be present inside the project root (the folder containing `environment.yml`):

```
project-root/
├── examiner_notebook/          ← this notebook and README
├── configs/
│   └── thesis_rag.yaml         ← pipeline configuration
├── src/                        ← pipeline source code
├── models/                     ← NOT included in this package (downloaded automatically)
│   └── all-MiniLM-L6-v2/      ← ~90 MB, downloaded on first run if absent
├── data_processed/
│   ├── Grampian-2020-2021/     ← pre-computed chunks, FAISS index, eval set
│   ├── Grampian-2021-2022/
│   ├── Grampian-2022-2023/
│   ├── Grampian-2023-2024/
│   └── Grampian-2024-2025/     ← (and older Grampian years for ANOVA cells)
├── results/                    ← pre-computed evaluation results
└── environment.yml             ← conda environment specification
```

If the `models/all-MiniLM-L6-v2/` folder is absent, the notebook will download the model automatically from HuggingFace Hub (~90 MB, requires internet access on first run only).

---

## Setup (one-time)

### 1. Install Conda (if not already installed)

Download Miniconda from https://docs.conda.io/en/latest/miniconda.html and follow the installer instructions for your operating system.

> **Windows users:** open **Anaconda Prompt** (or **Miniforge Prompt**) rather than a regular Command Prompt or PowerShell. All `conda` and `pip` commands below should be run inside that prompt.

### 2. Create the environment

Open a terminal (on Windows: Anaconda Prompt), navigate to the **project root** (the folder containing `environment.yml`), and run:

```bash
conda env create -f environment.yml
conda activate rag-pipeline
```

This installs Python 3.11, PyTorch, FAISS, sentence-transformers, and all other dependencies. It takes a few minutes on first run.

If you prefer a lighter install that skips optional packages:

```bash
conda env create -f environment_py312_smoke.yml
conda activate rag-pipeline
```

### 3. Install the statistical testing package

`statsmodels` is required for the ANOVA and post-hoc tests in Section 4 and is not included in the conda environment file:

```bash
conda activate rag-pipeline
pip install statsmodels
```

### 4. Register the kernel with Jupyter

```bash
conda activate rag-pipeline
pip install ipykernel
python -m ipykernel install --user --name rag-pipeline --display-name "Python 3 (rag-pipeline)"
```

---

## Launch the notebook

From the project root **or** from this folder:

```bash
conda activate rag-pipeline
jupyter notebook examiner_notebook/RAG_Pipeline_Walkthrough.ipynb
```

Or with JupyterLab:

```bash
jupyter lab examiner_notebook/RAG_Pipeline_Walkthrough.ipynb
```

Select the **"Python 3 (rag-pipeline)"** kernel when prompted.

---

## Notebook contents

| Section | What it demonstrates |
|---|---|
| 1 — Setup | Dependency check, path resolution |
| 2 — Architecture | Four-stage pipeline overview |
| 3 — Configuration | `configs/thesis_rag.yaml` parameters |
| 4 — Corpus | Preprocessed artifacts: chunks, pages, tables |
| 5 — Query set | 250 gold queries with difficulty tiers |
| 6 — Indexing | Embedding model, FAISS index statistics |
| 7 — Retrieval demo | Live dense → BM25 → hybrid RRF for one query |
| 8 — Results | Pre-computed metrics, thesis tables, bootstrap CI figure |
| 9 — Re-run guide | CLI commands to reproduce all pipeline stages |
| 10 — System summary | Library versions and corpus statistics |
| 11 — Appendix | Pipeline configuration, BM25 ablation, cross-encoder ablation, doc-constrained vs global retrieval, RAGAS failure examples |

---

## Optional: running the full pipeline from the original PDF reports

All results in this package are pre-computed and the notebook is fully functional without the original PDFs. However, if you wish to verify Stage 1 (preprocessing) from scratch, the NHS Grampian annual reports are publicly available.

### Step 1 — Download the reports

Visit: **https://www.nhsgrampian.org/about-us/annual-accounts/**

Under the section **NHS Grampian Annual Accounts**, download each of the five evaluation reports individually. The direct links follow the pattern:

```
https://www.nhsgrampian.org/siteassets/about-us/corporate-documents/annual-accounts/<filename>.pdf
```

The five reports used in this study are the 2020/21 through 2024/25 annual accounts.

### Step 2 — Place the files

Create a `Data/` folder in the project root and save each downloaded PDF there. You may keep the original filename — the document identifier is passed separately via the `--doc_id` flag.

```
project-root/
└── Data/
    ├── nhs_grampian_2020-21_annual_report.pdf
    ├── nhs_grampian_2021-22_annual_report.pdf
    ├── nhs_grampian_2022-23_annual_report.pdf
    ├── nhs_grampian_2023-24_annual_report.pdf
    └── nhs_grampian_2024-25_annual_report.pdf
```

### Step 3 — Run preprocessing (one command per report)

```bash
conda activate rag-pipeline
python scripts/preprocess_hybrid.py \
    --config configs/thesis_rag.yaml \
    --pdf_path Data/nhs_grampian_2024-25_annual_report.pdf \
    --doc_id Grampian-2024-2025
```

Repeat for each report, substituting the correct filename and doc_id (`Grampian-2020-2021`, `Grampian-2021-2022`, etc.). Each run writes its outputs to `data_processed/<doc_id>/`.

### Step 4 — Rebuild the FAISS index

```bash
python scripts/build_index.py --config configs/thesis_rag.yaml
```

### Step 5 — Re-run evaluation

```bash
python scripts/retrieval_eval.py --config configs/thesis_rag.yaml
```

> **Windows note:** Stage 1 preprocessing uses `camelot-py` for table extraction, which requires [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html) to be installed separately on Windows. The notebook walkthrough does not require Ghostscript.

Section 9 of the notebook provides the full set of CLI commands with additional options.

---

## Notes for the examiner

- **No PDF files are required.** The original NHS Grampian reports are not included for data governance reasons. All pre-computed outputs (chunks, FAISS indexes, evaluation sets, and results) are provided and the notebook is fully functional from Section 4 onwards without them.
- **Internet access** is only needed if the local model cache (`models/all-MiniLM-L6-v2/`) is absent. The notebook will download the model from HuggingFace Hub automatically (~90 MB, one-time only).
- Cells can be run individually or all at once via **Kernel → Restart & Run All**.
- The retrieval demo in Section 7 is fully live — you can change `QUERY_TEXT` and `GOLD_PAGES` in the "Select demo query" cell to test any question against the 2020-2021 document.
- **Windows only:** if you wish to re-run the full preprocessing pipeline (Stage 1, not required for the walkthrough), you will need to install [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html) separately. The notebook itself does not require Ghostscript.
