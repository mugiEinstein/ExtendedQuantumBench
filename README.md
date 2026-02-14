# ExtendedQuantumBench: Reproducing and Extending QuantumBench for Quantum Reasoning Evaluation

[![Paper Status](https://img.shields.io/badge/Paper-Submitted%20to%20QIP-orange)](#paper)
[![Repo Status](https://img.shields.io/badge/Code-Research%20Artifact-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

> **ExtendedQuantumBench** is a research artifact accompanying our QIP submission  
> **“Extending QuantumBench: A Selective Symbolic Enhancement and Multi-Perspective Evaluation Framework for Solving Quantum Problems”**.

This repository provides:

- A **strict reproduction** of the original **QuantumBench** MCQ benchmark (769 questions)
- A **selective symbolic augmentation** framework (Hybrid v1–v4, with deterministic gating)
- An **open-ended quantum reasoning task suite** (165 tasks) + a **dual-signal evaluator**
- A **graduate-level extrapolation benchmark** (**QuantumBench-Grad**, 71 tasks)
- Experiment scripts, cached outputs, and analysis utilities

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment & Installation](#3-environment--installation)
- [4. Quick Start (Reproduce Key Results)](#4-quick-start-reproduce-key-results)
- [5. Detailed Reproduction](#5-detailed-reproduction)
  - [5.1 Baseline MCQ](#51-baseline-mcq)
  - [5.2 Hybrid v1–v4 (Selective Symbolic Augmentation)](#52-hybrid-v1v4-selective-symbolic-augmentation)
  - [5.3 Open-Ended Task Suite + Dual Evaluation](#53-open-ended-task-suite--dual-evaluation)
  - [5.4 QuantumBench-Grad](#54-quantumbench-grad)
- [6. Outputs & Result Files](#6-outputs--result-files)
- [7. Notes on Reproducibility](#7-notes-on-reproducibility)
- [8. Paper](#8-paper)
- [9. Citation](#9-citation)
- [10. License](#10-license)
- [11. Contact](#11-contact)

---

## 1. Overview

### What is QuantumBench?

QuantumBench (Minami et al., 2025) is a multiple-choice benchmark for quantum problem solving.
It contains **769** problems across **9** quantum-related subfields, and each question is annotated with a question type:

- Conceptual Understanding
- Algebraic Calculation
- Numerical Calculation

### What does this repository add?

This repository **does not propose a new quantum solver**.
Instead, we **systematically extend the evaluation paradigm** with three directions:

- **(C1) Selective symbolic augmentation** (SymPy-assisted, gated, reproducible)
- **(C2) Open-ended quantum reasoning evaluation** (165 tasks, dual scoring)
- **(C3) Graduate-level extrapolation benchmark** (71 tasks, structured reasoning)

---

## 2. Repository Structure

> **Important note:** This repository is organized as a research artifact.  
> The core implementation lives inside `QuantumBench/`.

```
.
├── QuantumBench/                         # Main project root
│   ├── code/                             # All runnable scripts (100_*.py, 200_*.py, ...)
│   ├── data/
│   │   ├── open_ended_tasks.json          # 165 open-ended tasks
│   │   └── grad_benchmark/
│   │       ├── quantumbench_grad.csv      # 71 grad benchmark questions
│   │       └── category_grad.csv          # grad metadata annotations
│   ├── docs/                             # experiment logs + summaries (Chinese)
│   ├── outputs/                          # cached runs, predictions, logs (optional)
│   ├── quantumbench/                     # original QuantumBench data files
│   ├── pyproject.toml                    # dependencies (uv / pip)
│   └── ...
└── README.md
├── Reproducibility.md
└── ARTIFACT.md
```

### Key entry points (scripts)

All scripts are under:

- `QuantumBench/code/`

| Script | Purpose |
|---|---|
| `100_run_benchmark.py` | Baseline MCQ reproduction |
| `200_symbolic_hybrid_benchmark.py` | Hybrid framework (wrapper) |
| `230_symbolic_hybrid_v4.py` | Selective gating hybrid (main method) |
| `300_open_ended_framework.py` | Open-ended evaluation framework |
| `410_grad_benchmark_evaluator.py` | QuantumBench-Grad evaluation |
| `analyze_*` | Result aggregation & plotting |

---

## 3. Environment & Installation

### 3.1 Python version

The experiments were conducted with:

- Python **3.12**
- Windows 11
- RTX 4060 8GB
- Local inference via **Ollama**

(Other OS should work, but was not tested in the paper.)

### 3.2 Install dependencies

This project uses a minimal dependency set declared in:

- `QuantumBench/pyproject.toml`

#### Option A: Install via pip

```bash
cd QuantumBench
pip install -U pip
pip install .
```

#### Option B: Install via uv (recommended)

```bash
cd QuantumBench
pip install uv
uv sync
```

### 3.3 Model requirement (Ollama)

We used:

- `qwen2.5:7b`

Install and run:

```bash
ollama pull qwen2.5:7b
ollama serve
```

---

## 4. Quick Start (Reproduce Key Results)

> All commands below assume you are inside the `QuantumBench/` directory.

```bash
cd QuantumBench
```

---

### 4.1 Baseline MCQ (769 questions)

```bash
python code/100_run_benchmark.py
```

Expected output:

- Overall accuracy around **38%** (depends slightly on decoding + environment)

---

### 4.2 Selective symbolic hybrid v4 (main method)

```bash
python code/230_symbolic_hybrid_v4.py
```

Expected:

- Slight overall gain vs baseline
- Gains concentrated in Quantum Computation subfield

---

### 4.3 Open-ended evaluation (165 tasks)

```bash
python code/300_open_ended_framework.py
```

Outputs:

- AutoEvaluator score
- LLM-as-a-Judge score
- Weighted fusion score

---

### 4.4 Graduate benchmark (71 questions)

```bash
python code/410_grad_benchmark_evaluator.py
```

Outputs:

- Accuracy
- Reasoning stage coverage statistics

---

## 5. Detailed Reproduction

### 5.1 Baseline MCQ

- Dataset: `quantumbench/quantumbench/quantumbench.csv`
- Annotations: `quantumbench/quantumbench/category.csv`

The evaluation follows the original QuantumBench pipeline:

- 8-option MCQ
- fixed option shuffle with `seed=0`
- regex-based answer extraction
- unbiased fallback for parsing failures

---

### 5.2 Hybrid v1–v4 (Selective Symbolic Augmentation)

The hybrid framework implements:

- Two-stage solving: zero-shot → SymPy verification path
- Deterministic gating based on:
  - question type
  - subdomain

The main method in the paper is **Hybrid v4**:

- `code/230_symbolic_hybrid_v4.py`

---

### 5.3 Open-Ended Task Suite + Dual Evaluation

Open-ended tasks are stored in:

- `data/open_ended_tasks.json`

Task categories:

- code-free derivation
- concept explanation
- analytical QA
- error diagnosis
- multi-step reasoning

The evaluation uses:

- AutoEvaluator (rule-based, interpretable)
- LLMJudge (LLM-as-a-Judge rubric)
- Weighted fusion:
  - `Final = 0.4 * Auto + 0.6 * LLMJudge`

---

### 5.4 QuantumBench-Grad

Graduate benchmark files:

- `data/grad_benchmark/quantumbench_grad.csv`
- `data/grad_benchmark/category_grad.csv`

Protocol:

- same 8-option MCQ interface
- enforced A/B/C/D structured reasoning
- metrics:
  - accuracy
  - reasoning stage coverage

---

## 6. Outputs & Result Files

By default, scripts will write outputs to:

- `outputs/`

Example subfolders:

- `outputs/run_ollama/`
- `outputs/run_hybrid_v4_full/`
- `outputs/open_ended_eval/`
- `outputs/grad_eval/`

> Note: outputs can be large.  
> If you are preparing a lightweight clone, you may remove `outputs/` safely.

---

## 7. Notes on Reproducibility

### Determinism

We enforce:

- option shuffle `seed=0`
- fallback randomness tied to question id
- deterministic gating in Hybrid v4

### Compute cost

Hybrid v4 is slower than baseline due to:

- tool invocation
- SymPy execution
- additional parsing / matching steps

---

## 8. Paper

**Title:**  
Extending QuantumBench: A Selective Symbolic Enhancement and Multi-Perspective Evaluation Framework for Solving Quantum Problems

**Status:** Submitted to QIP (under review)

---

## 9. Citation

This work is currently under review.

Please cite the repository URL for now:

```bibtex
@misc{luo2026extendedquantumbench,
  title        = {ExtendedQuantumBench: Reproducing and Extending QuantumBench for Quantum Reasoning Evaluation},
  author       = {Luo, Yanyao},
  year         = {2026},
  howpublished = {\url{https://github.com/mugiEinstein/ExtendedQuantumBench}},
  note         = {QIP submission under review}
}
```

---

## 10. License

This repository is released under the **MIT License**.

See `LICENSE`.

---

## 11. Contact

For academic communication, please contact:

- **Yanyao Luo**  
- Email: `U202341233@xs.ustb.edu.cn`
