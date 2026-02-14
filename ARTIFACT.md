# Artifact Statement (QIP Submission)

This repository provides the research artifact accompanying the QIP submission:

**Extending QuantumBench: A Selective Symbolic Enhancement and Multi-Perspective Evaluation Framework for Solving Quantum Problems**

Author: **Yanyao Luo** (sole author)

---

## A. Artifact Overview

This artifact contains a complete and reproducible implementation of:

1. **QuantumBench MCQ reproduction** (769 questions)
2. **Selective symbolic augmentation framework** (Hybrid v1–v4)
3. **Open-ended quantum reasoning task suite** (165 tasks)
4. **Dual-signal evaluation framework** (AutoEvaluator + LLM-as-a-Judge)
5. **Graduate-level extrapolation benchmark** (**QuantumBench-Grad**, 71 tasks)

The primary goal of this artifact is **evaluation and diagnosis**, rather than proposing a new quantum solver.

---

## B. Claims Supported by the Artifact

The artifact supports the following main claims made in the submission:

- **C1:** Tool augmentation must be constrained by task structure and selectively triggered.  
  Hybrid v4 implements a deterministic gate for SymPy invocation and demonstrates stable net effects.

- **C2:** Multiple-choice accuracy alone cannot capture the correctness and reliability of quantum reasoning.  
  The open-ended suite + dual-signal scoring reveals decoupling between reasoning structure and correctness.

- **C3:** Enforcing multi-stage structured reasoning increases process visibility but does not guarantee correctness.  
  QuantumBench-Grad shows high stage coverage while accuracy decreases.

---

## C. Contents

The artifact includes:

- Source code for all evaluation pipelines
- Benchmark datasets and metadata (CSV/JSON)
- Scripts for running baseline and extended experiments
- Output folders for cached predictions and logs (optional)
- Analysis scripts for aggregating results and plotting

---

## D. Reproducibility

A detailed, step-by-step reproduction guide is provided in:

- `Reproducibility.md`

The reproduction protocol includes:

- fixed option shuffling (`seed=0`)
- deterministic hybrid gating (Hybrid v4)
- robust answer extraction and unbiased fallback logic
- consistent evaluation metrics across runs

---

## E. Dependencies

The paper experiments were run with:

- Python 3.12
- Ollama (local inference backend)
- Model: `qwen2.5:7b`
- SymPy for symbolic execution (Hybrid path)

The dependency list is provided in `QuantumBench/pyproject.toml`.

---

## F. Hardware Requirements

The paper reports results on:

- RTX 4060 (8GB)

However, reproduction should also be possible on:

- CPU-only machines (slower)
- Other GPUs (recommended)

Runtime estimates are given in `Reproducibility.md`.

---

## G. Usage Notes for Reviewers

This repository is structured as a **research artifact** and may not be packaged as a production-ready library.

If reproduction issues occur:

1. Ensure the Ollama server is running
2. Confirm the correct working directory (`QuantumBench/`)
3. Check Python version and dependency installation
4. Re-run scripts with clean output folders

---

## H. License

This artifact is intended for academic reproducibility and review.

**Recommended:** Release under the MIT License (standard for academic artifacts).

---

## I. Contact

For academic communication or reproduction clarification, please contact:

- **Yanyao Luo**  
- Email: `U202341233@xs.ustb.edu.cn`
