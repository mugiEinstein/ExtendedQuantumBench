# Reproducibility Guide for ExtendedQuantumBench

This document provides a precise, reviewer-oriented reproduction guide for the QIP submission:

**“Extending QuantumBench: A Selective Symbolic Enhancement and Multi-Perspective Evaluation Framework for Solving Quantum Problems.”**

The goal is to enable deterministic replication of the main experimental results reported in the manuscript.

---

# 1. Scope of Reproduction

This guide covers reproduction of:

1. Baseline MCQ results (769 QuantumBench questions)
2. Hybrid v4 selective symbolic augmentation results
3. Open-ended task evaluation (165 tasks; pilot and scaled subsets)
4. QuantumBench-Grad (71 graduate-level tasks)

---

# 2. System Configuration Used in the Paper

The experiments in the paper were conducted under the following configuration:

- OS: Windows 11
- Python: 3.12
- GPU: RTX 4060 (8GB)
- Inference backend: Ollama (local HTTP server)
- Model: qwen2.5:7b
- Random seed: 0 (for option shuffle and fallback control)

Other operating systems should work, but were not explicitly validated in the paper.

---

# 3. Environment Setup

## 3.1 Clone the Repository

```bash
git clone https://github.com/mugiEinstein/ExtendedQuantumBench
cd ExtendedQuantumBench/QuantumBench
```

## 3.2 Install Dependencies

Option A (pip):

```bash
pip install -U pip
pip install .
```

Option B (uv):

```bash
pip install uv
uv sync
```

Dependencies are declared in `pyproject.toml`.

---

# 4. Model Setup (Ollama)

Install Ollama:

https://ollama.com/

Then pull the model used in the paper:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Ensure the Ollama server is running before executing benchmark scripts.

---

# 5. Reproducing Main Results

All commands below assume execution from:

```
ExtendedQuantumBench/QuantumBench/
```

---

## 5.1 Baseline MCQ (769 Questions)

```bash
python code/100_run_benchmark.py
```

Expected outcome:

- Overall accuracy ≈ 38% (minor fluctuations possible due to decoding)

This corresponds to Table 2 in the paper.

---

## 5.2 Hybrid v4 (Selective Symbolic Augmentation)

```bash
python code/230_symbolic_hybrid_v4.py
```

Expected outcome:

- Slight net gain over baseline
- Gains concentrated in Quantum Computation subfield
- Increased token usage and runtime

This corresponds to Tables 3–6 in the paper.

---

## 5.3 Open-Ended Evaluation

```bash
python code/300_open_ended_framework.py
```

Outputs:

- AutoEvaluator score
- LLMJudge score
- Final weighted score

Fusion rule:

Final = 0.4 × Auto + 0.6 × LLMJudge

Matches Section 4.3 in the manuscript.

---

## 5.4 QuantumBench-Grad

```bash
python code/410_grad_benchmark_evaluator.py
```

Outputs:

- Accuracy
- Reasoning stage coverage statistics

Matches Section 5.5 in the manuscript.

---

# 6. Determinism and Randomness Control

The following mechanisms ensure reproducibility:

- Option shuffle uses fixed `seed=0`
- Fallback randomness is tied to question index
- Hybrid v4 gating is deterministic based on question type and subdomain
- Regex-based extraction follows a fixed parsing hierarchy

Note: Minor variability may occur if different decoding parameters or model builds are used.

---

# 7. Runtime Expectations

Approximate runtime on RTX 4060 (8GB):

- Baseline (769 MCQ): ~2–3 hours
- Hybrid v4 (769 MCQ): ~4–5 hours
- Open-ended (30 tasks subset): ~20 minutes
- Grad benchmark (71 tasks): ~30–60 minutes

---

# 8. Output Files

Results are stored under:

```
outputs/
```

Typical subfolders:

- run_ollama/
- run_hybrid_v4_full/
- open_ended_eval/
- grad_eval/

These can be safely removed if disk space is limited.

---

# 9. Known Limitations

- Only one model (Qwen2.5-7B) was evaluated in the submission.
- Subdomain statistical tests were not corrected for multiple comparisons.
- LLMJudge uses the same model family as the evaluated model (self-evaluation bias acknowledged in paper).

---

# 10. Reviewer Notes

This repository is structured as a research artifact rather than a polished software package.

If issues arise during reproduction:

1. Ensure Ollama server is running
2. Verify Python version compatibility
3. Check CUDA / PyTorch alignment (if applicable)
4. Confirm correct working directory before running scripts

For further clarification, please contact the author.

---

End of reproducibility guide.
