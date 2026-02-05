# QuantumBench Reproduction & Our Improvements

[![Paper Status](https://img.shields.io/badge/Paper-Unpublished-red)](#-intellectual-property--usage-policy)
[![Repo Status](https://img.shields.io/badge/Code-Research%20Project-blue)](#)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-black)](#-intellectual-property--usage-policy)

> This repository contains the full reproduction and our extended research work built upon **QuantumBench**, including experiment code, evaluation scripts, outputs/logs, and the full LaTeX manuscript.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Repository Structure](#-repository-structure)
* [Completed Work](#-completed-work)
* [Getting Started](#-getting-started)
* [Paper](#-paper)
* [FAQ](#-faq)
* [Citation](#-citation)
* [Intellectual Property & Usage Policy](#-intellectual-property--usage-policy)
* [Contact](#-contact)

---

## 🔍 Overview

This project focuses on **reproducing** and **improving** the QuantumBench-related benchmark and experimental pipeline.

The repository includes:

* Full reproduction of baseline methods and reported results
* Our own improvements / extensions on the original framework
* Multiple rounds of experiments (comparison & ablation)
* Full manuscript source files (LaTeX), figures, and writing materials

---

## 📁 Repository Structure

> Note: Some directories contain a large number of files. GitHub Web UI may fail to fully render them, but the contents will be complete after cloning.

```
.
├── QuantumBench/                  # Core reproduction + our improvements
├── 论文写作文件夹/                 # LaTeX manuscript and writing materials
│   └── 英语论文润色后.tex           # Main English paper (polished version)
├── outputs/                       # Logs, predictions, evaluation results (may be large)
└── ...
```

### Key Paths

* **Main code**: `QuantumBench/`
* **Main paper (English, polished)**:
  `论文写作文件夹/英语论文润色后.tex`

---

## ✅ Completed Work

This repository currently includes the following completed work:

### 1) Baseline Reproduction

* Environment setup and baseline pipeline reproduction
* Reproduced core evaluation results and metrics

### 2) Our Improvements / Extensions

* Implemented novel modifications on the original framework
* Added additional experimental settings and comparisons

### 3) Evaluation & Experiment Logging

* Multi-round experimental outputs saved for traceability
* Evaluation scripts and result summaries

### 4) Paper Writing

* Full LaTeX project included
* English polished version ready in the paper folder

---

## 🚀 Getting Started

> This repository is research-oriented and may require manual setup depending on your system.

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 2) Recommended environment setup

We recommend using `conda`:

```bash
conda create -n quantumbench python=3.10 -y
conda activate quantumbench
```

Then install dependencies (if requirements exist in the project):

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not provided, please check scripts inside `QuantumBench/` for dependency hints.

---

## 📝 Paper

The manuscript is included in this repository.

### Main English Version (Polished)

📌 `论文写作文件夹/英语论文润色后.tex`

---

## ❓ FAQ

### Q1: Why can’t GitHub Web UI open the full `QuantumBench/` directory?

This is a known limitation of GitHub’s web interface when a directory contains too many files.

✅ The repository contents will be complete after:

* `git clone`
* or `Download ZIP`

---

### Q2: Why is the repository large / slow to clone?

Possible reasons include:

* Large number of experiment outputs/log files
* Cached artifacts
* Many small files in evaluation assets

---

### Q3: I encounter dependency / CUDA / model download issues.

Common causes:

* mismatched `torch` / `cuda` versions
* incompatible `transformers` versions
* missing system dependencies
* restricted access to HuggingFace models

Suggested actions:

* use a clean conda environment
* align PyTorch with your CUDA version
* check error logs and install missing packages accordingly

---

### Q4: Can I reuse the code or paper for my own publication?

🚫 **No. Not without explicit written permission from the author.**

Please refer to the IP policy section below.

---

## 📚 Citation

This paper is currently **unpublished**, therefore no official BibTeX entry is provided yet.

If you need to cite or refer to this work, please contact the author for the most updated citation format.

---

## 🔒 Intellectual Property & Usage Policy

### ⚠️ IMPORTANT NOTICE (Unpublished Work)

This repository contains an **unpublished manuscript** and the full research pipeline.

**All intellectual property rights belong exclusively to the author.**
The author is the **first author** of this work.

### 🚫 Strictly Prohibited Without Permission

Any of the following actions are strictly prohibited without explicit written authorization:

* Copying or paraphrasing the manuscript for publication
* Reusing the method, experiments, or writing for another paper submission
* Plagiarism of any figures, tables, or textual descriptions
* Redistribution of this repository or its partial contents
* Commercial use of any part of this work

Violations may result in formal legal actions.

---

### 中文知识产权声明（强制）

本仓库包含尚未发表的论文全文、实验代码、创新方法、实验设计、图表与文字表述等内容。

**上述所有内容的知识产权完全归作者本人所有，作者为该论文第一作者。**

🚫 未经作者明确书面许可，严禁：

* 复制、改写、翻译或抄袭论文内容并用于投稿/发表
* 盗用创新点、实验设计、实验结果或方法描述
* 使用本仓库代码成果用于论文投稿或商业用途
* 将仓库内容进行二次传播或公开发布

如发现侵权行为，将保留追究法律责任的权利。

---

## 📬 Contact

For collaboration, authorization, or academic communication, please contact the author directly.
