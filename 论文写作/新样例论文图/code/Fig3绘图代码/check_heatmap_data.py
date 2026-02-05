# -*- coding: utf-8 -*-
"""临时脚本：检查原始Fig3绘图代码的热力图数据"""

import numpy as np
import pandas as pd

# 修正后的路径
BASELINE_CSV = 'QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv'
NEW_CSV = 'QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv'
CATEGORY_CSV = 'QuantumBench/quantumbench/quantumbench/category.csv'

# 读取数据
base_df = pd.read_csv(BASELINE_CSV)
new_df = pd.read_csv(NEW_CSV)
cat_df = pd.read_csv(CATEGORY_CSV)
cat_df = cat_df[['Question id', 'Subdomain_question', 'QuestionType']].copy()

# 合并
m = (
    base_df[["Question id", "Correct"]]
    .rename(columns={"Correct": "Correct_base"})
    .merge(
        new_df[["Question id", "Correct"]].rename(columns={"Correct": "Correct_new"}),
        on="Question id"
    )
    .merge(cat_df, on="Question id")
)

# 9个规范子领域
known9 = [
    "Quantum Computation", "Nuclear Physics", "Quantum Mechanics", "Quantum Chemistry",
    "Mathematics", "Quantum Field Theory", "Optics", "String Theory", "Photonics"
]
m = m[m["Subdomain_question"].isin(known9)]

print(f"合并后总题数: {len(m)}")
print()

# 题型 × 子领域 热力图数据
qtypes = ["Conceptual Understanding", "Numerical Calculation", "Algebraic Calculation"]
canonical_order = [
    "Quantum Computation", "Nuclear Physics", "Quantum Mechanics", "Quantum Chemistry",
    "Mathematics", "Quantum Field Theory", "Optics", "String Theory", "Photonics"
]
qtype_short = ["Conceptual", "Numerical", "Algebraic"]

delta_mat = np.full((len(qtypes), len(canonical_order)), np.nan)
n_mat = np.zeros((len(qtypes), len(canonical_order)), dtype=int)

print("=== Heatmap数据 (原始代码逻辑: n>=3 才计算) ===")
print()
header = "题型".ljust(12) + " ".join([f"{sd[:10]:>10}" for sd in canonical_order])
print(header)
print("-" * len(header))

for i, qt in enumerate(qtypes):
    row = []
    for j, sd in enumerate(canonical_order):
        cell = m[(m["QuestionType"] == qt) & (m["Subdomain_question"] == sd)]
        n = len(cell)
        n_mat[i, j] = n
        if n >= 3:
            delta = (cell["Correct_new"].mean() - cell["Correct_base"].mean()) * 100.0
            delta_mat[i, j] = delta
            row.append(f"{delta:+.1f}({n})")
        else:
            row.append(f"--({n})")
    print(f"{qtype_short[i]:<12} " + " ".join([f"{x:>10}" for x in row]))

print()
print(f"NaN格子数: {np.isnan(delta_mat).sum()} / {delta_mat.size}")
print()
print("结论：根据原始代码的逻辑（n>=3），热力图确实会有很多NaN格子。")
print("这是因为数据分布不均匀，很多题型×子领域的组合样本量极小。")
