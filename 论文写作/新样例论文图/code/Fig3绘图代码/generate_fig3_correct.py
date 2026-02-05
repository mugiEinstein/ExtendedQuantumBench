# -*- coding: utf-8 -*-
"""
Fig3: Subdomain Analysis (a/b) — 与论文原始样式完全一致

原始Fig3只有2个子图：
- (a) 子领域增益/损失 + bootstrap 95% CI (paired)
- (b) 增益 vs 基线准确率 (bubble size ∝ #questions)

数据来源:
- Baseline: outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv
- v4: outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv
- Category: quantumbench/quantumbench/category.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================
# 0) 路径配置
# =========================
BASELINE_CSV = "QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv"
NEW_CSV = "QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv"
CATEGORY_CSV = "QuantumBench/quantumbench/quantumbench/category.csv"
OUT_PREFIX = "论文写作/figures/Fig3_Subdomain_Analysis"

# =========================
# 1) 读取数据
# =========================
base_df = pd.read_csv(BASELINE_CSV)
new_df = pd.read_csv(NEW_CSV)
cat_df = pd.read_csv(CATEGORY_CSV)
cat_df = cat_df[["Question id", "Subdomain_question", "QuestionType"]].copy()

# =========================
# 2) 对齐数据
# =========================
base_keep = base_df[["Question id", "Correct"]].copy()
new_keep = new_df[["Question id", "Correct"]].copy()

base_labeled = base_keep.merge(cat_df, on="Question id", how="left")
new_labeled = new_keep.merge(cat_df, on="Question id", how="left")

m = base_labeled.merge(
    new_labeled,
    on=["Question id", "Subdomain_question", "QuestionType"],
    suffixes=("_base", "_new")
)

m["Correct_base"] = m["Correct_base"].astype(bool)
m["Correct_new"] = m["Correct_new"].astype(bool)

# =========================
# 3) 过滤：只保留9个规范子领域
# =========================
valid_subdomains = [
    "Quantum Mechanics", "Optics", "Quantum Field Theory", "Quantum Chemistry",
    "Quantum Computation", "Photonics", "Mathematics", "String Theory", "Nuclear Physics"
]
m = m[m["Subdomain_question"].isin(valid_subdomains)].copy()

# =========================
# 4) 计算子领域统计数据 (Bootstrap CI)
# =========================
rng = np.random.default_rng(0)

def bootstrap_delta_ci(df_sub, B=4000):
    x = df_sub["Correct_new"].to_numpy(dtype=int)
    y = df_sub["Correct_base"].to_numpy(dtype=int)
    d = x - y
    n = len(d)
    delta = d.mean() * 100.0
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, lo, hi

# 计算每个子领域的统计数据
results = []
for sd in valid_subdomains:
    df_sd = m[m["Subdomain_question"] == sd]
    n = len(df_sd)
    baseline_acc = df_sd["Correct_base"].mean() * 100.0
    delta, ci_lo, ci_hi = bootstrap_delta_ci(df_sd, B=4000)
    results.append({
        'subdomain': sd,
        'n': n,
        'baseline_acc': baseline_acc,
        'delta_pp': delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi
    })

df = pd.DataFrame(results)

# 按 delta_pp 排序（与原始论文一致）
df = df.sort_values("delta_pp").reset_index(drop=True)

print("=== Fig3 子领域数据（按Δpp排序） ===")
print(df.to_string(index=False))

# =========================
# 5) 绘图：2-panel layout（与论文原始样式一致）
# =========================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(10, 4.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

# ---------- (a) 子领域增益/损失 + 95% CI ----------
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(df))

# 画水平线从0到delta_pp
ax.hlines(y, 0, df["delta_pp"], lw=1.5)

# 画点
ax.plot(df["delta_pp"], y, 'o')

# 画CI区间
for i, row in enumerate(df.itertuples()):
    ax.plot([row.ci_lo, row.ci_hi], [i, i], lw=1, alpha=0.7)
    # 标注数值
    ax.text(row.delta_pp + (0.3 if row.delta_pp >= 0 else -0.3), i, 
            f"{row.delta_pp:+.2f}",
            va='center', ha='left' if row.delta_pp >= 0 else 'right', fontsize=9)

# 0参考线
ax.axvline(0, lw=1, alpha=0.6)

# Y轴标签
ax.set_yticks(y, df["subdomain"])
ax.set_xlabel("Δ Accuracy (v4 - baseline, pp)")
ax.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)")
ax.grid(True, axis='x', alpha=0.3)

# ---------- (b) 增益 vs 基线准确率 气泡图 ----------
ax2 = fig.add_subplot(gs[1, 0])

# 气泡大小与题量成正比
sizes = df["n"] * 3

ax2.scatter(df["baseline_acc"], df["delta_pp"], s=sizes, alpha=0.8)
ax2.margins(x=0.08, y=0.18)

# 只标注特定的子领域（Quantum Computation和Photonics）
for row in df.itertuples():
    if row.subdomain in ["Quantum Computation", "Photonics"]:
        ax2.annotate(row.subdomain, (row.baseline_acc, row.delta_pp), 
                     xytext=(8, 4), textcoords="offset points", fontsize=9, clip_on=False)

ax2.axhline(0, lw=1, alpha=0.6)
ax2.set_xlabel("Baseline accuracy (%)")
ax2.set_ylabel("Δ Accuracy (pp)")
ax2.set_title("(b) Gain vs baseline strength (bubble size ∝ #questions)")
ax2.grid(True, alpha=0.3)

# =========================
# 6) 保存输出
# =========================
plt.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.02)

print(f"\n✅ 图片已保存:")
print(f"   - {OUT_PREFIX}.png")
print(f"   - {OUT_PREFIX}.pdf")

plt.close(fig)
