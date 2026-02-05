# -*- coding: utf-8 -*-
"""
Fig3: Subdomain Analysis (a/b/c/d) — 直接从原始实验文件计算

确保所有4个子图使用完全一致的数据源：
- Baseline: outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv
- v4: outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv

数据处理逻辑：
- 只使用9个规范子领域（排除"Algebraic Calculation"异常值）
- 子图(a)(b): 子领域级别的delta_pp和baseline_acc
- 子图(c): v4在各子领域的正确数Pareto分析
- 子图(d): 题型×子领域的delta_pp热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import textwrap

# =========================
# 0) 路径配置
# =========================
BASELINE_CSV = "QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv"
V4_CSV = "QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv"
OUT_PREFIX = "论文写作/figures/Fig3_Subdomain_Analysis_final"

# =========================
# 1) 工具函数
# =========================
def wrap_labels(labels, width=14):
    return [textwrap.fill(str(s), width=width) for s in labels]

# =========================
# 2) 读取并合并数据
# =========================
base_df = pd.read_csv(BASELINE_CSV)
v4_df = pd.read_csv(V4_CSV)

# 合并：使用v4文件中的Subdomain和Question Type作为标签
merged = base_df[['Question id', 'Correct']].merge(
    v4_df[['Question id', 'Correct', 'Subdomain', 'Question Type']], 
    on='Question id', 
    suffixes=('_base', '_v4')
)

# 9个规范子领域（论文Table 5-4的口径）
known9 = [
    'Quantum Computation', 'Nuclear Physics', 'Quantum Mechanics', 'Quantum Chemistry',
    'Mathematics', 'Quantum Field Theory', 'Optics', 'String Theory', 'Photonics'
]
merged = merged[merged['Subdomain'].isin(known9)].copy()

print(f"=== 数据统计 ===")
print(f"合并后总题数: {len(merged)} (应为767)")

# =========================
# 3) 计算子领域统计（用于子图a和b）
# =========================
# 按论文Fig3的顺序（按delta_pp从高到低排列）
canonical_order = [
    "Quantum Computation",  # +18.33
    "Nuclear Physics",      # +5.56
    "Quantum Mechanics",    # +3.77
    "Quantum Chemistry",    # +0.00
    "Mathematics",          # -2.70
    "Quantum Field Theory", # -2.80
    "Optics",               # -3.18
    "String Theory",        # -3.03
    "Photonics",            # -5.26
]

# Bootstrap计算delta_pp和CI
rng = np.random.default_rng(42)

def bootstrap_delta_ci(base_correct, v4_correct, B=2000):
    """计算paired bootstrap delta和95% CI"""
    d = v4_correct.astype(int) - base_correct.astype(int)  # 每题的差异
    n = len(d)
    delta = d.mean() * 100.0
    
    # Bootstrap重采样
    boot_deltas = np.empty(B)
    for i in range(B):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[i] = d[idx].mean() * 100.0
    
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return delta, ci_lo, ci_hi

# 计算每个子领域的统计数据
subdomain_stats = []
for sd in canonical_order:
    sub = merged[merged['Subdomain'] == sd]
    n = len(sub)
    base_correct = sub['Correct_base'].values
    v4_correct = sub['Correct_v4'].values
    
    baseline_acc = base_correct.mean() * 100.0
    v4_acc = v4_correct.mean() * 100.0
    delta, ci_lo, ci_hi = bootstrap_delta_ci(base_correct, v4_correct)
    
    subdomain_stats.append({
        'subdomain': sd,
        'n': n,
        'baseline_acc': baseline_acc,
        'v4_acc': v4_acc,
        'delta_pp': delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'v4_correct': v4_correct.sum()
    })

stats_df = pd.DataFrame(subdomain_stats)

print("\n=== 子领域统计（论文Table 5-4数据） ===")
print(stats_df[['subdomain', 'n', 'baseline_acc', 'v4_acc', 'delta_pp']].to_string(index=False))

# 提取绘图数组
delta_pp = stats_df['delta_pp'].values
ci_low = stats_df['ci_lo'].values
ci_high = stats_df['ci_hi'].values
n_questions = stats_df['n'].values
baseline_acc = stats_df['baseline_acc'].values
v4_correct_counts = stats_df['v4_correct'].values

# =========================
# 4) Pareto数据（子图c）
# =========================
# 按v4正确数降序排列
pareto_order = np.argsort(v4_correct_counts)[::-1]
sub_sorted = np.array(canonical_order)[pareto_order]
counts_sorted = v4_correct_counts[pareto_order]
cum_pct_sorted = np.cumsum(counts_sorted) / counts_sorted.sum() * 100.0

print("\n=== Pareto数据（子图c） ===")
for i, sd in enumerate(sub_sorted):
    print(f"{sd:<25} correct={counts_sorted[i]:>3}, cum%={cum_pct_sorted[i]:>5.1f}%")

# =========================
# 5) Heatmap数据（子图d）
# =========================
qtypes = ['Conceptual Understanding', 'Numerical Calculation', 'Algebraic Calculation']
qtype_short = ['Conceptual', 'Numerical', 'Algebraic']

# 创建delta_pp矩阵
delta_mat = np.zeros((len(qtypes), len(canonical_order)))
n_mat = np.zeros((len(qtypes), len(canonical_order)), dtype=int)

for i, qt in enumerate(qtypes):
    for j, sd in enumerate(canonical_order):
        sub = merged[(merged['Question Type'] == qt) & (merged['Subdomain'] == sd)]
        n = len(sub)
        n_mat[i, j] = n
        if n >= 1:
            base_acc = sub['Correct_base'].mean()
            v4_acc = sub['Correct_v4'].mean()
            delta_mat[i, j] = (v4_acc - base_acc) * 100.0
        else:
            delta_mat[i, j] = np.nan

print("\n=== Heatmap数据（子图d）===")
print("样本数矩阵 (n):")
print(f"{'题型':<12} " + " ".join([f"{sd[:8]:>8}" for sd in canonical_order]))
for i, qt in enumerate(qtype_short):
    print(f"{qt:<12} " + " ".join([f"{n_mat[i,j]:>8}" for j in range(len(canonical_order))]))

print("\nDelta_pp矩阵:")
for i, qt in enumerate(qtype_short):
    row_str = []
    for j in range(len(canonical_order)):
        if n_mat[i, j] >= 3:
            row_str.append(f"{delta_mat[i,j]:>+7.1f}")
        elif n_mat[i, j] >= 1:
            row_str.append(f"{delta_mat[i,j]:>+7.1f}*")  # 样本少，标记*
        else:
            row_str.append("     --")
    print(f"{qt:<12} " + " ".join(row_str))

# =========================
# 6) 绘图：4-panel layout
# =========================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
})

fig = plt.figure(figsize=(11.4, 8.8))

gs = GridSpec(
    nrows=3, ncols=2,
    height_ratios=[1.0, 1.0, 1.25],
    hspace=0.55,
    wspace=0.50,
    figure=fig
)

ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, :])
ax_c = fig.add_subplot(gs[2, 0])
ax_d = fig.add_subplot(gs[2, 1])

fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.20)

# ---------- (a) 子领域增益/损失 + 95% CI ----------
palette = sns.color_palette("tab10", n_colors=len(canonical_order))
y = np.arange(len(canonical_order))

# 画CI区间
for i in range(len(canonical_order)):
    ax_a.hlines(y=i, xmin=ci_low[i], xmax=ci_high[i], color=palette[i], lw=2, alpha=0.4)

# 画点
ax_a.scatter(delta_pp, y, c=palette, s=50, zorder=3, edgecolors='white', linewidths=0.5)

# 0参考线
ax_a.axvline(0, color="gray", lw=1, linestyle='--', alpha=0.7)

# 标注数值
for i, v in enumerate(delta_pp):
    offset = 0.8 if v >= 0 else -0.8
    ax_a.text(v + offset, i, f"{v:+.2f}", va="center", 
              ha="left" if v >= 0 else "right", fontsize=9, fontweight='bold')

ax_a.set_yticks(y)
ax_a.set_yticklabels(canonical_order)
ax_a.invert_yaxis()
ax_a.set_xlabel("Δ Accuracy (v4 - baseline, pp)")
ax_a.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)", fontweight='bold')
ax_a.grid(True, axis='x', alpha=0.3)

# ---------- (b) 增益 vs 基线准确率 ----------
# 气泡大小与题量成正比
sizes = n_questions * 2.5
colors = [palette[i] for i in range(len(canonical_order))]

scatter = ax_b.scatter(baseline_acc, delta_pp, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidths=0.5)

ax_b.axhline(0, color="gray", lw=1, linestyle='--', alpha=0.7)

# 标注所有子领域名称
for i, name in enumerate(canonical_order):
    # 根据位置调整标注偏移
    if name == "Quantum Computation":
        ax_b.annotate(name, (baseline_acc[i], delta_pp[i]), 
                     xytext=(8, 3), textcoords="offset points", fontsize=8)
    elif name == "Photonics":
        ax_b.annotate(name, (baseline_acc[i], delta_pp[i]), 
                     xytext=(8, -8), textcoords="offset points", fontsize=8)
    elif name == "Quantum Mechanics":
        ax_b.annotate(name, (baseline_acc[i], delta_pp[i]), 
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    elif n_questions[i] >= 80:  # 只标注大样本子领域
        ax_b.annotate(name, (baseline_acc[i], delta_pp[i]), 
                     xytext=(5, 3), textcoords="offset points", fontsize=8)

ax_b.set_xlabel("Baseline accuracy (%)")
ax_b.set_ylabel("Δ Accuracy (pp)")
ax_b.set_title("(b) Gain vs baseline strength (bubble size ∝ #questions)", fontweight='bold')
ax_b.grid(True, alpha=0.3)
ax_b.margins(x=0.08, y=0.15)

# ---------- (c) Pareto分析 ----------
x = np.arange(len(sub_sorted))
bars = ax_c.bar(x, counts_sorted, color="#4C72B0", edgecolor='white', linewidth=0.5)

ax_c_twin = ax_c.twinx()
ax_c_twin.plot(x, cum_pct_sorted, color="darkorange", marker="o", lw=2, markersize=5)
ax_c_twin.set_ylim(0, 105)

ax_c.set_xticks(x)
ax_c.set_xticklabels(wrap_labels(sub_sorted, width=12), rotation=40, ha="right", fontsize=8)
ax_c.tick_params(axis="x", pad=2)
ax_c.set_ylabel("# Correct (v4)")
ax_c_twin.set_ylabel("Cumulative (%)", color="darkorange")
ax_c_twin.tick_params(axis='y', labelcolor='darkorange')
ax_c.set_title("(c) Pareto: subdomain contribution to v4 accuracy", fontweight='bold')

# ---------- (d) 题型×子领域 Heatmap ----------
# 对于样本数<2的格子，设为NaN（不显示）
delta_mat_display = delta_mat.copy()
for i in range(len(qtypes)):
    for j in range(len(canonical_order)):
        if n_mat[i, j] < 2:
            delta_mat_display[i, j] = np.nan

# 使用diverging colormap，中心为0
vmax = max(abs(np.nanmin(delta_mat_display)), abs(np.nanmax(delta_mat_display)))
vmin = -vmax

heatmap = sns.heatmap(
    delta_mat_display,
    ax=ax_d,
    cmap="RdYlGn",
    cbar=True,
    center=0,
    vmin=-25,
    vmax=25,
    xticklabels=wrap_labels(canonical_order, width=12),
    yticklabels=qtype_short,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8, "pad": 0.02, "label": "Δpp"}
)

ax_d.set_title("(d) Δpp by question type × subdomain (n≥2)", fontweight='bold')
ax_d.set_xlabel("")
ax_d.set_ylabel("")
ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=40, ha="right", fontsize=8)
ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0, fontsize=9)

# =========================
# 7) 保存输出
# =========================
print(f"\n✅ 图片已保存:")
import os
print(f"   - {os.path.abspath(f'{OUT_PREFIX}.png')}")
print(f"   - {os.path.abspath(f'{OUT_PREFIX}.pdf')}")
plt.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.03)

# 额外保存一份到当前目录，方便查找
plt.savefig("Fig3_FINAL.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(f"   - {os.path.abspath('Fig3_FINAL.png')}")

plt.close(fig)
