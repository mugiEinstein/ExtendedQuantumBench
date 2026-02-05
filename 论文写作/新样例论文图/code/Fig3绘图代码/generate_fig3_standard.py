# -*- coding: utf-8 -*-
"""
Fig3: Subdomain Analysis — 论文标准2-subplot版本 (a)(b)

这是论文正文中使用的标准版本，包含：
- (a) 子领域增益/损失棒图 + 95% Bootstrap CI
- (b) 增益 vs 基线准确率气泡图

数据来源：
- Baseline: outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv
- v4: outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv
- 标签: quantumbench/category.csv (Subdomain_question)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# =========================
# 0) 路径配置
# =========================
BASELINE_CSV = "QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv"
V4_CSV = "QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv"
CATEGORY_CSV = "QuantumBench/quantumbench/quantumbench/category.csv"
OUT_PREFIX = "论文写作/figures/Fig3_Subdomain_Analysis"

# =========================
# 1) 读取并合并数据
# =========================
base_df = pd.read_csv(BASELINE_CSV)
v4_df = pd.read_csv(V4_CSV)
cat_df = pd.read_csv(CATEGORY_CSV)[['Question id', 'Subdomain_question']].copy()

# 合并数据
merged = (
    base_df[['Question id', 'Correct']]
    .rename(columns={'Correct': 'Correct_base'})
    .merge(
        v4_df[['Question id', 'Correct']].rename(columns={'Correct': 'Correct_v4'}),
        on='Question id'
    )
    .merge(cat_df, on='Question id')
)

# 9个规范子领域
known9 = [
    'Quantum Computation', 'Nuclear Physics', 'Quantum Mechanics', 'Quantum Chemistry',
    'Mathematics', 'Quantum Field Theory', 'Optics', 'String Theory', 'Photonics'
]
merged = merged[merged['Subdomain_question'].isin(known9)].copy()

print(f"=== 数据统计 ===")
print(f"合并后总题数: {len(merged)} (应为767)")

# =========================
# 2) Bootstrap 计算 delta_pp 和 95% CI
# =========================
rng = np.random.default_rng(42)

def bootstrap_delta_ci(df, B=2000):
    """计算paired bootstrap delta和95% CI"""
    d = df['Correct_v4'].astype(int).values - df['Correct_base'].astype(int).values
    n = len(d)
    delta = d.mean() * 100.0
    
    boot_deltas = np.empty(B)
    for i in range(B):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[i] = d[idx].mean() * 100.0
    
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return delta, ci_lo, ci_hi

# 按delta_pp从高到低排序（论文Fig3的顺序）
stats = []
for sd in known9:
    df_sd = merged[merged['Subdomain_question'] == sd]
    n = len(df_sd)
    baseline_acc = df_sd['Correct_base'].mean() * 100.0
    v4_acc = df_sd['Correct_v4'].mean() * 100.0
    delta, ci_lo, ci_hi = bootstrap_delta_ci(df_sd)
    stats.append({
        'subdomain': sd,
        'n': n,
        'baseline_acc': baseline_acc,
        'v4_acc': v4_acc,
        'delta_pp': delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi
    })

df_stats = pd.DataFrame(stats).sort_values('delta_pp', ascending=False)

print("\n=== 子领域统计 ===")
print(df_stats[['subdomain', 'n', 'baseline_acc', 'v4_acc', 'delta_pp']].to_string(index=False))

# 提取绘图数组
subdomains = df_stats['subdomain'].values
delta_pp = df_stats['delta_pp'].values
ci_lo = df_stats['ci_lo'].values
ci_hi = df_stats['ci_hi'].values
n_questions = df_stats['n'].values
baseline_acc = df_stats['baseline_acc'].values

# =========================
# 3) 绘图：标准2-subplot版本
# =========================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

fig = plt.figure(figsize=(10, 4.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.38)

# ---------- (a) 子领域增益/损失 + 95% CI ----------
ax_a = fig.add_subplot(gs[0, 0])
y = np.arange(len(subdomains))
palette = sns.color_palette("tab10", n_colors=len(subdomains))

# 画CI区间
for i in range(len(subdomains)):
    ax_a.hlines(y=i, xmin=ci_lo[i], xmax=ci_hi[i], color=palette[i], lw=2, alpha=0.5)

# 画0参考线到delta的水平线
ax_a.hlines(y, 0, delta_pp, lw=1.5, color=[palette[i] for i in range(len(subdomains))], alpha=0.7)

# 画点
ax_a.scatter(delta_pp, y, c=palette, s=50, zorder=3, edgecolors='white', linewidths=0.5)

# 0参考线
ax_a.axvline(0, color="gray", lw=1, linestyle='--', alpha=0.6)

# 标注数值
for i, v in enumerate(delta_pp):
    offset = 0.3 if v >= 0 else -0.3
    ax_a.text(v + offset, i, f"{v:+.2f}", va="center", 
              ha="left" if v >= 0 else "right", fontsize=9, fontweight='bold')

ax_a.set_yticks(y)
ax_a.set_yticklabels(subdomains)
ax_a.set_xlabel("Δ Accuracy (v4 - baseline, pp)")
ax_a.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)", fontweight='bold')
ax_a.grid(True, axis='x', alpha=0.3)

# ---------- (b) 增益 vs 基线准确率气泡图 ----------
ax_b = fig.add_subplot(gs[1, 0])

# 气泡大小与题量成正比
sizes = n_questions * 3
colors = [palette[i] for i in range(len(subdomains))]

ax_b.scatter(baseline_acc, delta_pp, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidths=0.5)
ax_b.axhline(0, color="gray", lw=1, linestyle='--', alpha=0.6)

# 标注关键子领域
for i, sd in enumerate(subdomains):
    if sd in ["Quantum Computation", "Photonics"]:
        ax_b.annotate(sd, (baseline_acc[i], delta_pp[i]), 
                     xytext=(8, 4), textcoords="offset points", fontsize=9, clip_on=False)

ax_b.set_xlabel("Baseline accuracy (%)")
ax_b.set_ylabel("Δ Accuracy (pp)")
ax_b.set_title("(b) Gain vs baseline strength (bubble size ∝ #questions)", fontweight='bold')
ax_b.grid(True, alpha=0.3)
ax_b.margins(x=0.08, y=0.18)

# =========================
# 4) 保存输出
# =========================
fig.tight_layout(pad=0.8)
fig.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.03)

print(f"\n✅ 标准2-subplot版本已保存:")
print(f"   - {OUT_PREFIX}.png")
print(f"   - {OUT_PREFIX}.pdf")

plt.close(fig)
