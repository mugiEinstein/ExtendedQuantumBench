# -*- coding: utf-8 -*-
"""
Fig3: Subdomain Analysis (a/b/c/d) — 4子图版本

使用与原始样例图完全一致的数据源和绘图逻辑
数据来源: subdomain_bootstrap_ci.csv (包含预计算的bootstrap CI)
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
# 使用原始样例图的预计算数据
SUBDOMAIN_CI_CSV = "论文写作/新样例论文图/data/subdomain_bootstrap_ci.csv"

# MCQ数据（用于计算Pareto和Heatmap）
MCQ_MASTER_CSV = "论文写作/新样例论文图/data/mcq_master.csv.gz"

# 输出路径
OUT_PREFIX = "论文写作/figures/Fig3_Subdomain_Analysis_v2"

# =========================
# 1) 工具函数
# =========================
def wrap_labels(labels, width=14):
    return [textwrap.fill(str(s), width=width) for s in labels]

# =========================
# 2) 读取数据
# =========================
# 子领域bootstrap CI数据（与原始样例图完全一致）
sub_ci = pd.read_csv(SUBDOMAIN_CI_CSV)

# MCQ master数据
mcq = pd.read_csv(MCQ_MASTER_CSV)

print("=== 子领域数据（来自原始样例图） ===")
print(sub_ci[['subdomain', 'n', 'baseline_acc', 'delta_pp', 'ci_lo', 'ci_hi']].to_string(index=False))

# =========================
# 3) 准备绘图数据
# =========================
# 按论文Fig3顺序排列（按delta_pp从高到低）
canonical_order = [
    "Quantum Computation",
    "Nuclear Physics",
    "Quantum Mechanics",
    "Quantum Chemistry",
    "Mathematics",
    "Quantum Field Theory",
    "Optics",
    "String Theory",
    "Photonics",
]

# 确保数据按canonical_order排列
sub_ci_ordered = sub_ci.set_index('subdomain').loc[canonical_order].reset_index()

# 提取绘图所需数组
delta_pp = sub_ci_ordered['delta_pp'].values
ci_low = sub_ci_ordered['ci_lo'].values
ci_high = sub_ci_ordered['ci_hi'].values
n_questions = sub_ci_ordered['n'].values
baseline_acc = sub_ci_ordered['baseline_acc'].values
subdomains = sub_ci_ordered['subdomain'].values

# =========================
# 4) Pareto数据（c子图）
# =========================
# 从MCQ数据计算v4的正确数
v4_data = mcq[mcq['method'] == 'v4'].copy()
known9 = canonical_order

new_correct_counts = (
    v4_data[v4_data['subdomain'].isin(known9)]
    .groupby('subdomain')['is_correct']
    .sum()
    .reindex(canonical_order)
    .fillna(0)
    .astype(int)
)

# 按正确数降序排列
order = np.argsort(new_correct_counts.values)[::-1]
sub_sorted = np.array(canonical_order)[order]
counts_sorted = new_correct_counts.values[order]
cum_pct_sorted = np.cumsum(counts_sorted) / max(1, counts_sorted.sum()) * 100.0

print("\n=== Pareto数据 ===")
for i, sd in enumerate(sub_sorted):
    print(f"{sd:<25} correct={counts_sorted[i]:>3}, cum%={cum_pct_sorted[i]:>5.1f}%")

# =========================
# 5) Heatmap数据（d子图）
# =========================
qtypes = ["Conceptual", "Numerical", "Algebraic"]
qtype_disp = {
    "Conceptual": "Conceptual",
    "Numerical": "Numerical",
    "Algebraic": "Algebraic",
}

# 创建pivot表计算delta
pivot = mcq.pivot(index="question_id", columns="method", values="is_correct")
meta = v4_data[['question_id', 'subdomain', 'qtype']].drop_duplicates()

# 合并计算delta
base = pivot['baseline'].reset_index().rename(columns={'baseline': 'base'})
vv4 = pivot['v4'].reset_index().rename(columns={'v4': 'v4'})
j = base.merge(vv4, on='question_id').merge(meta, on='question_id', how='left')
j['delta'] = (j['v4'] - j['base']) * 100

# 按题型和子领域分组计算delta_pp
cell = j[j['subdomain'].isin(known9)].groupby(['qtype', 'subdomain']).agg(
    delta_pp=('delta', 'mean'), 
    n=('delta', 'size')
).reset_index()

# 创建heatmap矩阵
delta_mat = np.full((len(qtypes), len(canonical_order)), np.nan)
for i, qt in enumerate(qtypes):
    for jj, sd in enumerate(canonical_order):
        cell_data = cell[(cell['qtype'] == qt) & (cell['subdomain'] == sd)]
        if len(cell_data) > 0 and cell_data['n'].values[0] >= 3:
            delta_mat[i, jj] = cell_data['delta_pp'].values[0]

print("\n=== Heatmap数据 ===")
for i, qt in enumerate(qtypes):
    for jj, sd in enumerate(canonical_order):
        if not np.isnan(delta_mat[i, jj]):
            print(f"{qt:<12} × {sd:<25} Δ={delta_mat[i, jj]:>+6.2f}")

# =========================
# 6) 绘图：4-panel layout
# =========================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(11.4, 8.8))

gs = GridSpec(
    nrows=3, ncols=2,
    height_ratios=[1.0, 1.0, 1.25],
    hspace=0.60,
    wspace=0.55,
    figure=fig
)

ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, :])
ax_c = fig.add_subplot(gs[2, 0])
ax_d = fig.add_subplot(gs[2, 1])

fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.22)

# ---------- (a) 子领域增益/损失 + 95% CI ----------
palette = sns.color_palette("tab10", n_colors=len(canonical_order))
y = np.arange(len(canonical_order))

# 画CI区间
for i in range(len(canonical_order)):
    ax_a.hlines(
        y=i,
        xmin=ci_low[i],
        xmax=ci_high[i],
        color=palette[i],
        lw=2,
        alpha=0.35
    )

# 画点
ax_a.scatter(delta_pp, y, c=palette, s=35, zorder=3)

# 0参考线
ax_a.axvline(0, color="steelblue", lw=1)

# 标注数值
for i, v in enumerate(delta_pp):
    ax_a.text(
        v + (0.55 if v >= 0 else -0.55),
        i,
        f"{v:+.2f}",
        va="center",
        ha="left" if v >= 0 else "right",
        fontsize=9,
        clip_on=False
    )

ax_a.set_yticks(y)
ax_a.set_yticklabels(subdomains)
ax_a.invert_yaxis()
ax_a.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)")
ax_a.set_xlabel("")

# ---------- (b) 增益 vs 基线准确率 ----------
sizes = (n_questions * 3).clip(30, None)

ax_b.scatter(baseline_acc, delta_pp, s=sizes, alpha=0.8)
ax_b.margins(x=0.08, y=0.18)
ax_b.axhline(0, color="steelblue", lw=1)

# 只标注特定子领域
for i, name in enumerate(subdomains):
    if name in ["Quantum Computation", "Photonics"]:
        ax_b.annotate(name, (baseline_acc[i], delta_pp[i]), 
                     xytext=(8, 4), textcoords="offset points", fontsize=9, clip_on=False)

ax_b.set_xlabel("Baseline accuracy (%)")
ax_b.set_ylabel("Δ Accuracy (pp)")
ax_b.set_title("(b) Gain vs baseline strength (bubble size ∝ #questions)")
ax_b.grid(True, alpha=0.3)

# ---------- (c) Pareto分析 ----------
x = np.arange(len(sub_sorted))

ax_c.bar(x, counts_sorted, color="#4C72B0")

ax_c_twin = ax_c.twinx()
ax_c_twin.plot(x, cum_pct_sorted, color="black", marker="o", lw=1.5)

ax_c.set_xticks(x)
ax_c.set_xticklabels(wrap_labels(sub_sorted, width=14), rotation=35, ha="right", fontsize=8)
ax_c.tick_params(axis="x", pad=2)
ax_c.set_ylabel("# Correct (new)")
ax_c_twin.set_ylabel("Cumulative (%)", labelpad=6)
ax_c.set_title("(c) Pareto: subdomain contribution")

# ---------- (d) 题型×子领域 Heatmap ----------
sns.heatmap(
    delta_mat,
    ax=ax_d,
    cmap="viridis",
    cbar=True,
    xticklabels=wrap_labels(canonical_order, width=14),
    yticklabels=[qtype_disp[q] for q in qtypes],
    annot=False,
    vmin=np.nanmin(delta_mat),
    vmax=np.nanmax(delta_mat),
    cbar_kws={"shrink": 0.95, "pad": 0.02}
)

ax_d.set_title("(d) Type × subdomain interaction (Δpp)")
ax_d.set_xlabel("")
ax_d.set_ylabel("")
ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0)

# =========================
# 7) 保存输出
# =========================
plt.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.02)

print(f"\n✅ 图片已保存:")
print(f"   - {OUT_PREFIX}.png")
print(f"   - {OUT_PREFIX}.pdf")

plt.close(fig)
