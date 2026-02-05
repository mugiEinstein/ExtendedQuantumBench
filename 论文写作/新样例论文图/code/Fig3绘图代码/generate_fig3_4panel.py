# -*- coding: utf-8 -*-
"""
Fig3: Subdomain Analysis (a/b/c/d) — 使用原始代码逻辑，修正路径

直接复制原始Fig3绘图代码.py的逻辑，只修改路径
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import textwrap

# =========================
# 0) 路径配置（修正为正确路径）
# =========================
BASELINE_CSV = "QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv"
NEW_CSV = "QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv"
CATEGORY_CSV = "QuantumBench/quantumbench/quantumbench/category.csv"
OUT_PREFIX = "论文写作/figures/Fig3_Subdomain_Analysis"

# =========================
# 1) 工具函数
# =========================
def wrap_labels(labels, width=14):
    return [textwrap.fill(str(s), width=width) for s in labels]

# =========================
# 2) 读取数据
# =========================
base_df = pd.read_csv(BASELINE_CSV)
new_df = pd.read_csv(NEW_CSV)
cat_df = pd.read_csv(CATEGORY_CSV)
cat_df = cat_df[["Question id", "Subdomain_question", "QuestionType"]].copy()

# =========================
# 3) 对齐数据
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
# 4) 过滤：只保留9个规范子领域（>=3题）
# =========================
sub_counts = cat_df["Subdomain_question"].value_counts()
valid_subdomains = sub_counts[sub_counts >= 3].index.tolist()
m = m[m["Subdomain_question"].isin(valid_subdomains)].copy()

# 论文顺序
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
canonical_order = [sd for sd in canonical_order if sd in valid_subdomains]

# =========================
# 5) (a) 子领域 Δpp + paired bootstrap 95% CI
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

delta_pp = []
ci_low = []
ci_high = []
n_questions = []
baseline_acc = []

for sd in canonical_order:
    df_sd = m[m["Subdomain_question"] == sd]
    n_questions.append(len(df_sd))
    baseline_acc.append(df_sd["Correct_base"].mean() * 100.0)
    d, lo, hi = bootstrap_delta_ci(df_sd, B=4000)
    delta_pp.append(d)
    ci_low.append(lo)
    ci_high.append(hi)

delta_pp = np.array(delta_pp)
ci_low = np.array(ci_low)
ci_high = np.array(ci_high)
n_questions = np.array(n_questions)
baseline_acc = np.array(baseline_acc)

print("=== (a) 子领域数据 ===")
for i, sd in enumerate(canonical_order):
    print(f"{sd:<25} n={n_questions[i]:>3}, base={baseline_acc[i]:>5.2f}%, Δ={delta_pp[i]:>+6.2f}, CI=[{ci_low[i]:>+6.2f}, {ci_high[i]:>+6.2f}]")

# =========================
# 6) (c) Pareto
# =========================
new_correct_counts = (
    m.groupby("Subdomain_question")["Correct_new"]
     .sum()
     .reindex(canonical_order)
     .astype(int)
)

order = np.argsort(new_correct_counts.to_numpy())[::-1]
sub_sorted = np.array(canonical_order)[order]
counts_sorted = new_correct_counts.to_numpy()[order]
cum_pct_sorted = np.cumsum(counts_sorted) / max(1, counts_sorted.sum()) * 100.0

print("\n=== (c) Pareto数据 ===")
for i, sd in enumerate(sub_sorted):
    print(f"{sd:<25} correct={counts_sorted[i]:>3}, cum%={cum_pct_sorted[i]:>5.1f}%")

# =========================
# 7) (d) Heatmap
# =========================
qtypes = ["Conceptual Understanding", "Numerical Calculation", "Algebraic Calculation"]
qtype_disp = {
    "Conceptual Understanding": "Conceptual",
    "Numerical Calculation": "Numerical",
    "Algebraic Calculation": "Algebraic",
}

delta_mat = np.full((len(qtypes), len(canonical_order)), np.nan)

print("\n=== (d) Heatmap数据 ===")
for i, qt in enumerate(qtypes):
    for j, sd in enumerate(canonical_order):
        cell = m[(m["QuestionType"] == qt) & (m["Subdomain_question"] == sd)]
        if len(cell) >= 3:
            delta_val = (cell["Correct_new"].mean() - cell["Correct_base"].mean()) * 100.0
            delta_mat[i, j] = delta_val
            print(f"{qtype_disp[qt]:<12} × {sd:<25} n={len(cell):>3}, Δ={delta_val:>+6.2f}")

# =========================
# 8) 画图
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

# ---------- (a) ----------
palette = sns.color_palette("tab10", n_colors=len(canonical_order))
y = np.arange(len(canonical_order))

for i in range(len(canonical_order)):
    ax_a.hlines(y=i, xmin=ci_low[i], xmax=ci_high[i], color=palette[i], lw=2, alpha=0.35)

ax_a.scatter(delta_pp, y, c=palette, s=35, zorder=3)
ax_a.axvline(0, color="steelblue", lw=1)

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
ax_a.set_yticklabels(canonical_order)
ax_a.invert_yaxis()
ax_a.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)")
ax_a.set_xlabel("")

# ---------- (b) ----------
sizes = (n_questions * 10).clip(30, None)
ax_b.scatter(baseline_acc, delta_pp, s=sizes, alpha=0.7)
ax_b.axhline(0, color="steelblue", lw=1)

for i, name in enumerate(canonical_order):
    if n_questions[i] >= np.percentile(n_questions, 60):
        ax_b.text(baseline_acc[i] + 0.4, delta_pp[i], name, fontsize=9)

ax_b.set_xlabel("Baseline accuracy (%)")
ax_b.set_ylabel("Δ Accuracy (pp)")
ax_b.set_title("(b) Gain vs baseline accuracy (bubble size ∝ #questions)")

# ---------- (c) ----------
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

# ---------- (d) ----------
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
# 9) 保存
# =========================
plt.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.02)

print(f"\n✅ 图片已保存:")
print(f"   - {OUT_PREFIX}.png")
print(f"   - {OUT_PREFIX}.pdf")

plt.close(fig)
