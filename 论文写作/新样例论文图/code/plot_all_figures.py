
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generate 6 main figures + 4 supplementary figures from real master tables.
# Usage:
#   python plot_all_figures.py --data_dir ./data --out_dir ./figures

import argparse, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

# ---- Global plotting defaults tuned for LaTeX single-column (ctexart, 12pt) ----
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.titlepad": 6,
})


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n) / denom
    return (center-half, center+half)

def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    # Tighten layout to avoid label/title overlap under LaTeX scaling.
    try:
        fig.tight_layout(pad=0.8)
    except Exception:
        pass
    fig.savefig(os.path.join(out_dir,f"{name}.png"), bbox_inches="tight", pad_inches=0.03, dpi=300)
    fig.savefig(os.path.join(out_dir,f"{name}.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def mcnemar_exact_p(b, c):
    # exact two-sided binomial test under p=0.5
    import scipy.stats as st
    n = b + c
    p = 2 * st.binom.cdf(min(b,c), n, 0.5)
    return min(p, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing master CSVs")
    ap.add_argument("--out_dir", default="figures", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    mcq = pd.read_csv(os.path.join(args.data_dir, "mcq_master.csv.gz"))
    open30 = pd.read_csv(os.path.join(args.data_dir, "openended_30_master.csv"))
    grad = pd.read_csv(os.path.join(args.data_dir, "grad_master.csv"))
    sub_ci = pd.read_csv(os.path.join(args.data_dir, "subdomain_bootstrap_ci.csv"))

    methods = ["baseline","v1","v2","v3","v4"]

    # Overall stats
    overall=[]
    for m in methods:
        d=mcq[mcq.method==m]
        n=len(d); k=int(d.is_correct.sum())
        lo,hi = wilson_ci(k,n)
        overall.append((m,k,n,100*k/n,100*lo,100*hi,d.tokens_total.mean()))

    # Pivot for McNemar
    pivot = mcq.pivot(index="question_id", columns="method", values="is_correct")
    b = int(((pivot["baseline"]==1)&(pivot["v4"]==0)).sum())
    c = int(((pivot["baseline"]==0)&(pivot["v4"]==1)).sum())
    a = int(((pivot["baseline"]==1)&(pivot["v4"]==1)).sum())
    d0 = int(((pivot["baseline"]==0)&(pivot["v4"]==0)).sum())
    p_exact = mcnemar_exact_p(b,c)

    # Fig1
    fig=plt.figure(figsize=(9.5,3.2))
    gs=fig.add_gridspec(1,2,width_ratios=[1.6,1.0],wspace=0.25)
    ax=fig.add_subplot(gs[0,0])
    x=np.arange(len(methods))
    acc=np.array([r[3] for r in overall])
    low=np.array([r[4] for r in overall])
    high=np.array([r[5] for r in overall])
    ax.errorbar(x, acc, yerr=[acc-low, high-acc], fmt='o', capsize=3)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30,45)
    ax.grid(True, axis='y', alpha=0.3)
    delta = overall[-1][3] - overall[0][3]
    ax.annotate(f"+{delta:.2f} pp", xy=(4,acc[-1]), xytext=(3.2, acc[-1]+2),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("(a) Overall MCQ accuracy (Wilson 95% CI), N=769")

    ax2=fig.add_subplot(gs[0,1]); ax2.axis('off')
    x0,y0=0.03,0.15; w,h=0.32,0.35  # shrink to leave right margin for text
    cells=[[a,b],[c,d0]]
    labels=[["B✓ V4✓","B✓ V4✗"],["B✗ V4✓","B✗ V4✗"]]
    for i in range(2):
        for j in range(2):
            ax2.add_patch(Rectangle((x0+j*w, y0+(1-i)*h), w, h, fill=False, lw=1.0))
            ax2.text(x0+j*w+w/2, y0+(1-i)*h+h/2, f"{labels[i][j]}\n{cells[i][j]}",
                     ha='center', va='center', fontsize=9)
    ax2.add_patch(Rectangle((x0+w, y0+h), w, h, fill=False, lw=2.0))
    ax2.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=2.0))
    ax2.text(0.70, 0.90, "(b) McNemar paired test\nBaseline vs v4", fontsize=10, ha='left', va='top')
    ax2.text(0.70, 0.72, f"b={b}, c={c}\nExact p={p_exact:.3f}", fontsize=10, ha='left', va='top')
    save_fig(fig,args.out_dir,"Fig1_Overall_Significance")

    # Fig2: type-wise
    type_stats=mcq.groupby(["method","qtype"])["is_correct"].agg(["mean","count"]).reset_index()
    type_pivot=type_stats.pivot(index="qtype", columns="method", values="mean")*100
    type_counts=type_stats.pivot(index="qtype", columns="method", values="count")
    qtypes=["Conceptual","Numerical","Algebraic"]

    # Layout: (a)(b) on the top row, (c) flowchart spans the full bottom row.
    fig=plt.figure(figsize=(10,5.8))
    gs=fig.add_gridspec(2,2, height_ratios=[1.0,1.05], width_ratios=[1.25,1.0], hspace=0.35, wspace=0.35)

    # (a) Baseline vs v3/v4 by type
    ax=fig.add_subplot(gs[0,0])
    y=np.arange(len(qtypes))
    for i,qt in enumerate(qtypes):
        bacc=float(type_pivot.loc[qt,"baseline"])
        v3acc=float(type_pivot.loc[qt,"v3"])
        v4acc=float(type_pivot.loc[qt,"v4"])
        # Connectors
        ax.plot([bacc, v3acc],[i,i], lw=1, alpha=0.5)
        ax.plot([bacc, v4acc],[i,i], lw=2, alpha=0.75)
        # Points
        ax.plot(bacc,i,'o',label="Baseline" if i==0 else "")
        ax.plot(v3acc,i,'s',label="v3" if i==0 else "")
        ax.plot(v4acc,i,'D',label="v4" if i==0 else "")
        # n label (baseline count)
        nqt=int(type_counts.loc[qt,'baseline'])
        xmax=max(bacc,v3acc,v4acc)
        ax.annotate(f"n={nqt}", (xmax, i), xytext=(10,0), textcoords="offset points",
                    va='center', fontsize=9, clip_on=False)

    ax.set_yticks(y,qtypes)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("(a) Type-wise accuracy (baseline vs v3/v4)")
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(type_pivot.loc[qtypes,["baseline","v3","v4"]].min().min()-1.0,
                type_pivot.loc[qtypes,["baseline","v3","v4"]].max().max()+3.0)
    ax.legend(frameon=False, loc="upper left", ncol=1, handletextpad=0.4)

    # (b) Net gain / loss by type
    ax2=fig.add_subplot(gs[0,1])
    d_v3=(type_pivot["v3"]-type_pivot["baseline"]).loc[qtypes]
    d_v4=(type_pivot["v4"]-type_pivot["baseline"]).loc[qtypes]
    yy=np.arange(len(qtypes))
    ax2.axvline(0, lw=1, alpha=0.5)
    ax2.plot(d_v3.values, yy, 's', label="v3-baseline")
    ax2.plot(d_v4.values, yy, 'D', label="v4-baseline")
    for i,qt in enumerate(qtypes):
        ax2.annotate(f"{d_v3.loc[qt]:+.2f}", (d_v3.loc[qt], i), xytext=(8,-10),
                     textcoords="offset points", fontsize=8, clip_on=False)
        ax2.annotate(f"{d_v4.loc[qt]:+.2f}", (d_v4.loc[qt], i), xytext=(8,6),
                     textcoords="offset points", fontsize=8, clip_on=False)
    ax2.set_yticks(yy,qtypes)
    ax2.set_xlabel("Δ Accuracy (pp)")
    ax2.set_title("(b) Net gain / loss by type")
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_xlim(min(d_v3.min(), d_v4.min())-2.0, max(d_v3.max(), d_v4.max())+3.0)

    # (c) Flowchart
    ax3=fig.add_subplot(gs[1,:]); ax3.axis('off')
    ax3.set_xlim(0,1); ax3.set_ylim(0,1)

    def box(ax, xy, text, w=0.48, h=0.16, fs=9):
        x,y=xy
        patch=FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=0.03", lw=1, facecolor="white")
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

    # Left: main pipeline
    box(ax3,(0.05,0.78),"MCQ question\n+ options", w=0.50, h=0.16, fs=10)
    box(ax3,(0.05,0.58),"Type classifier\n(Conceptual / Numerical / Algebraic)", w=0.50, h=0.16, fs=10)
    box(ax3,(0.05,0.38),"Selective gate\n(if eligible → SymPy; else → Zero-shot)", w=0.50, h=0.16, fs=10)
    box(ax3,(0.05,0.18),"Answer extraction\n& correctness", w=0.50, h=0.16, fs=10)

    # arrows
    for y1,y2 in [(0.78,0.58),(0.58,0.38),(0.38,0.18)]:
        ax3.add_patch(FancyArrowPatch((0.30,y1),(0.30,y2+0.16), arrowstyle='->', mutation_scale=14, lw=1))

    # Right: rules callout
    rules_txt = "Rules (v4):\n• Conceptual → Zero-shot\n• Numerical → SymPy\n• Algebraic → SymPy in {Math, QC, QChem, QM}"
    ax3.text(0.54,0.62, rules_txt, fontsize=10, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.9))
    ax3.set_title("(c) v4 gating mechanism", pad=6)
    save_fig(fig,args.out_dir,"Fig2_Typewise_Gating")


    # Fig3: subdomain
    df=sub_ci.sort_values("delta_pp")
    fig=plt.figure(figsize=(10,4.3))
    gs=fig.add_gridspec(2,1,height_ratios=[1.2,1.0],hspace=0.35)
    ax=fig.add_subplot(gs[0,0])
    y=np.arange(len(df))
    ax.hlines(y, 0, df["delta_pp"], lw=1.5)
    ax.plot(df["delta_pp"], y, 'o')
    for i,row in enumerate(df.itertuples()):
        ax.plot([row.ci_lo,row.ci_hi],[i,i], lw=1, alpha=0.7)
        ax.text(row.delta_pp + (0.3 if row.delta_pp>=0 else -0.3), i, f"{row.delta_pp:+.2f}",
                va='center', ha='left' if row.delta_pp>=0 else 'right', fontsize=9)
    ax.axvline(0, lw=1, alpha=0.6)
    ax.set_yticks(y, df["subdomain"])
    ax.set_xlabel("Δ Accuracy (v4 - baseline, pp)")
    ax.set_title("(a) Subdomain gains/losses with bootstrap 95% CI (paired)")
    ax.grid(True, axis='x', alpha=0.3)

    ax2=fig.add_subplot(gs[1,0])
    sizes=df["n"]*3
    ax2.scatter(df["baseline_acc"], df["delta_pp"], s=sizes, alpha=0.8)
    ax2.margins(x=0.08, y=0.18)
    for row in df.itertuples():
        if row.subdomain in ["Quantum Computation","Photonics"]:
            ax2.annotate(row.subdomain, (row.baseline_acc, row.delta_pp), xytext=(8,4), textcoords="offset points", fontsize=9, clip_on=False)
    ax2.axhline(0, lw=1, alpha=0.6)
    ax2.set_xlabel("Baseline accuracy (%)")
    ax2.set_ylabel("Δ Accuracy (pp)")
    ax2.set_title("(b) Gain vs baseline strength (bubble size ∝ #questions)")
    ax2.grid(True, alpha=0.3)
    save_fig(fig,args.out_dir,"Fig3_Subdomain_Analysis")

    # Fig4: efficiency tradeoff
    v4_rows=mcq[mcq.method=="v4"].copy()
    v4_rows["route"]=v4_rows["strategy"].fillna("unknown")
    route_stats=v4_rows.groupby("route")["is_correct"].agg(["mean","count"]).reset_index()
    route_stats["coverage"]=route_stats["count"]/len(v4_rows)

    fig=plt.figure(figsize=(10,3.2))
    gs=fig.add_gridspec(1,3,width_ratios=[1.05,1.15,1.05], wspace=0.60)
    ax=fig.add_subplot(gs[0,0])
    routes=["sympy","zeroshot"]
    rs=route_stats.set_index("route")
    cov=[rs.loc[r,"coverage"]*100 for r in routes]
    acc_r=[rs.loc[r,"mean"]*100 for r in routes]
    ax.bar(routes, cov, alpha=0.7)
    ax.set_ylabel("Coverage (%)", labelpad=2); ax.set_ylim(0,100)
    axr=ax.twinx()
    axr.plot(routes, acc_r, 'o-', lw=1.5)
    axr.set_ylabel("Accuracy (%)", labelpad=2); axr.tick_params(axis="y", pad=1, labelsize=8); axr.set_ylim(0,60)
    for i,r in enumerate(routes):
        ax.text(i, cov[i]+2, f"{cov[i]:.1f}%", ha='center', fontsize=9)
        axr.text(i, acc_r[i]+2, f"{acc_r[i]:.1f}%", ha='center', fontsize=9)
    ax.set_title("(a) v4 routing: coverage & accuracy")
    ax.grid(True, axis='y', alpha=0.3)

    ax=fig.add_subplot(gs[0,1])
    tokens_base=mcq[mcq.method=="baseline"]["tokens_total"]
    tokens_v4=mcq[mcq.method=="v4"]["tokens_total"]
    ax.boxplot([tokens_base, tokens_v4], labels=["Baseline","v4"], showfliers=False)
    ax.set_ylabel("Tokens per question")
    ax.set_title("(b) Token cost (per question)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(1, np.median(tokens_base), f"mean={tokens_base.mean():.0f}", ha='center', va='bottom', fontsize=9)
    ax.text(2, np.median(tokens_v4), f"mean={tokens_v4.mean():.0f}", ha='center', va='bottom', fontsize=9)

    ax=fig.add_subplot(gs[0,2])
    x=[r[6] for r in overall]; y=[r[3] for r in overall]
    ax.scatter(x,y)
    for i,m in enumerate(methods):
        ax.text(x[i]+5,y[i]+0.1,m,fontsize=9)
    ax.set_xlabel("Avg tokens per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) Accuracy–token tradeoff")
    ax.grid(True, alpha=0.3)
    save_fig(fig,args.out_dir,"Fig4_Efficiency_Tradeoff")

    # Fig5: open-ended
    N=len(open30)
    means=open30[["Auto Score","LLM Score","Final Score"]].mean()
    stds=open30[["Auto Score","LLM Score","Final Score"]].std()
    cis=1.96*stds/np.sqrt(N)
    corr=np.corrcoef(open30["Auto Score"], open30["LLM Score"])[0,1]

    fig=plt.figure(figsize=(10,6.0))
    gs=fig.add_gridspec(2,2,wspace=0.3,hspace=0.35)
    ax=fig.add_subplot(gs[0,0]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

    def flow_box(ax,x,y,text):
        patch=FancyBboxPatch((x,y),0.42,0.18, boxstyle="round,pad=0.02,rounding_size=0.03", lw=1, facecolor="white")
        ax.add_patch(patch)
        ax.text(x+0.21,y+0.09,text,ha='center',va='center',fontsize=10)

    flow_box(ax,0.05,0.68,"Open-ended\nquestion")
    flow_box(ax,0.53,0.68,"Model\nresponse")
    flow_box(ax,0.05,0.38,"AutoEvaluator\n(rubric)")
    flow_box(ax,0.53,0.38,"LLMJudge\n(rubric)")
    flow_box(ax,0.29,0.08,"Weighted fusion\nFinal = 0.4·Auto + 0.6·LLM")

    def arrow(ax, x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->', mutation_scale=12, lw=1))

    arrow(ax,0.47,0.77,0.53,0.77)
    arrow(ax,0.26,0.68,0.26,0.56)
    arrow(ax,0.74,0.68,0.74,0.56)
    arrow(ax,0.26,0.38,0.46,0.26)
    arrow(ax,0.74,0.38,0.54,0.26)
    ax.set_title("(a) Open-ended evaluation framework")

    ax=fig.add_subplot(gs[0,1])
    labels=["Auto","LLM","Final"]
    vals=[means["Auto Score"],means["LLM Score"],means["Final Score"]]
    errs=[cis["Auto Score"],cis["LLM Score"],cis["Final Score"]]
    x=np.arange(len(labels))
    ax.errorbar(x, vals, yerr=errs, fmt='o', capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_title(f"(b) Summary (mean ± 95% CI), N={N}")
    ax.grid(True, axis='y', alpha=0.3)

    ax=fig.add_subplot(gs[1,0])
    data=[open30["Auto Score"], open30["LLM Score"], open30["Final Score"]]
    ax.violinplot(data, showmeans=False, showmedians=True)
    ax.set_xticks([1,2,3], labels)
    ax.set_ylabel("Score")
    ax.set_title("(c) Score distributions (violin + median)")
    ax.grid(True, axis='y', alpha=0.3)

    ax=fig.add_subplot(gs[1,1])
    ax.scatter(open30["Auto Score"], open30["LLM Score"], alpha=0.8)
    mn=min(open30["Auto Score"].min(), open30["LLM Score"].min())
    mx=max(open30["Auto Score"].max(), open30["LLM Score"].max())
    ax.plot([mn,mx],[mn,mx], lw=1, alpha=0.6)
    ax.set_xlabel("Auto score")
    ax.set_ylabel("LLM score")
    ax.set_title(f"(d) Auto vs LLM agreement (r={corr:.2f})")
    ax.grid(True, alpha=0.3)
    save_fig(fig,args.out_dir,"Fig5_OpenEnded_Framework")

    # Fig6: graduate benchmark
    grad = grad.copy()
    grad["is_correct"] = grad["Is Correct"].astype(int)
    n=len(grad); k=int(grad["is_correct"].sum())
    acc=100*k/n
    lo,hi = wilson_ci(k,n)
    diff_stats=grad.groupby("Difficulty")["is_correct"].agg(["mean","count"]).reset_index()
    diff_stats["acc"]=diff_stats["mean"]*100
    dom_stats=grad.groupby("Domain")["is_correct"].agg(["mean","count"]).reset_index()
    dom_stats["acc"]=dom_stats["mean"]*100
    dom_f=dom_stats[dom_stats["count"]>=3].sort_values("acc")

    incorrect=grad[grad["is_correct"]==0].copy()
    incorrect["error_mode"]=np.where(incorrect["Stages Found"]<=4,"Low-stage reasoning","Wrong final answer")
    err_stats=(incorrect["error_mode"].value_counts(normalize=True)*100).sort_values()

    fig=plt.figure(figsize=(10,6.0))
    gs=fig.add_gridspec(2,2,wspace=0.35,hspace=0.4)

    ax=fig.add_subplot(gs[0,0])
    ax.errorbar([0],[acc], yerr=[[acc-100*lo],[100*hi-acc]], fmt='o', capsize=4)
    ax.set_xlim(-0.5,0.5)
    ax.set_xticks([0],["Graduate\nBenchmark"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0,100)
    ax.grid(True, axis='y', alpha=0.3)
    ax2=ax.twinx()
    ax2.bar([0.3],[grad["Stages Found"].mean()], width=0.2, alpha=0.4)
    ax2.set_ylabel("Avg stages found")
    ax.set_title(f"(a) Overall accuracy (Wilson 95% CI), N={n}")

    ax=fig.add_subplot(gs[0,1])
    order=["Graduate-1","Graduate-2","Graduate-3"]
    accs=[float(diff_stats.set_index("Difficulty").loc[o,"acc"]) for o in order if o in diff_stats.set_index("Difficulty").index]
    ns=[int(diff_stats.set_index("Difficulty").loc[o,"count"]) for o in order if o in diff_stats.set_index("Difficulty").index]
    yy=np.arange(len(accs))
    ax.plot(accs, yy,'o-')
    for i in range(len(accs)):
        ax.text(accs[i]+1, i, f"n={ns[i]}", va='center', fontsize=9)
    ax.set_yticks(yy, [o for o in order if o in diff_stats.set_index("Difficulty").index])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("(b) Accuracy by difficulty")
    ax.grid(True, axis='x', alpha=0.3)

    ax=fig.add_subplot(gs[1,0])
    y=np.arange(len(dom_f))
    ax.hlines(y, 0, dom_f["acc"], lw=1.5)
    ax.plot(dom_f["acc"], y, 'o')
    for i,row in enumerate(dom_f.itertuples()):
        ax.text(row.acc+1, i, f"n={row.count}", va='center', fontsize=8)
    ax.set_yticks(y, dom_f["Domain"])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("(c) Accuracy by domain (n ≥ 3)")
    ax.grid(True, axis='x', alpha=0.3)

    ax=fig.add_subplot(gs[1,1])
    ax.barh(err_stats.index.tolist(), err_stats.values)
    ax.set_xlabel("Share among incorrect (%)")
    ax.set_title("(d) Error mode breakdown (heuristic, from summary fields)")
    ax.grid(True, axis='x', alpha=0.3)
    save_fig(fig,args.out_dir,"Fig6_Graduate_Benchmark")

    # Supplement: heatmap type x subdomain (delta)
    known9 = ["Mathematics","Nuclear Physics","Optics","Quantum Chemistry","Quantum Computation",
              "Quantum Field Theory","Quantum Mechanics","Photonics","String Theory"]
    meta = mcq[mcq.method=="v4"][["question_id","subdomain","qtype"]].drop_duplicates()
    base = pivot["baseline"].reset_index().rename(columns={"baseline":"base"})
    vv4 = pivot["v4"].reset_index().rename(columns={"v4":"v4"})
    j = base.merge(vv4, on="question_id").merge(meta, on="question_id", how="left")
    j["delta"]=(j["v4"]-j["base"])*100
    cell=j[j.subdomain.isin(known9)].groupby(["qtype","subdomain"]).agg(delta_pp=("delta","mean"), n=("delta","size")).reset_index()
    heat=cell.pivot(index="qtype", columns="subdomain", values="delta_pp").loc[["Conceptual","Numerical","Algebraic"], known9]
    nmat=cell.pivot(index="qtype", columns="subdomain", values="n").loc[["Conceptual","Numerical","Algebraic"], known9]

    fig=plt.figure(figsize=(10,3.2))
    ax=fig.add_subplot(111)
    data=heat.values.astype(float)
    masked=np.ma.masked_invalid(data)
    im=ax.imshow(masked, aspect='auto')
    ax.set_yticks(np.arange(heat.shape[0]), heat.index)
    ax.set_xticks(np.arange(heat.shape[1]), heat.columns, rotation=45, ha='right')
    ax.set_title("Supp. Fig.1  Δ Accuracy (v4 - baseline, pp) by question type × subdomain")
    for i in range(heat.shape[0]):
        for j0 in range(heat.shape[1]):
            val=heat.iloc[i,j0]
            nn=nmat.iloc[i,j0]
            if pd.isna(val):
                continue
            ax.text(j0,i,f"{val:+.1f}\n(n={int(nn)})",ha='center',va='center',fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Δpp")
    save_fig(fig,args.out_dir,"Supp_Fig1_TypeSubdomain_Heatmap")

    # Supp Fig2: CI forest
    df=sub_ci.sort_values("delta_pp")
    fig=plt.figure(figsize=(10,4.0))
    ax=fig.add_subplot(111)
    y=np.arange(len(df))
    ax.errorbar(df["delta_pp"], y, xerr=[df["delta_pp"]-df["ci_lo"], df["ci_hi"]-df["delta_pp"]], fmt='o', capsize=3)
    ax.axvline(0, lw=1, alpha=0.6)
    ax.set_yticks(y, df["subdomain"])
    ax.set_xlabel("Δ Accuracy (pp), bootstrap 95% CI")
    ax.set_title("Supp. Fig.2  Subdomain deltas with paired bootstrap CI")
    for i,row in enumerate(df.itertuples()):
        ax.text(row.delta_pp, i+0.15, f"n={row.n}", fontsize=8, ha='center')
    ax.grid(True, axis='x', alpha=0.3)
    save_fig(fig,args.out_dir,"Supp_Fig2_Bootstrap_CI")

    # Supp Fig3: gating validity (baseline vs full tool vs selective)
    sel=["baseline","v1","v4"]
    xs=[dict((r[0],r[6]) for r in overall)[m] for m in sel]
    ys=[dict((r[0],r[3]) for r in overall)[m] for m in sel]
    labels={"baseline":"Baseline (zero-shot)","v1":"Full SymPy","v4":"Selective gate"}
    fig=plt.figure(figsize=(10,4.0))
    ax=fig.add_subplot(111)
    ax.scatter(xs,ys)
    ax.margins(x=0.06, y=0.12)
    for i,m in enumerate(sel):
        ax.annotate(labels[m], (xs[i],ys[i]), xytext=(8,6), textcoords="offset points", fontsize=9, clip_on=False)
    ax.set_xlabel("Avg tokens per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Supp. Fig.3  Gating validity: selective vs full tool use", pad=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig,args.out_dir,"Supp_Fig3_Gating_Validity")

    # Supp Fig4: open-ended stability via bootstrapped subsampling
    rng=np.random.default_rng(args.seed)
    scores=open30["Final Score"].to_numpy()
    ks=range(5,31)
    reps=800
    rows=[]
    for k in ks:
        means=[]
        for _ in range(reps):
            samp=rng.choice(scores, size=k, replace=True)
            means.append(float(np.mean(samp)))
        means=np.array(means)
        rows.append((k, float(np.median(means)), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))))
    stab=pd.DataFrame(rows, columns=["k","median","lo","hi"])
    fig=plt.figure(figsize=(10,4.0))
    ax=fig.add_subplot(111)
    ax.plot(stab["k"].values, stab["median"].values, marker='o', lw=1.5, label="Median of bootstrap mean")
    ax.fill_between(stab["k"].values, stab["lo"].values, stab["hi"].values, alpha=0.2, label="95% interval")
    ax.set_xlabel("Sample size k (bootstrap from N=30)")
    ax.set_ylabel("Mean Final score")
    ax.set_title("Supp. Fig.4  Open-ended score stability vs sample size")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig,args.out_dir,"Supp_Fig4_OpenEnded_Stability")

    print(f"Saved figures to: {args.out_dir}")

if __name__ == "__main__":
    main()