
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Build real master tables from QuantumBench outputs.
Usage:
  python build_master_tables_from_outputs.py --data_root /path/to/论文复现工作/QuantumBench/outputs --out_dir ./data
It will write:
  mcq_master.csv.gz
  openended_30_master.csv
  grad_master.csv
  subdomain_bootstrap_ci.csv

import argparse, os, glob, math
import pandas as pd
import numpy as np

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n) / denom
    return (center-half, center+half)

def find_one(patterns):
    for pat in patterns:
        xs = glob.glob(pat, recursive=True)
        if xs:
            return xs[0]
    raise FileNotFoundError(f"Could not find any file with patterns: {patterns}")

def read_run_csv(outputs_root, run_name):
    path = find_one([
        os.path.join(outputs_root, run_name, "quantumbench_results_*.csv"),
    ])
    return pd.read_csv(path), path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to QuantumBench/outputs")
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for CI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # MCQ runs (full)
    runs = {
        "baseline": "run_ollama",
        "v1": "run_symbolic_full",
        "v2": "run_hybrid_v2_full",
        "v3": "run_hybrid_v3_full",
        "v4": "run_hybrid_v4_full",
    }
    dfs = {}
    paths = {}
    for m, run in runs.items():
        df, p = read_run_csv(args.data_root, run)
        dfs[m] = df
        paths[m] = p

    # meta from v4 (contains question type)
    meta = dfs["v4"][["Question id","Subdomain","Question Type"]].copy()
    meta.columns = ["question_id","subdomain","question_type"]
    type_map = {
        "Algebraic Calculation": "Algebraic",
        "Numerical Calculation": "Numerical",
        "Conceptual Understanding": "Conceptual",
    }
    meta["qtype"] = meta["question_type"].map(type_map)

    def method_table(df, method):
        t = df[["Question id","Correct","Prompt tokens","Completion tokens","Cached tokens","Model response"]].copy()
        t.columns=["question_id","correct","prompt_tokens","completion_tokens","cached_tokens","model_response"]
        t["method"]=method
        t["tokens_total"]=t["prompt_tokens"]+t["completion_tokens"]
        return t

    mcq = pd.concat([method_table(dfs[m], m) for m in runs.keys()], ignore_index=True)

    # add strategy for v4
    strat = dfs["v4"][["Question id","Strategy"]].copy()
    strat.columns=["question_id","strategy"]
    mcq = mcq.merge(strat, on="question_id", how="left")
    mcq = mcq.merge(meta[["question_id","subdomain","qtype"]], on="question_id", how="left")
    mcq["is_correct"] = mcq["correct"].astype(int)

    mcq.to_csv(os.path.join(args.out_dir,"mcq_master.csv.gz"), index=False, compression="gzip")

    # Open-ended (n=30)
    open_summary_path = find_one([
        os.path.join(args.data_root, "open_ended_eval_30", "evaluation_summary.csv"),
    ])
    open30 = pd.read_csv(open_summary_path)
    open30.to_csv(os.path.join(args.out_dir,"openended_30_master.csv"), index=False)

    # Graduate
    grad_path = find_one([
        os.path.join(args.data_root, "grad_eval", "grad_eval_summary.csv"),
    ])
    grad = pd.read_csv(grad_path)
    grad.to_csv(os.path.join(args.out_dir,"grad_master.csv"), index=False)

    # Subdomain bootstrap CI (paired, baseline vs v4)
    pivot = mcq.pivot(index="question_id", columns="method", values="is_correct")
    base = pivot["baseline"]
    v4 = pivot["v4"]
    meta_sd = meta.set_index("question_id")["subdomain"]

    known9 = ["Mathematics","Nuclear Physics","Optics","Quantum Chemistry","Quantum Computation",
              "Quantum Field Theory","Quantum Mechanics","Photonics","String Theory"]

    rng = np.random.default_rng(args.seed)

    def bootstrap_delta(qids):
        qids = np.asarray(qids)
        deltas = np.empty(args.bootstrap, dtype=float)
        for i in range(args.bootstrap):
            samp = rng.choice(qids, size=len(qids), replace=True)
            deltas[i] = (v4.loc[samp].mean() - base.loc[samp].mean()) * 100.0
        return deltas.mean(), np.percentile(deltas, [2.5, 97.5])

    rows=[]
    for sd in known9:
        qids = meta_sd[meta_sd==sd].index.values
        mean, (lo,hi) = bootstrap_delta(qids)
        rows.append({
            "subdomain": sd,
            "n": int(len(qids)),
            "delta_pp": float(mean),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "baseline_acc": float(base.loc[qids].mean()*100.0),
            "v4_acc": float(v4.loc[qids].mean()*100.0),
        })
    out_df = pd.DataFrame(rows).sort_values("delta_pp")
    out_df.to_csv(os.path.join(args.out_dir,"subdomain_bootstrap_ci.csv"), index=False)

    print("Built master tables from real outputs:")
    for m,p in paths.items():
        print(f"  {m}: {p}")
    print(f"  open-ended (30): {open_summary_path}")
    print(f"  grad: {grad_path}")
    print(f"  wrote to: {args.out_dir}")

if __name__ == "__main__":
    main()
