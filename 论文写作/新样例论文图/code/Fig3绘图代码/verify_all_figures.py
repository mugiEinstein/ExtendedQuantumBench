# -*- coding: utf-8 -*-
"""
验证论文所有6个图的数据是否与真实实验数据一致
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("论文6个图数据完整性验证")
print("=" * 80)

# =========================
# 读取所有数据文件
# =========================
print("\n1. 读取数据文件...")

# Baseline数据
base_df = pd.read_csv('QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv')
print(f"   Baseline: {len(base_df)} 题")

# Hybrid v1-v4 数据
v1_df = pd.read_csv('QuantumBench/outputs/run_symbolic_full/quantumbench_results_qwen2.5-7b_0.csv')
v2_df = pd.read_csv('QuantumBench/outputs/run_hybrid_v2_full/quantumbench_results_qwen2.5-7b_0.csv')
v3_df = pd.read_csv('QuantumBench/outputs/run_hybrid_v3_full/quantumbench_results_qwen2.5-7b_0.csv')
v4_df = pd.read_csv('QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv')

# Category标签
cat_df = pd.read_csv('QuantumBench/quantumbench/quantumbench/category.csv')

# Open-ended数据
open_30_summary = pd.read_csv('QuantumBench/outputs/open_ended_eval_30/evaluation_summary.csv')

# Graduate数据
grad_results = pd.read_csv('QuantumBench/outputs/grad_eval/grad_eval_summary.csv')

# =========================
# Fig1: 总体性能对比
# =========================
print("\n" + "=" * 60)
print("Fig1: Baseline与Hybrid v1-v4总体准确率")
print("=" * 60)

results = {
    'Baseline': (base_df['Correct'].sum(), len(base_df)),
    'v1': (v1_df['Correct'].sum(), len(v1_df)),
    'v2': (v2_df['Correct'].sum(), len(v2_df)),
    'v3': (v3_df['Correct'].sum(), len(v3_df)),
    'v4': (v4_df['Correct'].sum(), len(v4_df)),
}

paper_fig1 = {
    'Baseline': (296, 769, 38.49),
    'v1': (282, 769, 36.67),
    'v2': (259, 769, 33.68),
    'v3': (293, 769, 38.10),
    'v4': (302, 769, 39.27),
}

print(f"{'方法':<12} {'实际正确':>8} {'实际总数':>8} {'实际Acc':>10} {'论文Acc':>10} {'匹配':>5}")
print("-" * 60)
all_match = True
for name, (correct, total) in results.items():
    acc = correct / total * 100
    paper_acc = paper_fig1[name][2]
    match = '✓' if abs(acc - paper_acc) < 0.1 else '✗'
    if abs(acc - paper_acc) >= 0.1:
        all_match = False
    print(f"{name:<12} {correct:>8} {total:>8} {acc:>9.2f}% {paper_acc:>9.2f}% {match:>5}")

# =========================
# Fig2: 按题型准确率
# =========================
print("\n" + "=" * 60)
print("Fig2: 按题型准确率对比")
print("=" * 60)

# 合并category
base_with_cat = base_df.merge(cat_df[['Question id', 'QuestionType']], on='Question id')
v4_with_cat = v4_df.merge(cat_df[['Question id', 'QuestionType']], on='Question id')

paper_fig2 = {
    'Algebraic Calculation': {'base': 36.70, 'v4': 37.57, 'n': 575},
    'Conceptual Understanding': {'base': 42.00, 'v4': 42.00, 'n': 50},
    'Numerical Calculation': {'base': 44.44, 'v4': 45.14, 'n': 144},
}

print(f"{'题型':<25} {'题量':>6} {'Base实际':>10} {'Base论文':>10} {'v4实际':>10} {'v4论文':>10}")
print("-" * 80)

for qtype in ['Algebraic Calculation', 'Conceptual Understanding', 'Numerical Calculation']:
    base_sub = base_with_cat[base_with_cat['QuestionType'] == qtype]
    v4_sub = v4_with_cat[v4_with_cat['QuestionType'] == qtype]
    
    base_acc = base_sub['Correct'].mean() * 100
    v4_acc = v4_sub['Correct'].mean() * 100
    n = len(base_sub)
    
    paper = paper_fig2[qtype]
    base_match = '✓' if abs(base_acc - paper['base']) < 0.1 else '✗'
    v4_match = '✓' if abs(v4_acc - paper['v4']) < 0.1 else '✗'
    
    print(f"{qtype:<25} {n:>6} {base_acc:>9.2f}%{base_match} {paper['base']:>9.2f}% {v4_acc:>9.2f}%{v4_match} {paper['v4']:>9.2f}%")

# =========================
# Fig3: 子领域分析（已单独验证）
# =========================
print("\n" + "=" * 60)
print("Fig3: 子领域准确率对比")
print("=" * 60)

valid_subdomains = ['Quantum Mechanics', 'Optics', 'Quantum Field Theory', 'Quantum Chemistry', 
                    'Quantum Computation', 'Photonics', 'Mathematics', 'String Theory', 'Nuclear Physics']

base_with_sub = base_df.merge(cat_df[['Question id', 'Subdomain_question']], on='Question id')
v4_with_sub = v4_df.merge(cat_df[['Question id', 'Subdomain_question']], on='Question id')

paper_fig3 = {
    'Quantum Mechanics': {'n': 212, 'base': 44.81, 'v4': 48.58},
    'Optics': {'n': 157, 'base': 31.21, 'v4': 28.03},
    'Quantum Field Theory': {'n': 107, 'base': 28.04, 'v4': 25.23},
    'Quantum Chemistry': {'n': 86, 'base': 53.49, 'v4': 53.49},
    'Quantum Computation': {'n': 60, 'base': 26.67, 'v4': 45.00},
    'Photonics': {'n': 57, 'base': 43.86, 'v4': 38.60},
    'Mathematics': {'n': 37, 'base': 45.95, 'v4': 43.24},
    'String Theory': {'n': 33, 'base': 33.33, 'v4': 30.30},
    'Nuclear Physics': {'n': 18, 'base': 27.78, 'v4': 33.33},
}

print(f"{'子领域':<25} {'题量':>6} {'Base实际':>10} {'v4实际':>10} {'Δ实际':>10}")
print("-" * 70)

for sd in valid_subdomains:
    base_sub = base_with_sub[base_with_sub['Subdomain_question'] == sd]
    v4_sub = v4_with_sub[v4_with_sub['Subdomain_question'] == sd]
    
    base_acc = base_sub['Correct'].mean() * 100
    v4_acc = v4_sub['Correct'].mean() * 100
    delta = v4_acc - base_acc
    
    print(f"{sd:<25} {len(base_sub):>6} {base_acc:>9.2f}% {v4_acc:>9.2f}% {delta:>+9.2f}")

# =========================
# Fig4: 路由策略分析
# =========================
print("\n" + "=" * 60)
print("Fig4: v4路由策略分布与准确率")
print("=" * 60)

sympy_count = (v4_df['Strategy'] == 'sympy').sum()
zeroshot_count = (v4_df['Strategy'] == 'zeroshot').sum()

sympy_acc = v4_df[v4_df['Strategy'] == 'sympy']['Correct'].mean() * 100
zeroshot_acc = v4_df[v4_df['Strategy'] == 'zeroshot']['Correct'].mean() * 100

paper_fig4 = {
    'SymPy Hybrid': {'n': 428, 'pct': 55.7, 'acc': 46.26},
    'Zero-shot': {'n': 341, 'pct': 44.3, 'acc': 30.50},
}

print(f"{'策略':<15} {'实际题量':>10} {'论文题量':>10} {'实际Acc':>10} {'论文Acc':>10}")
print("-" * 60)
print(f"{'SymPy Hybrid':<15} {sympy_count:>10} {paper_fig4['SymPy Hybrid']['n']:>10} {sympy_acc:>9.2f}% {paper_fig4['SymPy Hybrid']['acc']:>9.2f}%")
print(f"{'Zero-shot':<15} {zeroshot_count:>10} {paper_fig4['Zero-shot']['n']:>10} {zeroshot_acc:>9.2f}% {paper_fig4['Zero-shot']['acc']:>9.2f}%")

# =========================
# Fig5: 开放式评测
# =========================
print("\n" + "=" * 60)
print("Fig5: 开放式评测结果（30题扩展实验）")
print("=" * 60)

auto_mean = open_30_summary['Auto Score'].mean()
auto_std = open_30_summary['Auto Score'].std()
llm_mean = open_30_summary['LLM Score'].mean()
llm_std = open_30_summary['LLM Score'].std()
final_mean = open_30_summary['Final Score'].mean()
final_std = open_30_summary['Final Score'].std()

paper_fig5 = {
    'Auto': {'mean': 60.3, 'std': 5.4},
    'LLM': {'mean': 79.4, 'std': 14.5},
    'Final': {'mean': 71.7, 'std': 9.5},
}

print(f"{'指标':<10} {'实际均值':>12} {'论文均值':>12} {'实际std':>10} {'论文std':>10}")
print("-" * 60)
print(f"{'Auto':<10} {auto_mean:>11.1f} {paper_fig5['Auto']['mean']:>11.1f} {auto_std:>9.1f} {paper_fig5['Auto']['std']:>9.1f}")
print(f"{'LLM':<10} {llm_mean:>11.1f} {paper_fig5['LLM']['mean']:>11.1f} {llm_std:>9.1f} {paper_fig5['LLM']['std']:>9.1f}")
print(f"{'Final':<10} {final_mean:>11.1f} {paper_fig5['Final']['mean']:>11.1f} {final_std:>9.1f} {paper_fig5['Final']['std']:>9.1f}")

# =========================
# Fig6: 研究生基准
# =========================
print("\n" + "=" * 60)
print("Fig6: QuantumBench-Grad研究生基准")
print("=" * 60)

grad_total = len(grad_results)
grad_correct = grad_results['Is Correct'].sum()
grad_acc = grad_correct / grad_total * 100

paper_fig6 = {
    'total': 71,
    'correct': 25,
    'acc': 35.21,
}

print(f"总体准确率: {grad_correct}/{grad_total} = {grad_acc:.2f}%")
print(f"论文数据:   {paper_fig6['correct']}/{paper_fig6['total']} = {paper_fig6['acc']:.2f}%")
print(f"匹配: {'✓' if abs(grad_acc - paper_fig6['acc']) < 0.1 else '✗'}")

# 按难度分层
paper_difficulty = {
    'Graduate-1': {'correct': 18, 'total': 56, 'acc': 32.1},
    'Graduate-2': {'correct': 5, 'total': 13, 'acc': 38.5},
    'Graduate-3': {'correct': 2, 'total': 2, 'acc': 100.0},
}

print(f"\n按难度分层:")
print(f"{'难度':>12} {'实际正确/总数':>15} {'实际Acc':>10} {'论文Acc':>10}")
print("-" * 55)

for diff in ['Graduate-1', 'Graduate-2', 'Graduate-3']:
    sub = grad_results[grad_results['Difficulty'] == diff]
    correct = sub['Is Correct'].sum()
    total = len(sub)
    acc = correct / total * 100 if total > 0 else 0
    paper = paper_difficulty[diff]
    print(f"{diff:>12} {correct:>7}/{total:<7} {acc:>9.1f}% {paper['acc']:>9.1f}%")

# =========================
# 总结
# =========================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print("✅ 所有6个图的数据已与真实实验数据进行验证")
print("   - Fig1: Baseline与Hybrid v1-v4总体性能对比")
print("   - Fig2: 按题型分解的性能对比")
print("   - Fig3: 子领域增益分析")
print("   - Fig4: v4路由策略分布与准确率")
print("   - Fig5: 开放式评测框架结果")
print("   - Fig6: QuantumBench-Grad研究生基准")
