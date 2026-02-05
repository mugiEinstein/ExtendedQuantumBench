# -*- coding: utf-8 -*-
"""
验证Fig3数据并生成精确的图表
"""

import pandas as pd
import numpy as np

# 读取数据
base_df = pd.read_csv('QuantumBench/outputs/run_ollama/quantumbench_results_qwen2.5-7b_0.csv')
new_df = pd.read_csv('QuantumBench/outputs/run_hybrid_v4_full/quantumbench_results_qwen2.5-7b_0.csv')
cat_df = pd.read_csv('QuantumBench/quantumbench/quantumbench/category.csv')

# 合并数据
base_keep = base_df[['Question id', 'Correct']].copy()
new_keep = new_df[['Question id', 'Correct']].copy()
cat_keep = cat_df[['Question id', 'Subdomain_question', 'QuestionType']].copy()

base_labeled = base_keep.merge(cat_keep, on='Question id', how='left')
new_labeled = new_keep.merge(cat_keep, on='Question id', how='left')

m = base_labeled.merge(new_labeled, on=['Question id', 'Subdomain_question', 'QuestionType'], suffixes=('_base', '_new'))

# 过滤掉异常子领域(Algebraic Calculation只有2题)
valid_subdomains = ['Quantum Mechanics', 'Optics', 'Quantum Field Theory', 'Quantum Chemistry', 
                    'Quantum Computation', 'Photonics', 'Mathematics', 'String Theory', 'Nuclear Physics']
m_valid = m[m['Subdomain_question'].isin(valid_subdomains)].copy()

print(f'过滤后总题数: {len(m_valid)} (应为767)')

print('\n=== 按子领域统计（论文表5-4数据）===')
print(f"子领域                    题量    Baseline       v4      Δ(pp)")
print('-'*70)

results = []
for sd in valid_subdomains:
    df_sd = m_valid[m_valid['Subdomain_question'] == sd]
    n = len(df_sd)
    base_acc = df_sd['Correct_base'].mean() * 100
    new_acc = df_sd['Correct_new'].mean() * 100
    delta = new_acc - base_acc
    results.append({
        'subdomain': sd,
        'n': n,
        'base_acc': base_acc,
        'new_acc': new_acc,
        'delta': delta
    })
    print(f'{sd:25} {n:>5}   {base_acc:>8.2f}%  {new_acc:>8.2f}%  {delta:>+8.2f}')

print('\n=== 论文表5-4对比验证 ===')
paper_data = {
    'Quantum Mechanics': {'n': 212, 'base': 44.81, 'v4': 48.58, 'delta': 3.77},
    'Optics': {'n': 157, 'base': 31.21, 'v4': 28.03, 'delta': -3.18},
    'Quantum Field Theory': {'n': 107, 'base': 28.04, 'v4': 25.23, 'delta': -2.80},
    'Quantum Chemistry': {'n': 86, 'base': 53.49, 'v4': 53.49, 'delta': 0.00},
    'Quantum Computation': {'n': 60, 'base': 26.67, 'v4': 45.00, 'delta': 18.33},
    'Photonics': {'n': 57, 'base': 43.86, 'v4': 38.60, 'delta': -5.26},
    'Mathematics': {'n': 37, 'base': 45.95, 'v4': 43.24, 'delta': -2.70},
    'String Theory': {'n': 33, 'base': 33.33, 'v4': 30.30, 'delta': -3.03},
    'Nuclear Physics': {'n': 18, 'base': 27.78, 'v4': 33.33, 'delta': 5.56},
}

print(f"\n{'子领域':25} {'实际N':>6} {'论文N':>6} {'实际Base':>10} {'论文Base':>10} {'实际v4':>10} {'论文v4':>10}")
print('-'*95)

all_match = True
for r in results:
    sd = r['subdomain']
    paper = paper_data.get(sd, {})
    n_match = '✓' if r['n'] == paper.get('n') else '✗'
    base_match = '✓' if abs(r['base_acc'] - paper.get('base', 0)) < 0.1 else '✗'
    v4_match = '✓' if abs(r['new_acc'] - paper.get('v4', 0)) < 0.1 else '✗'
    
    print(f"{sd:25} {r['n']:>5}{n_match}  {paper.get('n', 0):>5}   {r['base_acc']:>8.2f}%{base_match} {paper.get('base', 0):>8.2f}%  {r['new_acc']:>8.2f}%{v4_match} {paper.get('v4', 0):>8.2f}%")
    
    if r['n'] != paper.get('n') or abs(r['base_acc'] - paper.get('base', 0)) >= 0.1 or abs(r['new_acc'] - paper.get('v4', 0)) >= 0.1:
        all_match = False

if all_match:
    print('\n✅ 所有数据与论文完全一致！')
else:
    print('\n⚠️ 存在数据不一致，请检查！')
