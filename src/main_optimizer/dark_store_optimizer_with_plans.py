"""
前置仓SKU选品与库存配置联合优化
========================================
依赖: pip install ortools pandas numpy scipy matplotlib

输入文件（由 dark_store_data_prep.py 生成）:
  - sku_params.csv    : SKU参数（需求分布、成本、体积/重量）
  - daily_demand.csv  : 每日需求矩阵（用于SAA与Rolling Window实验）

三种算法:
  1. Greedy        : 贪心启发式（极快，可解释）
  2. DP_Lagrangian : 拉格朗日松弛动态规划（中等规模）
  3. MILP_CpSAT    : OR-Tools CP-SAT 整数规划（最优基准）

实验设计:
  - 3组情景 × Rolling Window（每周滑动，用前N周历史估计，预测下1周）
  - 每组情景下对比4种方法（含鲁棒均值版本）
  - 输出：满足率、缺货率、目标值、运行时间对比表 + 图表
  - 新增输出：plans.csv（每个窗口、每个算法的SKU备货方案）
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from scipy.stats import nbinom, poisson

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results" / "main_optimizer"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 0. 全局参数
# ══════════════════════════════════════════
Q_MAX = 15
V_CAP = 800.0
W_CAP = 400.0
SCALE = 100
N_SKU = 500
WINDOW_TRAIN = 8
WINDOW_TEST = 1
ROBUST_Z = 1.645

# ══════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════
def load_data(sku_path=None, demand_path=None, n_sku=N_SKU):
    if sku_path is None:
        sku_path = DATA_DIR / 'sku_params.csv'
    if demand_path is None:
        demand_path = DATA_DIR / 'daily_demand.csv'

    params = pd.read_csv(sku_path)
    params = params.nlargest(n_sku, 'total_qty').reset_index(drop=True)

    demand = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    demand = demand[[c for c in params['SKU'] if c in demand.columns]]

    params = params[params['SKU'].isin(demand.columns)].reset_index(drop=True)
    demand = demand[params['SKU'].tolist()]

    weekly = demand.resample('W').sum()
    return params, weekly

# ══════════════════════════════════════════
# 2. 期望价值系数预计算
# ══════════════════════════════════════════
def compute_E_min(mu, r, use_negbin, q_max):
    e_min = np.zeros(q_max + 1)
    for q in range(1, q_max + 1):
        s = 0.0
        for k in range(q):
            if use_negbin and r < 9000:
                p = r / (r + mu) if (r + mu) > 0 else 1.0
                cdf_k = nbinom.cdf(k, r, p)
            else:
                cdf_k = poisson.cdf(k, mu)
            s += (1.0 - cdf_k)
        e_min[q] = s
    return e_min


def compute_G(params, mu_override=None, q_max=Q_MAX):
    N = len(params)
    G = np.zeros((N, q_max + 1))

    for i, row in params.iterrows():
        mu = mu_override[i] if mu_override is not None else row['mu_weekly']
        r = row['r_negbin']
        nb = bool(row['use_negbin'])
        u, h, b, f = row['u_i'], row['h_i'], row['b_i'], row['f_i']

        e_min = compute_E_min(mu, r, nb, q_max)
        for q in range(q_max + 1):
            if q == 0:
                G[i, q] = 0.0
            else:
                G[i, q] = (u + h + b) * e_min[q] - h * q - b * mu - f

    return G


def compute_resource(params, q_max=Q_MAX):
    N = len(params)
    s_v = np.ones(N) * 2.0
    s_w = np.ones(N) * 0.5

    A_V = np.zeros((N, q_max + 1))
    A_W = np.zeros((N, q_max + 1))
    for i, row in params.iterrows():
        for q in range(1, q_max + 1):
            A_V[i, q] = s_v[i] + row['v_i'] * q
            A_W[i, q] = s_w[i] + row['w_i'] * q
    return A_V, A_W

# ══════════════════════════════════════════
# 3. 评估函数（解析期望）
# ══════════════════════════════════════════
def evaluate(params, q_vec, mu_true=None):
    N = len(params)
    total_satisfied = 0.0
    total_demand = 0.0
    total_stockout = 0.0
    n_listed = 0

    for i, row in params.iterrows():
        q = int(q_vec[i])
        mu = mu_true[i] if mu_true is not None else row['mu_weekly']
        r = row['r_negbin']
        nb = bool(row['use_negbin'])

        if q == 0:
            total_demand += mu
            total_stockout += 1.0
            continue

        n_listed += 1
        e_min = compute_E_min(mu, r, nb, q)[q]
        if nb and r < 9000:
            p = r / (r + mu) if (r + mu) > 0 else 1.0
            p_stockout = 1 - nbinom.cdf(q, r, p)
        else:
            p_stockout = 1 - poisson.cdf(q, mu)

        total_satisfied += e_min
        total_demand += mu
        total_stockout += p_stockout

    line_fill_rate = total_satisfied / total_demand if total_demand > 0 else 0.0
    avg_stockout = total_stockout / N
    return {
        'line_fill_rate': line_fill_rate,
        'avg_stockout': avg_stockout,
        'n_listed': n_listed,
    }

# ══════════════════════════════════════════
# 4. 算法实现
# ══════════════════════════════════════════
def greedy(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX):
    N = len(params)
    q_star = np.argmax(G, axis=1)

    density = np.zeros(N)
    for i in range(N):
        q = q_star[i]
        if q == 0 or G[i, q] <= 0:
            density[i] = -np.inf
            continue
        cap_use = A_V[i, q] / v_cap + A_W[i, q] / w_cap + 1e-9
        density[i] = G[i, q] / cap_use

    order = np.argsort(-density)
    q_vec = np.zeros(N, dtype=int)
    v_used, w_used = 0.0, 0.0

    for i in order:
        if density[i] == -np.inf:
            continue
        for q in range(q_star[i], 0, -1):
            if G[i, q] <= 0:
                break
            if v_used + A_V[i, q] <= v_cap and w_used + A_W[i, q] <= w_cap:
                q_vec[i] = q
                v_used += A_V[i, q]
                w_used += A_W[i, q]
                break
    return q_vec


def dp_lagrangian(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX, n_lambda=20):
    N = len(params)
    V_int = int(v_cap)

    def solve_dp(lam):
        G_pen = G - lam * A_W
        INF = -1e18
        dp = np.full(V_int + 1, INF)
        dp[0] = 0.0
        choice = np.zeros((N, V_int + 1), dtype=np.int8)

        for i in range(N):
            new_dp = np.full(V_int + 1, INF)
            for v in range(V_int + 1):
                if dp[v] == INF:
                    continue
                for q in range(q_max + 1):
                    av = int(A_V[i, q])
                    if v + av > V_int:
                        continue
                    val = dp[v] + G_pen[i, q]
                    if val > new_dp[v + av]:
                        new_dp[v + av] = val
                        choice[i, v + av] = q
            dp = new_dp

        best_v = int(np.argmax(dp))
        q_vec = np.zeros(N, dtype=int)
        v_rem = best_v
        for i in range(N - 1, -1, -1):
            q_vec[i] = choice[i, v_rem]
            v_rem -= int(A_V[i, q_vec[i]])

        w_used = sum(A_W[i, q_vec[i]] for i in range(N))
        return q_vec, w_used

    lam_lo, lam_hi = 0.0, 5.0
    best_q = None

    for _ in range(n_lambda):
        lam_mid = (lam_lo + lam_hi) / 2
        q_vec, w_used = solve_dp(lam_mid)
        if w_used <= w_cap:
            best_q = q_vec.copy()
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid

    if best_q is None:
        best_q, _ = solve_dp(lam_hi)
        for i in range(N):
            if best_q[i] > 0:
                while best_q[i] > 0 and (sum(A_W[j, best_q[j]] for j in range(N)) > w_cap):
                    best_q[i] -= 1

    return best_q


def milp_cpsat(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX, time_limit=60, scale=SCALE):
    N = len(params)
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f'x_{i}_{q}') for q in range(q_max + 1)] for i in range(N)]

    for i in range(N):
        model.AddExactlyOne(x[i])

    model.Add(
        sum(int(A_V[i, q] * scale) * x[i][q] for i in range(N) for q in range(q_max + 1))
        <= int(v_cap * scale)
    )

    model.Add(
        sum(int(A_W[i, q] * scale) * x[i][q] for i in range(N) for q in range(q_max + 1))
        <= int(w_cap * scale)
    )

    obj_terms = []
    for i in range(N):
        for q in range(q_max + 1):
            coef = int(G[i, q] * scale)
            obj_terms.append(coef * x[i][q])
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    q_vec = np.zeros(N, dtype=int)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            for q in range(q_max + 1):
                if solver.Value(x[i][q]) == 1:
                    q_vec[i] = q
                    break
    else:
        print(f"  [CP-SAT] 未找到可行解，状态: {solver.StatusName(status)}")

    return q_vec, solver.StatusName(status)

# ══════════════════════════════════════════
# 5. 辅助：保存方案
# ══════════════════════════════════════════
def extract_plan_records(params, q_vec, window_id, scenario_name, algorithm_key):
    records = []
    for i, sku in enumerate(params['SKU']):
        q = int(q_vec[i])
        if q > 0:
            records.append({
                'window': window_id,
                'scenario': scenario_name,
                'algorithm': algorithm_key,
                'sku': sku,
                'stock_qty': q,
            })
    return records

# ══════════════════════════════════════════
# 6. Rolling Window 实验框架
# ══════════════════════════════════════════
def rolling_window_experiment(params, weekly_demand,
                              train_weeks=WINDOW_TRAIN,
                              test_weeks=WINDOW_TEST,
                              scenario_name='baseline',
                              q_max=Q_MAX, v_cap=V_CAP, w_cap=W_CAP):
    n_weeks = len(weekly_demand)
    results = []
    plans_records = []
    windows = list(range(0, n_weeks - train_weeks - test_weeks + 1))
    print(f"\n  [Rolling] 共 {len(windows)} 个窗口")

    for w_idx, w_start in enumerate(windows):
        train = weekly_demand.iloc[w_start: w_start + train_weeks]
        test = weekly_demand.iloc[w_start + train_weeks: w_start + train_weeks + test_weeks]

        mu_est = train.mean().values
        se_est = train.std().values / np.sqrt(len(train))
        mu_rob = mu_est + ROBUST_Z * se_est
        mu_true = test.mean().values

        params_now = params.copy()
        params_now['mu_weekly'] = mu_est

        G_nom = compute_G(params_now, mu_override=mu_est, q_max=q_max)
        G_rob = compute_G(params_now, mu_override=mu_rob, q_max=q_max)
        A_V, A_W = compute_resource(params_now, q_max=q_max)

        row = {'window': w_idx, 'scenario': scenario_name}

        for method_name, G_use in [('nominal', G_nom), ('robust', G_rob)]:
            for alg_name in ['greedy', 'dp', 'cpsat']:
                t0 = time.time()

                if alg_name == 'greedy':
                    q_vec = greedy(params_now, G_use, A_V, A_W, v_cap, w_cap, q_max)
                elif alg_name == 'dp':
                    q_vec = dp_lagrangian(params_now, G_use, A_V, A_W, v_cap, w_cap, q_max)
                else:
                    q_vec, _ = milp_cpsat(params_now, G_use, A_V, A_W, v_cap, w_cap, q_max)

                elapsed = time.time() - t0
                metrics = evaluate(params_now, q_vec, mu_true=mu_true)

                key = f'{alg_name}_{method_name}'
                row[f'{key}_lfr'] = metrics['line_fill_rate']
                row[f'{key}_stockout'] = metrics['avg_stockout']
                row[f'{key}_time'] = elapsed
                row[f'{key}_n_listed'] = metrics['n_listed']

                plans_records.extend(
                    extract_plan_records(
                        params=params_now,
                        q_vec=q_vec,
                        window_id=w_idx,
                        scenario_name=scenario_name,
                        algorithm_key=key,
                    )
                )

        results.append(row)
        if (w_idx + 1) % 5 == 0:
            print(f"    完成窗口 {w_idx + 1}/{len(windows)}")

    return pd.DataFrame(results), pd.DataFrame(plans_records)

# ══════════════════════════════════════════
# 7. 三组实验情景
# ══════════════════════════════════════════
def run_all_scenarios(params_full, weekly_demand_full):
    all_results = []
    all_plans = []

    params_full = params_full.sort_values('total_qty', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 55)
    print("情景1: Baseline (Top 200 SKU, V=800, W=400)")
    print("=" * 55)
    params_200 = params_full.head(200).reset_index(drop=True)
    demand_200 = weekly_demand_full[params_200['SKU'].tolist()]
    params_200['mu_weekly'] = demand_200.mean().values
    print(f"  实际SKU数: {len(params_200)}")

    df1, plans1 = rolling_window_experiment(
        params_200, demand_200, scenario_name='baseline', v_cap=800, w_cap=400
    )
    all_results.append(df1)
    all_plans.append(plans1)

    print("\n" + "=" * 55)
    print("情景2: 高需求波动 (Top 200 NegBin SKU only)")
    print("=" * 55)
    params_nb = (
        params_full[params_full['use_negbin'] == 1]
        .nlargest(200, 'total_qty')
        .reset_index(drop=True)
    )
    if len(params_nb) < 200:
        print(f"  [警告] 全量池中仅找到 {len(params_nb)} 个 use_negbin=1 的SKU，将按实际数量运行。")
    else:
        print(f"  实际SKU数: {len(params_nb)}")
    demand_nb = weekly_demand_full[params_nb['SKU'].tolist()]
    params_nb['mu_weekly'] = demand_nb.mean().values

    df2, plans2 = rolling_window_experiment(
        params_nb, demand_nb, scenario_name='high_volatility', v_cap=800, w_cap=400
    )
    all_results.append(df2)
    all_plans.append(plans2)

    print("\n" + "=" * 55)
    print("情景3: 大SKU规模 (Top 500 SKU, 容量收紧 V=600, W=300)")
    print("=" * 55)
    params_500 = params_full.head(500).reset_index(drop=True)
    demand_500 = weekly_demand_full[params_500['SKU'].tolist()]
    params_500['mu_weekly'] = demand_500.mean().values
    print(f"  实际SKU数: {len(params_500)}")

    df3, plans3 = rolling_window_experiment(
        params_500, demand_500, scenario_name='large_scale', v_cap=600, w_cap=300
    )
    all_results.append(df3)
    all_plans.append(plans3)

    df_all = pd.concat(all_results, ignore_index=True)
    plans_all = pd.concat(all_plans, ignore_index=True)
    return df_all, plans_all

# ══════════════════════════════════════════
# 8. 汇总与可视化
# ══════════════════════════════════════════
METHODS = [
    ('greedy_nominal', 'Greedy', 'steelblue', 'o'),
    ('greedy_robust', 'Greedy-Rob', 'deepskyblue', 's'),
    ('dp_nominal', 'DP-Lagrangian', 'coral', 'o'),
    ('dp_robust', 'DP-Rob', 'tomato', 's'),
    ('cpsat_nominal', 'CP-SAT', 'seagreen', 'o'),
    ('cpsat_robust', 'CP-SAT-Rob', 'mediumseagreen', 's'),
]
SCENARIOS = ['baseline', 'high_volatility', 'large_scale']
SCENARIO_LABELS = {
    'baseline': 'Baseline\n(N=200, std cap)',
    'high_volatility': 'High Volatility\n(Top 200 NegBin, std cap)',
    'large_scale': 'Large Scale\n(N=500, tight cap)',
}


def summarize(df_all):
    rows = []
    for sc in SCENARIOS:
        sub = df_all[df_all['scenario'] == sc]
        for key, label, *_ in METHODS:
            lfr_col = f'{key}_lfr'
            soc_col = f'{key}_stockout'
            time_col = f'{key}_time'
            if lfr_col not in sub.columns:
                continue
            rows.append({
                'scenario': sc,
                'method': label,
                'lfr_mean': sub[lfr_col].mean(),
                'lfr_ci': 1.96 * sub[lfr_col].std() / np.sqrt(len(sub)),
                'so_mean': sub[soc_col].mean(),
                'so_ci': 1.96 * sub[soc_col].std() / np.sqrt(len(sub)),
                'time_mean': sub[time_col].mean(),
                'n_windows': len(sub),
            })
    return pd.DataFrame(rows)


def plot_results(summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Algorithm Comparison Across Scenarios', fontsize=13, fontweight='bold')

    metrics = [
        ('lfr_mean', 'lfr_ci', 'Line Fill Rate (higher is better)'),
        ('so_mean', 'so_ci', 'Avg Stockout Prob (lower is better)'),
        ('time_mean', None, 'Runtime (seconds, log scale)'),
    ]

    for ax, (col, ci_col, title) in zip(axes, metrics):
        for sc_idx, sc in enumerate(SCENARIOS):
            sub = summary[summary['scenario'] == sc]
            x = np.arange(len(sub)) + sc_idx * (len(sub) + 1)
            colors = [m[2] for m in METHODS[:len(sub)]]
            yerr = sub[ci_col].values if ci_col else None
            ax.bar(x, sub[col].values, color=colors, alpha=0.8, yerr=yerr, capsize=3, edgecolor='white')
            mid = x.mean()
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                    SCENARIO_LABELS[sc], ha='center', va='top', fontsize=7, color='gray')
        if col == 'time_mean':
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=m[2], alpha=0.85) for m in METHODS]
    labels = [m[1] for m in METHODS]
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(RESULTS_DIR / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {RESULTS_DIR / 'algorithm_comparison.png'}")


def plot_rolling_lfr(df_all):
    sub = df_all[df_all['scenario'] == 'baseline'].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    for key, label, color, marker in METHODS:
        col = f'{key}_lfr'
        if col not in sub.columns:
            continue
        ax.plot(
            sub['window'], sub[col], label=label, color=color, linewidth=1.5,
            linestyle='-' if 'nominal' in key else '--', marker=marker, markersize=4, alpha=0.85
        )
    ax.set_title('Rolling Window Line Fill Rate — Baseline Scenario')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Line Fill Rate')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rolling_lfr_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {RESULTS_DIR / 'rolling_lfr_baseline.png'}")

# ══════════════════════════════════════════
# 9. 主函数
# ══════════════════════════════════════════
def main():
    print("=" * 55)
    print("前置仓SKU选品优化实验 — OR-Tools CP-SAT")
    print("=" * 55)

    print("\n[1] 加载数据...")
    params_full, weekly_demand_full = load_data(
        sku_path=DATA_DIR / 'sku_params.csv',
        demand_path=DATA_DIR / 'daily_demand.csv',
        n_sku=N_SKU,
    )
    print(f"    加载完成: {len(params_full)} SKU, {len(weekly_demand_full)} 周")
    if len(params_full) < 500:
        print(f"    [提示] 当前可用SKU仅 {len(params_full)} 个，后续 Top 500 情景将按实际数量运行。")

    print("\n[2] 开始滚动窗口实验...")
    t_start = time.time()
    df_all, plans_all = run_all_scenarios(params_full, weekly_demand_full)
    print(f"\n    全部实验完成，耗时 {time.time() - t_start:.1f} 秒")

    df_all.to_csv(RESULTS_DIR / 'rolling_results.csv', index=False)
    print(f"已保存: {RESULTS_DIR / 'rolling_results.csv'}")

    plans_all.to_csv(RESULTS_DIR / 'plans.csv', index=False)
    print(f"已保存: {RESULTS_DIR / 'plans.csv'}")

    print("\n[3] 汇总结果...")
    summary = summarize(df_all)
    summary.to_csv(RESULTS_DIR / 'summary.csv', index=False)
    print(f"已保存: {RESULTS_DIR / 'summary.csv'}")

    print("\n" + "=" * 80)
    print("算法对比汇总表（均值 ± 95%CI，基于滑动窗口）")
    print("=" * 80)
    for sc in SCENARIOS:
        print(f"\n▶ {SCENARIO_LABELS[sc].replace(chr(10), ' ')}")
        sub = summary[summary['scenario'] == sc][['method', 'lfr_mean', 'lfr_ci', 'so_mean', 'so_ci', 'time_mean']].copy()
        sub['LFR'] = sub.apply(lambda r: f"{r.lfr_mean:.3f} ± {r.lfr_ci:.3f}", axis=1)
        sub['Stockout'] = sub.apply(lambda r: f"{r.so_mean:.3f} ± {r.so_ci:.3f}", axis=1)
        sub['Time(s)'] = sub['time_mean'].apply(lambda x: f"{x:.3f}")
        print(sub[['method', 'LFR', 'Stockout', 'Time(s)']].to_string(index=False))

    print("\n[4] 生成图表...")
    plot_results(summary)
    plot_rolling_lfr(df_all)

    print("\n" + "=" * 55)
    print(f"全部完成！输出文件在 {RESULTS_DIR}/ 目录下。")
    print("=" * 55)


if __name__ == '__main__':
    main()
