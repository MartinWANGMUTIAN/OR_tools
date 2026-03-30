"""
前置仓SKU选品与库存配置联合优化
========================================
依赖: pip install ortools pandas numpy scipy matplotlib

输入文件（由 dark_store_data_prep.py 生成）:
  - /Users/sandmwong/Desktop/OR_tools/dark store scanner data/sku_params.csv      : SKU参数（需求分布、成本、体积/重量）
  - /Users/sandmwong/Desktop/OR_tools/dark store scanner data/sku_params.csv    : 每日需求矩阵（用于SAA与Rolling Window实验）

三种算法:
  1. Greedy        : 贪心启发式（极快，可解释）
  2. DP_Lagrangian : 拉格朗日松弛动态规划（中等规模）
  3. MILP_CpSAT    : OR-Tools CP-SAT 整数规划（最优基准）

实验设计:
  - 3组情景 × Rolling Window（每周滑动，用前N周历史估计，预测下1周）
  - 每组情景下对比4种方法（含鲁棒均值版本）
  - 输出：满足率、缺货率、目标值、运行时间对比表 + 图表
"""

import pandas as pd
import numpy as np
from scipy.stats import nbinom, poisson
import time
import warnings
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

# OR-Tools
from ortools.sat.python import cp_model

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 0. 全局参数
# ══════════════════════════════════════════
Q_MAX   = 15       # 最大备货量（件）
V_CAP   = 800.0    # 前置仓体积容量（升）—— 可调参数
W_CAP   = 400.0    # 前置仓重量容量（kg）—— 可调参数
SCALE   = 100      # CP-SAT 需要整数系数，将浮点×SCALE后取整
N_SKU   = 500      # 主数据池规模：先读取Top 500 SKU，再按情景切片
WINDOW_TRAIN = 8   # 训练窗口（周）
WINDOW_TEST  = 1   # 预测窗口（周）
ROBUST_Z     = 1.645  # 鲁棒均值置信水平（90% 单侧）

# ══════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════
def load_data(sku_path=None, demand_path=None, n_sku=N_SKU):
    if sku_path is None:
        sku_path = BASE_DIR / 'sku_params.csv'
    if demand_path is None:
        demand_path = BASE_DIR / 'daily_demand.csv'

    params = pd.read_csv(sku_path)
    # 取 Top n_sku 高销量SKU
    params = params.nlargest(n_sku, 'total_qty').reset_index(drop=True)

    demand = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    demand = demand[[c for c in params['SKU'] if c in demand.columns]]

    # 对齐列顺序
    params = params[params['SKU'].isin(demand.columns)].reset_index(drop=True)
    demand = demand[params['SKU'].tolist()]

    # 按周汇总
    weekly = demand.resample('W').sum()
    return params, weekly

# ══════════════════════════════════════════
# 2. 期望价值系数预计算
# ══════════════════════════════════════════
def compute_E_min(mu, r, use_negbin, q_max):
    """
    计算 E[min(D, q)] for q = 0..q_max
    利用 E[min(D,q)] = sum_{k=1}^{q} P(D >= k) = sum_{k=0}^{q-1} (1 - F(k))
    """
    e_min = np.zeros(q_max + 1)
    for q in range(1, q_max + 1):
        s = 0.0
        for k in range(q):
            if use_negbin and r < 9000:
                # 负二项：r=离散度参数，p=r/(r+mu)
                p = r / (r + mu) if (r + mu) > 0 else 1.0
                cdf_k = nbinom.cdf(k, r, p)
            else:
                cdf_k = poisson.cdf(k, mu)
            s += (1.0 - cdf_k)
        e_min[q] = s
    return e_min


def compute_G(params, mu_override=None, q_max=Q_MAX):
    """
    计算每个SKU每个备货量的期望净贡献 G[i, q]
    G[i,q] = (u+h+b)*E[min(D,q)] - h*q - b*mu - f*(q>0)

    mu_override: 可传入鲁棒均值（数组，长度=N）
    返回: G shape (N, q_max+1)
    """
    N = len(params)
    G = np.zeros((N, q_max + 1))

    for i, row in params.iterrows():
        mu  = mu_override[i] if mu_override is not None else row['mu_weekly']
        r   = row['r_negbin']
        nb  = bool(row['use_negbin'])
        u, h, b, f = row['u_i'], row['h_i'], row['b_i'], row['f_i']

        e_min = compute_E_min(mu, r, nb, q_max)
        for q in range(q_max + 1):
            if q == 0:
                G[i, q] = 0.0
            else:
                G[i, q] = (u + h + b) * e_min[q] - h * q - b * mu - f

    return G


def compute_resource(params, q_max=Q_MAX):
    """
    计算每个SKU每个备货量的资源消耗
    A_V[i,q], A_W[i,q]: q=0时为0，q>0时为固定槽位+单件消耗
    """
    N = len(params)
    s_v = np.ones(N) * 2.0   # 固定槽位体积（升）—— 可按品类替换
    s_w = np.ones(N) * 0.5   # 固定槽位重量（kg）

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
    """
    给定备货量向量 q_vec（长度N，0=不上架），
    用真实需求均值（mu_true）计算各指标的解析期望值。
    """
    N = len(params)
    total_satisfied = 0.0
    total_demand    = 0.0
    total_stockout  = 0.0
    n_listed        = 0

    for i, row in params.iterrows():
        q = int(q_vec[i])
        mu = mu_true[i] if mu_true is not None else row['mu_weekly']
        r  = row['r_negbin']
        nb = bool(row['use_negbin'])

        if q == 0:
            total_demand += mu
            total_stockout += 1.0  # 未上架视为必缺货
            continue

        n_listed += 1
        e_min = compute_E_min(mu, r, nb, q)[q]
        if nb and r < 9000:
            p = r / (r + mu) if (r + mu) > 0 else 1.0
            p_stockout = 1 - nbinom.cdf(q, r, p)
        else:
            p_stockout = 1 - poisson.cdf(q, mu)

        total_satisfied += e_min
        total_demand    += mu
        total_stockout  += p_stockout

    line_fill_rate = total_satisfied / total_demand if total_demand > 0 else 0.0
    avg_stockout   = total_stockout / N
    return {
        'line_fill_rate': line_fill_rate,
        'avg_stockout':   avg_stockout,
        'n_listed':       n_listed,
    }

# ══════════════════════════════════════════
# 4. 算法实现
# ══════════════════════════════════════════

# ── 4a. 贪心 ──────────────────────────────
def greedy(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX):
    N = len(params)
    # 每个SKU的单品最优备货量（不考虑容量）
    q_star = np.argmax(G, axis=1)  # shape (N,)

    # 价值密度（归一化容量加权）
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
        # 尝试从 q_star[i] 逐步降低直到可行
        for q in range(q_star[i], 0, -1):
            if G[i, q] <= 0:
                break
            if v_used + A_V[i, q] <= v_cap and w_used + A_W[i, q] <= w_cap:
                q_vec[i] = q
                v_used += A_V[i, q]
                w_used += A_W[i, q]
                break
    return q_vec


# ── 4b. 拉格朗日松弛 DP ───────────────────
def dp_lagrangian(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX,
                  n_lambda=20):
    """
    对重量约束做拉格朗日松弛，对体积做多选背包DP。
    二分搜索 lambda 使重量约束尽量满足。
    """
    N = len(params)
    V_int = int(v_cap)

    def solve_dp(lam):
        # 惩罚后的价值: G[i,q] - lam * A_W[i,q]
        G_pen = G - lam * A_W   # shape (N, q_max+1)
        # 多选背包DP（体积维度）
        # dp[v] = 最优值（处理前 i 个SKU，体积 <= v）
        INF = -1e18
        dp   = np.full(V_int + 1, INF)
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

        # 回溯最优解
        best_v = int(np.argmax(dp))
        q_vec  = np.zeros(N, dtype=int)
        v_rem  = best_v
        for i in range(N - 1, -1, -1):
            q_vec[i] = choice[i, v_rem]
            v_rem   -= int(A_V[i, q_vec[i]])

        w_used = sum(A_W[i, q_vec[i]] for i in range(N))
        return q_vec, w_used

    # 二分搜索 lambda 使 w_used <= w_cap
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
        # 若全程不可行，退化为 lam=lam_hi 的解（可能轻微违反重量约束，截断处理）
        best_q, _ = solve_dp(lam_hi)
        for i in range(N):
            if best_q[i] > 0:
                while best_q[i] > 0 and (
                    sum(A_W[j, best_q[j]] for j in range(N)) > w_cap
                ):
                    best_q[i] -= 1

    return best_q


# ── 4c. OR-Tools CP-SAT ───────────────────
def milp_cpsat(params, G, A_V, A_W, v_cap=V_CAP, w_cap=W_CAP, q_max=Q_MAX,
               time_limit=60, scale=SCALE):
    """
    用 OR-Tools CP-SAT 求解多选多维背包。
    CP-SAT 只支持整数，所有浮点系数乘以 scale 后取整。
    """
    N = len(params)
    model = cp_model.CpModel()

    # 变量: x[i][q] ∈ {0,1}
    x = [[model.NewBoolVar(f'x_{i}_{q}') for q in range(q_max + 1)]
         for i in range(N)]

    # 约束1: 每个SKU恰好选一个备货量
    for i in range(N):
        model.AddExactlyOne(x[i])

    # 约束2: 体积容量
    model.Add(
        sum(int(A_V[i, q] * scale) * x[i][q]
            for i in range(N) for q in range(q_max + 1))
        <= int(v_cap * scale)
    )

    # 约束3: 重量容量
    model.Add(
        sum(int(A_W[i, q] * scale) * x[i][q]
            for i in range(N) for q in range(q_max + 1))
        <= int(w_cap * scale)
    )

    # 目标: 最大化期望净收益
    # G 可能为负，需整体平移保证系数可行（CP-SAT目标支持负系数）
    obj_terms = []
    for i in range(N):
        for q in range(q_max + 1):
            coef = int(G[i, q] * scale)
            obj_terms.append(coef * x[i][q])
    model.Maximize(sum(obj_terms))

    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers  = 4    # 并行线程数
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
# 5. Rolling Window 实验框架
# ══════════════════════════════════════════
def rolling_window_experiment(params, weekly_demand,
                               train_weeks=WINDOW_TRAIN,
                               test_weeks=WINDOW_TEST,
                               scenario_name='baseline',
                               q_max=Q_MAX, v_cap=V_CAP, w_cap=W_CAP):
    """
    滑动窗口实验：用前 train_weeks 周历史估计需求参数，
    在下 test_weeks 周评估各算法表现。
    """
    n_weeks = len(weekly_demand)
    results  = []
    windows  = list(range(0, n_weeks - train_weeks - test_weeks + 1))
    print(f"\n  [Rolling] 共 {len(windows)} 个窗口")

    for w_idx, w_start in enumerate(windows):
        train = weekly_demand.iloc[w_start : w_start + train_weeks]
        test  = weekly_demand.iloc[w_start + train_weeks :
                                   w_start + train_weeks + test_weeks]

        # ── 用训练窗口估计需求参数 ──────────────
        mu_est = train.mean().values          # 估计均值
        se_est = train.std().values / np.sqrt(len(train))  # 标准误
        mu_rob = mu_est + ROBUST_Z * se_est   # 鲁棒均值

        # 真实均值（用测试窗口的实际均值代理）
        mu_true = test.mean().values

        # 更新 params 中的周需求均值
        params = params.copy()
        params['mu_weekly'] = mu_est

        # ── 预计算系数矩阵 ───────────────────────
        G_nom = compute_G(params, mu_override=mu_est, q_max=q_max)
        G_rob = compute_G(params, mu_override=mu_rob, q_max=q_max)
        A_V, A_W = compute_resource(params, q_max=q_max)

        row = {'window': w_idx, 'scenario': scenario_name}

        for method_name, G_use in [('nominal', G_nom), ('robust', G_rob)]:
            for alg_name in ['greedy', 'dp', 'cpsat']:
                t0 = time.time()

                if alg_name == 'greedy':
                    q_vec = greedy(params, G_use, A_V, A_W, v_cap, w_cap, q_max)
                    status = 'greedy'
                elif alg_name == 'dp':
                    q_vec = dp_lagrangian(params, G_use, A_V, A_W, v_cap, w_cap, q_max)
                    status = 'dp'
                else:  # cpsat
                    q_vec, status = milp_cpsat(params, G_use, A_V, A_W,
                                               v_cap, w_cap, q_max)

                elapsed = time.time() - t0
                metrics = evaluate(params, q_vec, mu_true=mu_true)

                key = f'{alg_name}_{method_name}'
                row[f'{key}_lfr']      = metrics['line_fill_rate']
                row[f'{key}_stockout'] = metrics['avg_stockout']
                row[f'{key}_time']     = elapsed
                row[f'{key}_n_listed'] = metrics['n_listed']

        results.append(row)
        if (w_idx + 1) % 5 == 0:
            print(f"    完成窗口 {w_idx+1}/{len(windows)}")

    return pd.DataFrame(results)

# ══════════════════════════════════════════
# 6. 三组实验情景
# ══════════════════════════════════════════
def run_all_scenarios(params_full, weekly_demand_full):
    all_results = []

    # 先确保全量池按销量降序，后续 head / nlargest 才有明确含义
    params_full = params_full.sort_values('total_qty', ascending=False).reset_index(drop=True)

    # ── 情景1: Baseline（Top 200 SKU, 标准容量）
    print("\n" + "="*55)
    print("情景1: Baseline (Top 200 SKU, V=800, W=400)")
    print("="*55)
    params_200 = params_full.head(200).reset_index(drop=True)
    demand_200 = weekly_demand_full[params_200['SKU'].tolist()]
    params_200['mu_weekly'] = demand_200.mean().values
    print(f"  实际SKU数: {len(params_200)}")

    df1 = rolling_window_experiment(
        params_200, demand_200,
        scenario_name='baseline', v_cap=800, w_cap=400
    )
    all_results.append(df1)

    # ── 情景2: 高波动（从全量500池中选 Top 200 NegBin SKU）
    print("\n" + "="*55)
    print("情景2: 高需求波动 (Top 200 NegBin SKU only)")
    print("="*55)
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

    df2 = rolling_window_experiment(
        params_nb, demand_nb,
        scenario_name='high_volatility', v_cap=800, w_cap=400
    )
    all_results.append(df2)

    # ── 情景3: 大SKU规模（Top 500，容量收紧）
    print("\n" + "="*55)
    print("情景3: 大SKU规模 (Top 500 SKU, 容量收紧 V=600, W=300)")
    print("="*55)
    params_500 = params_full.head(500).reset_index(drop=True)
    demand_500 = weekly_demand_full[params_500['SKU'].tolist()]
    params_500['mu_weekly'] = demand_500.mean().values
    print(f"  实际SKU数: {len(params_500)}")

    df3 = rolling_window_experiment(
        params_500, demand_500,
        scenario_name='large_scale', v_cap=600, w_cap=300
    )
    all_results.append(df3)

    return pd.concat(all_results, ignore_index=True)

# ══════════════════════════════════════════
# 7. 汇总与可视化
# ══════════════════════════════════════════
METHODS = [
    ('greedy_nominal',  'Greedy',         'steelblue',   'o'),
    ('greedy_robust',   'Greedy-Rob',     'deepskyblue', 's'),
    ('dp_nominal',      'DP-Lagrangian',  'coral',       'o'),
    ('dp_robust',       'DP-Rob',         'tomato',      's'),
    ('cpsat_nominal',   'CP-SAT',         'seagreen',    'o'),
    ('cpsat_robust',    'CP-SAT-Rob',     'mediumseagreen','s'),
]
SCENARIOS = ['baseline', 'high_volatility', 'large_scale']
SCENARIO_LABELS = {
    'baseline':       'Baseline\n(N=200, std cap)',
    'high_volatility':'High Volatility\n(Top 200 NegBin, std cap)',
    'large_scale':    'Large Scale\n(N=500, tight cap)',
}


def summarize(df_all):
    rows = []
    for sc in SCENARIOS:
        sub = df_all[df_all['scenario'] == sc]
        for key, label, *_ in METHODS:
            lfr_col  = f'{key}_lfr'
            soc_col  = f'{key}_stockout'
            time_col = f'{key}_time'
            if lfr_col not in sub.columns:
                continue
            rows.append({
                'scenario': sc,
                'method':   label,
                'lfr_mean': sub[lfr_col].mean(),
                'lfr_ci':   1.96 * sub[lfr_col].std() / np.sqrt(len(sub)),
                'so_mean':  sub[soc_col].mean(),
                'so_ci':    1.96 * sub[soc_col].std() / np.sqrt(len(sub)),
                'time_mean':sub[time_col].mean(),
                'n_windows': len(sub),
            })
    return pd.DataFrame(rows)


def plot_results(summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Algorithm Comparison Across Scenarios', fontsize=13, fontweight='bold')

    metrics = [
        ('lfr_mean',  'lfr_ci',  'Line Fill Rate (higher is better)'),
        ('so_mean',   'so_ci',   'Avg Stockout Prob (lower is better)'),
        ('time_mean', None,      'Runtime (seconds, log scale)'),
    ]

    for ax, (col, ci_col, title) in zip(axes, metrics):
        for sc_idx, sc in enumerate(SCENARIOS):
            sub = summary[summary['scenario'] == sc]
            x   = np.arange(len(sub)) + sc_idx * (len(sub) + 1)
            colors = [next(c for k, _, c, _ in METHODS if _ == sub.iloc[j]['method']
                           or METHODS[j % len(METHODS)][1] == sub.iloc[j]['method'])
                      for j in range(len(sub))]
            colors = [m[2] for m in METHODS[:len(sub)]]
            yerr   = sub[ci_col].values if ci_col else None
            ax.bar(x, sub[col].values, color=colors, alpha=0.8,
                   yerr=yerr, capsize=3, edgecolor='white')
            # 情景标签
            mid = x.mean()
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                    SCENARIO_LABELS[sc], ha='center', va='top',
                    fontsize=7, color='gray')
        if col == 'time_mean':
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)

    # 统一图例
    handles = [plt.Rectangle((0,0),1,1, color=m[2], alpha=0.85)
               for m in METHODS]
    labels  = [m[1] for m in METHODS]
    fig.legend(handles, labels, loc='lower center', ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(RESULTS_DIR / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {RESULTS_DIR / 'algorithm_comparison.png'}")


def plot_rolling_lfr(df_all):
    """各算法在 Baseline 情景下的滚动满足率时序图"""
    sub = df_all[df_all['scenario'] == 'baseline'].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    for key, label, color, marker in METHODS:
        col = f'{key}_lfr'
        if col not in sub.columns:
            continue
        ax.plot(sub['window'], sub[col], label=label,
                color=color, linewidth=1.5,
                linestyle='-' if 'nominal' in key else '--',
                marker=marker, markersize=4, alpha=0.85)
    ax.set_title('Rolling Window Line Fill Rate — Baseline Scenario')
    ax.set_xlabel('Window Index'); ax.set_ylabel('Line Fill Rate')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rolling_lfr_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {RESULTS_DIR / 'rolling_lfr_baseline.png'}")

# ══════════════════════════════════════════
# 8. 主函数
# ══════════════════════════════════════════
def main():
    print("=" * 55)
    print("前置仓SKU选品优化实验 — OR-Tools CP-SAT")
    print("=" * 55)

    # 加载数据
    print("\n[1] 加载数据...")
    params_full, weekly_demand_full = load_data(
        sku_path=BASE_DIR / 'sku_params.csv',
        demand_path=BASE_DIR / 'daily_demand.csv',
        n_sku=N_SKU
    )
    print(f"    加载完成: {len(params_full)} SKU, {len(weekly_demand_full)} 周")
    if len(params_full) < 500:
        print(f"    [提示] 当前可用SKU仅 {len(params_full)} 个，后续 Top 500 情景将按实际数量运行。")

    # 运行实验
    print("\n[2] 开始滚动窗口实验...")
    t_start = time.time()
    df_all = run_all_scenarios(params_full, weekly_demand_full)
    print(f"\n    全部实验完成，耗时 {time.time()-t_start:.1f} 秒")

    # 保存原始结果
    df_all.to_csv(RESULTS_DIR / 'rolling_results.csv', index=False)
    print(f"已保存: {RESULTS_DIR / 'rolling_results.csv'}")

    # 汇总
    print("\n[3] 汇总结果...")
    summary = summarize(df_all)
    summary.to_csv(RESULTS_DIR / 'summary.csv', index=False)
    print(f"已保存: {RESULTS_DIR / 'summary.csv'}")

    # 打印汇总表
    print("\n" + "=" * 80)
    print("算法对比汇总表（均值 ± 95%CI，基于滑动窗口）")
    print("=" * 80)
    for sc in SCENARIOS:
        print(f"\n▶ {SCENARIO_LABELS[sc].replace(chr(10), ' ')}")
        sub = summary[summary['scenario'] == sc][
            ['method', 'lfr_mean', 'lfr_ci', 'so_mean', 'so_ci', 'time_mean']
        ].copy()
        sub['LFR']     = sub.apply(lambda r: f"{r.lfr_mean:.3f} ± {r.lfr_ci:.3f}", axis=1)
        sub['Stockout']= sub.apply(lambda r: f"{r.so_mean:.3f} ± {r.so_ci:.3f}", axis=1)
        sub['Time(s)'] = sub['time_mean'].apply(lambda x: f"{x:.3f}")
        print(sub[['method', 'LFR', 'Stockout', 'Time(s)']].to_string(index=False))

    # 可视化
    print("\n[4] 生成图表...")
    plot_results(summary)
    plot_rolling_lfr(df_all)

    print("\n" + "=" * 55)
    print(f"全部完成！输出文件在 {RESULTS_DIR}/ 目录下。")
    print("=" * 55)


if __name__ == '__main__':
    main()
