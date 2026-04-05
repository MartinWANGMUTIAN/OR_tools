"""
前置仓选品文献基线复现实验
==========================

目的：
  - 使用与你论文主代码相同的输入数据集：sku_params.csv + daily_demand.csv
  - 在相同的滚动窗口、容量约束和评估口径下，运行一个独立的文献基线算法
  - 输出可直接与论文主代码结果对比的 CSV

说明：
  当前 `对比文献/defiguard_repro.py` 属于 DeFi 交易图分类任务，数据结构与前置仓 SKU
  选品问题完全不同，不能直接复用到本数据集。本脚本提供一个同问题设定下的对比基线。

基线算法：
  Marginal Density Greedy
  - 对每个 SKU 按补 1 件带来的边际收益 / 边际资源占用进行排序
  - 在体积、重量双约束下逐步补货，直到没有正收益或容量耗尽
  - 同时提供 nominal / robust 两个版本，方便与主代码对比
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

matplotlib.rcParams["font.family"] = "DejaVu Sans"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "dark store scanner data" / "or tools模型" / "optimizer"
PAPER_RESULTS_DIR = ROOT_DIR / "dark store scanner data" / "or tools模型" / "results"
RESULTS_DIR = Path(__file__).resolve().parent / "results_dark_store"
os.makedirs(RESULTS_DIR, exist_ok=True)

Q_MAX = 15
N_SKU = 500
WINDOW_TRAIN = 8
WINDOW_TEST = 1
ROBUST_Z = 1.645

SCENARIOS = [
    ("baseline", 200, 800.0, 400.0, False),
    ("high_volatility", 200, 800.0, 400.0, True),
    ("large_scale", 500, 600.0, 300.0, False),
]


def load_data(
    sku_path: Path | None = None,
    demand_path: Path | None = None,
    n_sku: int = N_SKU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sku_path is None:
        sku_path = DATA_DIR / "sku_params.csv"
    if demand_path is None:
        demand_path = DATA_DIR / "daily_demand.csv"

    params = pd.read_csv(sku_path)
    params = params.nlargest(n_sku, "total_qty").reset_index(drop=True)

    demand = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    demand = demand[[c for c in params["SKU"] if c in demand.columns]]

    params = params[params["SKU"].isin(demand.columns)].reset_index(drop=True)
    demand = demand[params["SKU"].tolist()]
    weekly = demand.resample("W").sum()
    return params, weekly


def compute_e_min(mu: float, r: float, use_negbin: bool, q_max: int) -> np.ndarray:
    e_min = np.zeros(q_max + 1)
    for q in range(1, q_max + 1):
        s = 0.0
        for k in range(q):
            if use_negbin and r < 9000:
                p = r / (r + mu) if (r + mu) > 0 else 1.0
                cdf_k = nbinom.cdf(k, r, p)
            else:
                cdf_k = poisson.cdf(k, mu)
            s += 1.0 - cdf_k
        e_min[q] = s
    return e_min


def compute_g(params: pd.DataFrame, mu_override=None, q_max: int = Q_MAX) -> np.ndarray:
    n = len(params)
    g = np.zeros((n, q_max + 1))

    for i, row in params.iterrows():
        mu = mu_override[i] if mu_override is not None else row["mu_weekly"]
        r = row["r_negbin"]
        nb = bool(row["use_negbin"])
        u, h, b, f = row["u_i"], row["h_i"], row["b_i"], row["f_i"]

        e_min = compute_e_min(mu, r, nb, q_max)
        for q in range(q_max + 1):
            if q == 0:
                g[i, q] = 0.0
            else:
                g[i, q] = (u + h + b) * e_min[q] - h * q - b * mu - f

    return g


def compute_resource(params: pd.DataFrame, q_max: int = Q_MAX) -> tuple[np.ndarray, np.ndarray]:
    n = len(params)
    s_v = np.ones(n) * 2.0
    s_w = np.ones(n) * 0.5

    a_v = np.zeros((n, q_max + 1))
    a_w = np.zeros((n, q_max + 1))
    for i, row in params.iterrows():
        for q in range(1, q_max + 1):
            a_v[i, q] = s_v[i] + row["v_i"] * q
            a_w[i, q] = s_w[i] + row["w_i"] * q
    return a_v, a_w


def evaluate_plan(params: pd.DataFrame, q_vec: np.ndarray, mu_true=None) -> dict[str, float]:
    n = len(params)
    total_satisfied = 0.0
    total_demand = 0.0
    total_stockout = 0.0
    n_listed = 0

    for i, row in params.iterrows():
        q = int(q_vec[i])
        mu = mu_true[i] if mu_true is not None else row["mu_weekly"]
        r = row["r_negbin"]
        nb = bool(row["use_negbin"])

        if q == 0:
            total_demand += mu
            total_stockout += 1.0
            continue

        n_listed += 1
        e_min = compute_e_min(mu, r, nb, q)[q]
        if nb and r < 9000:
            p = r / (r + mu) if (r + mu) > 0 else 1.0
            p_stockout = 1 - nbinom.cdf(q, r, p)
        else:
            p_stockout = 1 - poisson.cdf(q, mu)

        total_satisfied += e_min
        total_demand += mu
        total_stockout += p_stockout

    return {
        "line_fill_rate": total_satisfied / total_demand if total_demand > 0 else 0.0,
        "avg_stockout": total_stockout / n,
        "n_listed": n_listed,
    }


def marginal_density_greedy(
    g: np.ndarray,
    a_v: np.ndarray,
    a_w: np.ndarray,
    v_cap: float,
    w_cap: float,
    q_max: int = Q_MAX,
) -> np.ndarray:
    """
    文献型基线：允许 SKU 从当前库存位直接跳到任意更优库存位，
    每步选择“增量收益 / 增量资源占用”最高的可行动作。
    这样既保留贪心近似的可解释性，也能处理固定上架成本导致的初始跳跃问题。
    """
    n = g.shape[0]
    q_vec = np.zeros(n, dtype=int)
    v_used = 0.0
    w_used = 0.0

    while True:
        best_i = None
        best_q = None
        best_score = -np.inf

        for i in range(n):
            q_now = q_vec[i]
            if q_now >= q_max:
                continue

            for q_next in range(q_now + 1, q_max + 1):
                delta_v = a_v[i, q_next] - a_v[i, q_now]
                delta_w = a_w[i, q_next] - a_w[i, q_now]
                if v_used + delta_v > v_cap or w_used + delta_w > w_cap:
                    continue

                delta_g = g[i, q_next] - g[i, q_now]
                if delta_g <= 0:
                    continue

                density = delta_g / (delta_v / v_cap + delta_w / w_cap + 1e-9)
                if density > best_score:
                    best_score = density
                    best_i = i
                    best_q = q_next

        if best_i is None or best_q is None:
            break

        v_used += a_v[best_i, best_q] - a_v[best_i, q_vec[best_i]]
        w_used += a_w[best_i, best_q] - a_w[best_i, q_vec[best_i]]
        q_vec[best_i] = best_q

    return q_vec


def extract_plan_records(
    params: pd.DataFrame,
    q_vec: np.ndarray,
    window_id: int,
    scenario_name: str,
    method_key: str,
) -> list[dict[str, object]]:
    records = []
    for i, sku in enumerate(params["SKU"]):
        if int(q_vec[i]) > 0:
            records.append(
                {
                    "window": window_id,
                    "scenario": scenario_name,
                    "algorithm": method_key,
                    "sku": sku,
                    "stock_qty": int(q_vec[i]),
                }
            )
    return records


def rolling_window_baseline(
    params: pd.DataFrame,
    weekly_demand: pd.DataFrame,
    scenario_name: str,
    v_cap: float,
    w_cap: float,
    train_weeks: int = WINDOW_TRAIN,
    test_weeks: int = WINDOW_TEST,
    q_max: int = Q_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_weeks = len(weekly_demand)
    windows = list(range(0, n_weeks - train_weeks - test_weeks + 1))
    rows = []
    plan_rows = []

    print(f"\n[Rolling] {scenario_name}: 共 {len(windows)} 个窗口")

    for w_idx, w_start in enumerate(windows):
        train = weekly_demand.iloc[w_start : w_start + train_weeks]
        test = weekly_demand.iloc[w_start + train_weeks : w_start + train_weeks + test_weeks]

        mu_est = train.mean().values
        se_est = train.std().fillna(0.0).values / np.sqrt(max(len(train), 1))
        mu_rob = mu_est + ROBUST_Z * se_est
        mu_true = test.mean().values

        params_now = params.copy()
        params_now["mu_weekly"] = mu_est
        a_v, a_w = compute_resource(params_now, q_max=q_max)

        row = {"window": w_idx, "scenario": scenario_name}
        for suffix, mu_use in [("nominal", mu_est), ("robust", mu_rob)]:
            g = compute_g(params_now, mu_override=mu_use, q_max=q_max)

            t0 = time.time()
            q_vec = marginal_density_greedy(g, a_v, a_w, v_cap=v_cap, w_cap=w_cap, q_max=q_max)
            elapsed = time.time() - t0

            metrics = evaluate_plan(params_now, q_vec, mu_true=mu_true)
            key = f"literature_{suffix}"
            row[f"{key}_lfr"] = metrics["line_fill_rate"]
            row[f"{key}_stockout"] = metrics["avg_stockout"]
            row[f"{key}_time"] = elapsed
            row[f"{key}_n_listed"] = metrics["n_listed"]

            plan_rows.extend(
                extract_plan_records(
                    params=params_now,
                    q_vec=q_vec,
                    window_id=w_idx,
                    scenario_name=scenario_name,
                    method_key=key,
                )
            )

        rows.append(row)
        if (w_idx + 1) % 5 == 0 or (w_idx + 1) == len(windows):
            print(f"  完成窗口 {w_idx + 1}/{len(windows)}")

    return pd.DataFrame(rows), pd.DataFrame(plan_rows)


def run_all_scenarios(params_full: pd.DataFrame, weekly_full: pd.DataFrame):
    result_frames = []
    plan_frames = []
    params_full = params_full.sort_values("total_qty", ascending=False).reset_index(drop=True)

    for name, top_n, v_cap, w_cap, negbin_only in SCENARIOS:
        params_now = params_full.copy()
        if negbin_only:
            params_now = params_now[params_now["use_negbin"] == 1].copy()

        params_now = params_now.head(top_n).reset_index(drop=True)
        if params_now.empty:
            print(f"[跳过] {name}: 没有可用 SKU")
            continue

        demand_now = weekly_full[params_now["SKU"].tolist()]
        df_result, df_plan = rolling_window_baseline(
            params=params_now,
            weekly_demand=demand_now,
            scenario_name=name,
            v_cap=v_cap,
            w_cap=w_cap,
        )
        result_frames.append(df_result)
        plan_frames.append(df_plan)

    return pd.concat(result_frames, ignore_index=True), pd.concat(plan_frames, ignore_index=True)


def summarize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        for key in ["literature_nominal", "literature_robust"]:
            rows.append(
                {
                    "scenario": scenario,
                    "method": key,
                    "lfr_mean": sub[f"{key}_lfr"].mean(),
                    "lfr_std": sub[f"{key}_lfr"].std(),
                    "stockout_mean": sub[f"{key}_stockout"].mean(),
                    "stockout_std": sub[f"{key}_stockout"].std(),
                    "time_mean": sub[f"{key}_time"].mean(),
                    "n_listed_mean": sub[f"{key}_n_listed"].mean(),
                    "n_windows": len(sub),
                }
            )
    return pd.DataFrame(rows)


def compare_with_paper_results(
    baseline_summary: pd.DataFrame,
    paper_results_path: Path = PAPER_RESULTS_DIR / "summary.csv",
) -> pd.DataFrame | None:
    if not paper_results_path.exists():
        return None

    paper_summary = pd.read_csv(paper_results_path)
    paper_summary = paper_summary.rename(
        columns={
            "so_mean": "stockout_mean",
            "time_mean": "time_mean",
        }
    )
    paper_summary = paper_summary[
        ["scenario", "method", "lfr_mean", "stockout_mean", "time_mean"]
    ].copy()

    baseline_comp = baseline_summary[
        ["scenario", "method", "lfr_mean", "stockout_mean", "time_mean"]
    ].copy()

    merged = pd.concat([paper_summary, baseline_comp], ignore_index=True)
    return merged.sort_values(["scenario", "lfr_mean"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    print("=" * 60)
    print("对比文献基线实验: 前置仓同数据集对比")
    print("=" * 60)

    params_full, weekly_full = load_data(
        sku_path=DATA_DIR / "sku_params.csv",
        demand_path=DATA_DIR / "daily_demand.csv",
        n_sku=N_SKU,
    )
    print(f"[数据] SKU={len(params_full)}, 周数={len(weekly_full)}")

    t0 = time.time()
    results_df, plans_df = run_all_scenarios(params_full, weekly_full)
    print(f"[完成] 总耗时 {time.time() - t0:.2f} 秒")

    summary_df = summarize_baseline(results_df)
    comparison_df = compare_with_paper_results(summary_df)

    results_path = RESULTS_DIR / "literature_rolling_results.csv"
    plans_path = RESULTS_DIR / "literature_plans.csv"
    summary_path = RESULTS_DIR / "literature_summary.csv"
    results_df.to_csv(results_path, index=False)
    plans_df.to_csv(plans_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[输出] {results_path}")
    print(f"[输出] {plans_path}")
    print(f"[输出] {summary_path}")

    if comparison_df is not None:
        comparison_path = RESULTS_DIR / "paper_vs_literature_summary.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"[输出] {comparison_path}")
    else:
        print("[提示] 未找到论文主代码 summary.csv，暂未生成合并对比表")


if __name__ == "__main__":
    main()
