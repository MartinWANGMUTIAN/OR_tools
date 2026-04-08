
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Li (2024) + Transchel et al. (2022) hybrid baseline v2
------------------------------------------------------
Upgraded version with a simplified EXOGENOUS substitution matrix.

Compared with v1:
1) keeps Li-style assortment proxy stage
2) replaces independent inventory sizing with a substitution-aware local search
3) adds a quality/price-distance-based substitution matrix over selected SKUs
4) evaluates substitution-aware expected sales/profit during sizing

Still NOT an exact reproduction of either paper:
- Li et al. needs basket-level orders; we only have daily SKU totals
- Transchel et al. needs endogenous utility-based substitution matrices for all
  availability states; here we use an exogenous matrix estimated from similarity

Use:
python li_transchel_hybrid_baseline_v2.py \
  --sku_params ".../dark store scanner data/or tools模型/optimizer/sku_params.csv" \
  --daily_demand ".../dark store scanner data/or tools模型/optimizer/daily_demand.csv" \
  --output_dir ".../对比文献2/results_li_transchel_hybrid_v2"
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "dark store scanner data" / "or tools模型" / "optimizer"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "对比文献2" / "results_li_transchel_hybrid_v2"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clip_prob(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class DistSpec:
    kind: str
    mu: float
    r: float | None = None


def make_dist(spec: DistSpec):
    if spec.kind == "nbinom" and spec.r is not None and spec.r > 0:
        p = spec.r / (spec.r + spec.mu) if spec.mu > 0 else 1.0
        p = clip_prob(p)
        return nbinom(spec.r, p)
    return poisson(spec.mu)


def expected_min(spec: DistSpec, q: int) -> float:
    q = max(0, int(q))
    if q <= 0:
        return 0.0
    dist = make_dist(spec)
    ks = np.arange(q)
    return float(np.sum(dist.sf(ks)))  # sum P(D>k), k=0..q-1


def demand_ppf(spec: DistSpec, alpha: float) -> int:
    alpha = clip_prob(alpha)
    dist = make_dist(spec)
    val = dist.ppf(alpha)
    if np.isnan(val):
        return 0
    return max(0, int(val))


def critical_ratio(u_i: float, h_i: float, b_i: float) -> float:
    denom = u_i + h_i + b_i
    if denom <= 0:
        return 0.5
    return clip_prob((u_i + b_i) / denom, 0.01, 0.99)


def sku_profit_value(u_i: float, h_i: float, b_i: float, f_i: float, mu: float, e_min: float, q: int) -> float:
    if q <= 0:
        return 0.0
    return (u_i + h_i + b_i) * e_min - h_i * q - b_i * mu - f_i


def get_train_mu_for_horizon(mu_daily: float, horizon_days: int) -> float:
    return max(0.0, float(mu_daily) * float(horizon_days))


# -----------------------------
# Data loading
# -----------------------------
def load_data(sku_params_path: Path, daily_demand_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sku = pd.read_csv(sku_params_path)
    dem = pd.read_csv(daily_demand_path)
    dem["Date"] = pd.to_datetime(dem["Date"])
    dem = dem.sort_values("Date").reset_index(drop=True)

    sku_cols = [c for c in dem.columns if c != "Date"]
    sku = sku[sku["SKU"].isin(sku_cols)].copy()

    keep = ["Date"] + sku["SKU"].tolist()
    dem = dem[keep].copy()

    for c in dem.columns:
        if c != "Date":
            dem[c] = pd.to_numeric(dem[c], errors="coerce").fillna(0.0)

    return sku.reset_index(drop=True), dem


# -----------------------------
# Stage A: Li-style proxy MCI
# -----------------------------
def compute_mci_scores(train_dem: pd.DataFrame, sku_params: pd.DataFrame) -> pd.DataFrame:
    sku_cols = [c for c in train_dem.columns if c != "Date"]
    demand_presence = (train_dem[sku_cols] > 0).mean(axis=0).rename("presence_prob")
    demand_mean = train_dem[sku_cols].mean(axis=0).rename("train_mu_daily")
    demand_total = train_dem[sku_cols].sum(axis=0).rename("train_total")

    out = sku_params.merge(demand_presence, left_on="SKU", right_index=True, how="left")
    out = out.merge(demand_mean, left_on="SKU", right_index=True, how="left")
    out = out.merge(demand_total, left_on="SKU", right_index=True, how="left")
    out["presence_prob"] = out["presence_prob"].fillna(0.0)
    out["train_mu_daily"] = out["train_mu_daily"].fillna(0.0)
    out["train_total"] = out["train_total"].fillna(0.0)

    out["mci_score"] = 1_000_000 * out["presence_prob"] + 1_000 * out["train_mu_daily"] + out["train_total"]
    return out


def select_assortment_mci(scored_df: pd.DataFrame, k_sku: int) -> pd.DataFrame:
    ranked = scored_df.sort_values(
        ["mci_score", "presence_prob", "train_mu_daily", "train_total"],
        ascending=False
    ).reset_index(drop=True)
    selected = ranked.head(k_sku).copy()
    selected["selected"] = 1
    return selected


# -----------------------------
# Distribution + substitution prep
# -----------------------------
def build_dist_spec(row: pd.Series, mu_horizon: float) -> DistSpec:
    use_negbin = int(row.get("use_negbin", 0)) == 1
    r_val = float(row.get("r_negbin", np.nan))
    if use_negbin and np.isfinite(r_val) and r_val > 0 and r_val < 1e8:
        return DistSpec(kind="nbinom", mu=mu_horizon, r=r_val)
    return DistSpec(kind="poisson", mu=mu_horizon, r=None)


def build_substitution_matrix(
    df: pd.DataFrame,
    lambda_price: float = 2.0,
    lambda_quality: float = 1.0,
    base_strength: float = 0.35,
) -> pd.DataFrame:
    """
    Exogenous substitution matrix S where S[j,i] = fraction of unmet demand of j
    that can flow to i, based on similarity in avg_price / avg_profit / popularity.
    Row sums are <= base_strength, leaving mass for no-purchase / other channels.
    """
    tmp = df.copy().reset_index(drop=True)
    n = len(tmp)
    if n == 0:
        return pd.DataFrame()

    # quality proxy: avg_price then avg_profit then popularity
    price = tmp["avg_price"].astype(float).values
    profit = tmp["avg_profit"].astype(float).values if "avg_profit" in tmp.columns else np.zeros(n)
    pop = tmp["presence_prob"].astype(float).values if "presence_prob" in tmp.columns else np.zeros(n)

    def normalize(x):
        x = np.asarray(x, dtype=float)
        if np.allclose(x.max(), x.min()):
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    price_n = normalize(price)
    profit_n = normalize(profit)
    pop_n = normalize(pop)

    W = np.zeros((n, n), dtype=float)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            d_price = abs(price_n[i] - price_n[j])
            d_quality = abs(profit_n[i] - profit_n[j])

            # slight preference to nearby / similar products and slightly more popular ones
            sim = math.exp(-lambda_price * d_price - lambda_quality * d_quality)
            bias = 0.75 + 0.25 * pop_n[i]
            W[j, i] = sim * bias

    S = np.zeros((n, n), dtype=float)
    for j in range(n):
        row = W[j]
        s = row.sum()
        if s > 0:
            S[j] = base_strength * row / s

    return pd.DataFrame(S, index=tmp["SKU"].tolist(), columns=tmp["SKU"].tolist())


# -----------------------------
# Substitution-aware evaluation for a candidate q vector
# -----------------------------
def expected_sales_with_substitution(
    sku_list: List[str],
    q_map: Dict[str, int],
    mu_map: Dict[str, float],
    spec_map: Dict[str, DistSpec],
    S: pd.DataFrame,
    iterations: int = 3,
) -> Dict[str, float]:
    """
    Approximate effective demand / expected sales using a few fixed-point iterations:
      unmet_j = max(mu_j - sold_j, 0)
      extra_to_i = sum_j S[j,i] * unmet_j
      eff_mu_i = mu_i + extra_to_i
      sold_i = E[min(D_i_eff, q_i)] approximated with same distribution family and shifted mean

    This is a simplified exogenous-substitution analogue, not the exact Transchel dynamics.
    """
    sold = {}
    unmet = {}
    eff_mu = {sku: float(mu_map[sku]) for sku in sku_list}

    for _ in range(iterations):
        sold_new = {}
        unmet_new = {}
        for sku in sku_list:
            q = int(q_map.get(sku, 0))
            base_spec = spec_map[sku]
            spec_eff = DistSpec(kind=base_spec.kind, mu=max(0.0, eff_mu[sku]), r=base_spec.r)
            sold_val = expected_min(spec_eff, q)
            sold_new[sku] = sold_val
            unmet_new[sku] = max(0.0, eff_mu[sku] - sold_val)

        eff_mu_next = {sku: float(mu_map[sku]) for sku in sku_list}
        for j in sku_list:
            if j not in S.index:
                continue
            for i in sku_list:
                if i == j or i not in S.columns:
                    continue
                eff_mu_next[i] += float(S.loc[j, i]) * unmet_new[j]

        sold, unmet, eff_mu = sold_new, unmet_new, eff_mu_next

    return {
        sku: float(sold.get(sku, 0.0))
        for sku in sku_list
    }


def compute_plan_profit(
    df: pd.DataFrame,
    q_map: Dict[str, int],
    sold_map: Dict[str, float],
    mu_map: Dict[str, float],
) -> float:
    total = 0.0
    for _, row in df.iterrows():
        sku = row["SKU"]
        q = int(q_map.get(sku, 0))
        e_min = float(sold_map.get(sku, 0.0))
        mu_h = float(mu_map[sku])
        total += sku_profit_value(
            u_i=float(row["u_i"]),
            h_i=float(row["h_i"]),
            b_i=float(row["b_i"]),
            f_i=float(row["f_i"]),
            mu=mu_h,
            e_min=e_min,
            q=q,
        )
    return float(total)


# -----------------------------
# Stage B v2: substitution-aware local search
# -----------------------------
def transchel_like_allocate_inventory_v2(
    selected_df: pd.DataFrame,
    horizon_days: int,
    v_cap: float,
    w_cap: float,
    q_max: int = 30,
    lambda_price: float = 2.0,
    lambda_quality: float = 1.0,
    base_sub_strength: float = 0.35,
    local_iters: int = 400,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Upgraded Stage B:
      1) build exogenous substitution matrix on selected SKUs
      2) initialize q via critical-ratio quantiles
      3) repair to capacity
      4) hill-climb with substitution-aware expected sales/profit
    """
    df = selected_df.copy().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # prep stats
    mu_map: Dict[str, float] = {}
    spec_map: Dict[str, DistSpec] = {}
    seed_q: Dict[str, int] = {}
    sku_list = df["SKU"].tolist()

    for _, row in df.iterrows():
        sku = row["SKU"]
        mu_h = get_train_mu_for_horizon(float(row["train_mu_daily"]), horizon_days)
        mu_map[sku] = mu_h
        spec = build_dist_spec(row, mu_h)
        spec_map[sku] = spec
        alpha = critical_ratio(float(row["u_i"]), float(row["h_i"]), float(row["b_i"]))
        seed_q[sku] = min(q_max, demand_ppf(spec, alpha))

    # build substitution matrix
    S = build_substitution_matrix(
        df,
        lambda_price=lambda_price,
        lambda_quality=lambda_quality,
        base_strength=base_sub_strength,
    )

    # initial q
    q_map = {sku: int(seed_q[sku]) for sku in sku_list}

    def total_capacity(qm: Dict[str, int]) -> Tuple[float, float]:
        used_v = 0.0
        used_w = 0.0
        for _, row in df.iterrows():
            sku = row["SKU"]
            q = int(qm.get(sku, 0))
            used_v += float(row["v_i"]) * q
            used_w += float(row["w_i"]) * q
        return used_v, used_w

    def score_remove_one(sku: str, sold_map: Dict[str, float]) -> float:
        row = df.loc[df["SKU"] == sku].iloc[0]
        q = int(q_map[sku])
        if q <= 0:
            return -1e18
        cur = sku_profit_value(float(row["u_i"]), float(row["h_i"]), float(row["b_i"]), float(row["f_i"]), mu_map[sku], sold_map[sku], q)
        # rough local approximation by reducing one unit and recomputing own sales only
        spec = spec_map[sku]
        sold_minus = expected_min(DistSpec(spec.kind, mu_map[sku], spec.r), q - 1)
        new = sku_profit_value(float(row["u_i"]), float(row["h_i"]), float(row["b_i"]), float(row["f_i"]), mu_map[sku], sold_minus, q - 1)
        v = max(1e-9, float(row["v_i"]))
        w = max(1e-9, float(row["w_i"]))
        return (cur - new) / (v / max(v_cap, 1e-9) + w / max(w_cap, 1e-9))

    # repair to capacity by removing low value-density units
    sold_map = expected_sales_with_substitution(sku_list, q_map, mu_map, spec_map, S)
    used_v, used_w = total_capacity(q_map)
    while used_v > v_cap + 1e-9 or used_w > w_cap + 1e-9:
        candidates = [(score_remove_one(sku, sold_map), sku) for sku in sku_list if q_map[sku] > 0]
        if not candidates:
            break
        _, worst_sku = min(candidates, key=lambda x: x[0])
        q_map[worst_sku] -= 1
        sold_map = expected_sales_with_substitution(sku_list, q_map, mu_map, spec_map, S)
        used_v, used_w = total_capacity(q_map)

    # local search: try +1 or -1,+1 swaps
    best_sold = expected_sales_with_substitution(sku_list, q_map, mu_map, spec_map, S)
    best_profit = compute_plan_profit(df, q_map, best_sold, mu_map)
    best_q = dict(q_map)

    for _ in range(local_iters):
        improved = False
        used_v, used_w = total_capacity(best_q)

        # single-add moves
        add_candidates = []
        for _, row in df.iterrows():
            sku = row["SKU"]
            q = int(best_q[sku])
            if q >= q_max:
                continue
            new_v = used_v + float(row["v_i"])
            new_w = used_w + float(row["w_i"])
            if new_v <= v_cap + 1e-9 and new_w <= w_cap + 1e-9:
                trial_q = dict(best_q)
                trial_q[sku] += 1
                trial_sold = expected_sales_with_substitution(sku_list, trial_q, mu_map, spec_map, S)
                trial_profit = compute_plan_profit(df, trial_q, trial_sold, mu_map)
                add_candidates.append((trial_profit, trial_q, trial_sold))

        if add_candidates:
            trial_profit, trial_q, trial_sold = max(add_candidates, key=lambda x: x[0])
            if trial_profit > best_profit + 1e-8:
                best_profit = trial_profit
                best_q = trial_q
                best_sold = trial_sold
                improved = True

        if improved:
            continue

        # swap moves: remove one unit from a SKU and add one to another
        swap_best = None
        for sku_out in sku_list:
            if best_q[sku_out] <= 0:
                continue
            row_out = df.loc[df["SKU"] == sku_out].iloc[0]
            for sku_in in sku_list:
                if sku_in == sku_out or best_q[sku_in] >= q_max:
                    continue
                row_in = df.loc[df["SKU"] == sku_in].iloc[0]
                new_v = used_v - float(row_out["v_i"]) + float(row_in["v_i"])
                new_w = used_w - float(row_out["w_i"]) + float(row_in["w_i"])
                if new_v <= v_cap + 1e-9 and new_w <= w_cap + 1e-9:
                    trial_q = dict(best_q)
                    trial_q[sku_out] -= 1
                    trial_q[sku_in] += 1
                    trial_sold = expected_sales_with_substitution(sku_list, trial_q, mu_map, spec_map, S)
                    trial_profit = compute_plan_profit(df, trial_q, trial_sold, mu_map)
                    if (swap_best is None) or (trial_profit > swap_best[0]):
                        swap_best = (trial_profit, trial_q, trial_sold)

        if swap_best and swap_best[0] > best_profit + 1e-8:
            best_profit = swap_best[0]
            best_q = swap_best[1]
            best_sold = swap_best[2]
            improved = True

        if not improved:
            break

    used_v, used_w = total_capacity(best_q)

    rows = []
    for _, row in df.iterrows():
        sku = row["SKU"]
        q = int(best_q[sku])
        rows.append({
            "SKU": sku,
            "q_hybrid_v2": q,
            "expected_sales_horizon_subaware": float(best_sold[sku]),
            "mu_horizon": float(mu_map[sku]),
            "presence_prob": float(row["presence_prob"]),
            "train_mu_daily": float(row["train_mu_daily"]),
            "quality_proxy_avg_price": float(row["avg_price"]),
            "avg_profit": float(row.get("avg_profit", 0.0)),
            "v_i": float(row["v_i"]),
            "w_i": float(row["w_i"]),
            "u_i": float(row["u_i"]),
            "h_i": float(row["h_i"]),
            "b_i": float(row["b_i"]),
            "f_i": float(row["f_i"]),
        })

    out = pd.DataFrame(rows)
    out.attrs["used_v"] = used_v
    out.attrs["used_w"] = used_w
    out.attrs["total_profit"] = best_profit
    out.attrs["total_exp_sales"] = float(sum(best_sold.values()))
    out.attrs["n_listed"] = int((out["q_hybrid_v2"] > 0).sum())
    return out, S


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_week_static_plan(plan_df: pd.DataFrame, test_week: pd.DataFrame, q_col: str) -> Dict[str, float]:
    sku_to_q = dict(zip(plan_df["SKU"], plan_df[q_col]))
    sku_cols = [c for c in test_week.columns if c != "Date"]

    demand_total = 0.0
    fulfilled_total = 0.0
    listed_demand = 0.0
    full_day_days = 0
    positive_days = 0

    weekly_demand = test_week[sku_cols].sum(axis=0)

    for sku in sku_cols:
        d = float(weekly_demand.get(sku, 0.0))
        q = float(sku_to_q.get(sku, 0.0))
        demand_total += d
        fulfilled_total += min(d, q)
        if q > 0:
            listed_demand += d

    remaining = {sku: float(sku_to_q.get(sku, 0.0)) for sku in sku_cols}
    for _, row in test_week.iterrows():
        day_total = row[sku_cols].sum()
        if day_total > 0:
            positive_days += 1
        feasible = True
        for sku in sku_cols:
            d = float(row[sku])
            if d > remaining.get(sku, 0.0) + 1e-9:
                feasible = False
                break
        if day_total > 0 and feasible:
            full_day_days += 1
        # deplete partially / fully
        for sku in sku_cols:
            d = float(row[sku])
            take = min(d, remaining.get(sku, 0.0))
            remaining[sku] = max(0.0, remaining.get(sku, 0.0) - take)

    return {
        "test_total_demand": demand_total,
        "test_total_fulfilled": fulfilled_total,
        "line_fill_rate": fulfilled_total / demand_total if demand_total > 0 else 0.0,
        "listed_demand_share": listed_demand / demand_total if demand_total > 0 else 0.0,
        "full_day_fill_proxy": full_day_days / positive_days if positive_days > 0 else 0.0,
    }


# -----------------------------
# Rolling experiment
# -----------------------------
def run_rolling_experiment(
    sku_params: pd.DataFrame,
    daily_demand: pd.DataFrame,
    output_dir: Path,
    top_n: int = 200,
    k_sku: int = 120,
    train_weeks: int = 8,
    test_weeks: int = 1,
    horizon_days: int = 7,
    q_max: int = 30,
    v_cap: float = 800.0,
    w_cap: float = 400.0,
    lambda_price: float = 2.0,
    lambda_quality: float = 1.0,
    base_sub_strength: float = 0.35,
    local_iters: int = 400,
) -> None:
    ensure_dir(output_dir)

    sku_params = sku_params.sort_values("total_qty", ascending=False).head(top_n).copy()
    keep_skus = sku_params["SKU"].tolist()
    daily_demand = daily_demand[["Date"] + keep_skus].copy()

    total_days = len(daily_demand)
    train_len = train_weeks * 7
    test_len = test_weeks * 7

    plan_frames = []
    metric_rows = []
    sub_frames = []

    start = 0
    wid = 0
    while start + train_len + test_len <= total_days:
        train_df = daily_demand.iloc[start:start + train_len].copy()
        test_df = daily_demand.iloc[start + train_len:start + train_len + test_len].copy()

        scored = compute_mci_scores(train_df, sku_params)
        selected = select_assortment_mci(scored, min(k_sku, len(scored)))

        plan, S = transchel_like_allocate_inventory_v2(
            selected_df=selected,
            horizon_days=horizon_days,
            v_cap=v_cap,
            w_cap=w_cap,
            q_max=q_max,
            lambda_price=lambda_price,
            lambda_quality=lambda_quality,
            base_sub_strength=base_sub_strength,
            local_iters=local_iters,
        )

        eval_metrics = evaluate_week_static_plan(plan, test_df, q_col="q_hybrid_v2")

        plan["window_id"] = wid
        plan["train_start"] = train_df["Date"].min()
        plan["train_end"] = train_df["Date"].max()
        plan["test_start"] = test_df["Date"].min()
        plan["test_end"] = test_df["Date"].max()
        plan_frames.append(plan)

        if not S.empty:
            sub_long = S.stack().reset_index()
            sub_long.columns = ["sku_from", "sku_to", "sub_rate"]
            sub_long["window_id"] = wid
            sub_frames.append(sub_long)

        metric_rows.append({
            "window_id": wid,
            "train_start": str(train_df["Date"].min().date()),
            "train_end": str(train_df["Date"].max().date()),
            "test_start": str(test_df["Date"].min().date()),
            "test_end": str(test_df["Date"].max().date()),
            "n_selected": int(len(selected)),
            "n_stocked_positive_q": int((plan["q_hybrid_v2"] > 0).sum()),
            "used_volume": float(plan.attrs["used_v"]),
            "used_weight": float(plan.attrs["used_w"]),
            "expected_profit_horizon": float(plan.attrs["total_profit"]),
            "expected_sales_horizon": float(plan.attrs["total_exp_sales"]),
            **eval_metrics,
        })

        start += test_len
        wid += 1

    plans_df = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    sub_df = pd.concat(sub_frames, ignore_index=True) if sub_frames else pd.DataFrame()

    summary = {}
    if not metrics_df.empty:
        for col in [
            "line_fill_rate",
            "listed_demand_share",
            "full_day_fill_proxy",
            "expected_profit_horizon",
            "expected_sales_horizon",
            "n_stocked_positive_q",
            "used_volume",
            "used_weight",
        ]:
            summary[f"avg_{col}"] = float(metrics_df[col].mean())

    plans_df.to_csv(output_dir / "hybrid_v2_plans_by_window.csv", index=False)
    metrics_df.to_csv(output_dir / "hybrid_v2_window_metrics.csv", index=False)
    sub_df.to_csv(output_dir / "hybrid_v2_substitution_matrix_long.csv", index=False)

    if not plans_df.empty:
        latest = plans_df["window_id"].max()
        plans_df[plans_df["window_id"] == latest].sort_values(
            ["q_hybrid_v2", "expected_sales_horizon_subaware"], ascending=[False, False]
        ).to_csv(output_dir / "hybrid_v2_latest_window_plan.csv", index=False)

    with open(output_dir / "hybrid_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Li + Transchel hybrid baseline v2")
    p.add_argument("--sku_params", type=str, default=str(DEFAULT_DATA_DIR / "sku_params.csv"))
    p.add_argument("--daily_demand", type=str, default=str(DEFAULT_DATA_DIR / "daily_demand.csv"))
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--top_n", type=int, default=200)
    p.add_argument("--k_sku", type=int, default=120)
    p.add_argument("--train_weeks", type=int, default=8)
    p.add_argument("--test_weeks", type=int, default=1)
    p.add_argument("--horizon_days", type=int, default=7)
    p.add_argument("--q_max", type=int, default=30)
    p.add_argument("--v_cap", type=float, default=800.0)
    p.add_argument("--w_cap", type=float, default=400.0)
    p.add_argument("--lambda_price", type=float, default=2.0)
    p.add_argument("--lambda_quality", type=float, default=1.0)
    p.add_argument("--base_sub_strength", type=float, default=0.35)
    p.add_argument("--local_iters", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    sku_params, daily_demand = load_data(Path(args.sku_params), Path(args.daily_demand))
    run_rolling_experiment(
        sku_params=sku_params,
        daily_demand=daily_demand,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        k_sku=args.k_sku,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
        horizon_days=args.horizon_days,
        q_max=args.q_max,
        v_cap=args.v_cap,
        w_cap=args.w_cap,
        lambda_price=args.lambda_price,
        lambda_quality=args.lambda_quality,
        base_sub_strength=args.base_sub_strength,
        local_iters=args.local_iters,
    )
    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
