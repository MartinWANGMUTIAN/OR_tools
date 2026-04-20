
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Li (2024) + Transchel et al. (2022) hybrid baseline v2
------------------------------------------------------
Upgraded version with a simplified EXOGENOUS substitution matrix.

Still NOT an exact reproduction of either paper:
- Li et al. needs basket-level orders; we only have daily SKU totals
- Transchel et al. needs endogenous utility-based substitution matrices for all
  availability states; here we use an exogenous matrix estimated from similarity

Use:
python li_transchel_hybrid_baseline_v2.py \
  --sku_params ".../data/processed/sku_params.csv" \
  --daily_demand ".../data/processed/daily_demand.csv" \
  --output_dir ".../results/literature/hybrid_v2"
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "literature" / "hybrid_v2"
SLOT_V_COST = 2.0
SLOT_W_COST = 0.5


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


def total_capacity_with_slots(
    q_arr: np.ndarray,
    v_arr: np.ndarray,
    w_arr: np.ndarray,
    slot_v_cost: float = SLOT_V_COST,
    slot_w_cost: float = SLOT_W_COST,
) -> Tuple[float, float]:
    listed = (q_arr > 0).astype(float)
    used_v = float(np.dot(v_arr, q_arr) + slot_v_cost * listed.sum())
    used_w = float(np.dot(w_arr, q_arr) + slot_w_cost * listed.sum())
    return used_v, used_w


def marginal_capacity_delta(
    current_q: int,
    delta_q: int,
    unit_v: float,
    unit_w: float,
    slot_v_cost: float = SLOT_V_COST,
    slot_w_cost: float = SLOT_W_COST,
) -> Tuple[float, float]:
    new_q = current_q + delta_q
    if new_q < 0:
        raise ValueError("inventory cannot become negative")

    slot_delta = 0.0
    if current_q <= 0 < new_q:
        slot_delta = 1.0
    elif current_q > 0 and new_q <= 0:
        slot_delta = -1.0

    delta_v = float(unit_v * delta_q + slot_v_cost * slot_delta)
    delta_w = float(unit_w * delta_q + slot_w_cost * slot_delta)
    return delta_v, delta_w


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
    local_iters: int = 10,
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

    print(
        f"  [Stage B] selected={len(df)}, q_max={q_max}, local_iters={local_iters}",
        flush=True,
    )

    sku_list = df["SKU"].tolist()
    n = len(sku_list)

    v_arr = df["v_i"].astype(float).to_numpy()
    w_arr = df["w_i"].astype(float).to_numpy()
    u_arr = df["u_i"].astype(float).to_numpy()
    h_arr = df["h_i"].astype(float).to_numpy()
    b_arr = df["b_i"].astype(float).to_numpy()
    f_arr = df["f_i"].astype(float).to_numpy()
    price_arr = df["avg_price"].astype(float).to_numpy()
    profit_arr = (
        df["avg_profit"].fillna(0.0).astype(float).to_numpy()
        if "avg_profit" in df.columns
        else np.zeros(n, dtype=float)
    )
    presence_arr = df["presence_prob"].astype(float).to_numpy()
    train_mu_daily_arr = df["train_mu_daily"].astype(float).to_numpy()

    # prep stats
    mu_arr = np.zeros(n, dtype=float)
    spec_list: List[DistSpec] = []
    q_seed_arr = np.zeros(n, dtype=int)

    for idx, row in df.iterrows():
        mu_h = get_train_mu_for_horizon(float(row["train_mu_daily"]), horizon_days)
        mu_arr[idx] = mu_h
        spec = build_dist_spec(row, mu_h)
        spec_list.append(spec)
        alpha = critical_ratio(float(row["u_i"]), float(row["h_i"]), float(row["b_i"]))
        q_seed_arr[idx] = min(q_max, demand_ppf(spec, alpha))

    # build substitution matrix
    S = build_substitution_matrix(
        df,
        lambda_price=lambda_price,
        lambda_quality=lambda_quality,
        base_strength=base_sub_strength,
    )
    S_arr = S.to_numpy(dtype=float) if not S.empty else np.zeros((n, n), dtype=float)

    # initial q
    q_arr = q_seed_arr.copy()

    def q_arr_to_map(qm: np.ndarray) -> Dict[str, int]:
        return {sku: int(qm[idx]) for idx, sku in enumerate(sku_list)}

    def sold_arr_to_map(sold: np.ndarray) -> Dict[str, float]:
        return {sku: float(sold[idx]) for idx, sku in enumerate(sku_list)}

    def total_capacity(qm: np.ndarray) -> Tuple[float, float]:
        return total_capacity_with_slots(qm, v_arr, w_arr)

    def expected_sales_with_substitution_arr(qm: np.ndarray, iterations: int = 3) -> np.ndarray:
        eff_mu = mu_arr.copy()
        sold = np.zeros(n, dtype=float)
        for _ in range(iterations):
            sold = np.array(
                [
                    expected_min(DistSpec(spec.kind, float(eff_mu[idx]), spec.r), int(qm[idx]))
                    for idx, spec in enumerate(spec_list)
                ],
                dtype=float,
            )
            unmet = np.maximum(eff_mu - sold, 0.0)
            eff_mu = mu_arr + unmet @ S_arr
        return sold

    def compute_plan_profit_arr(qm: np.ndarray, sold: np.ndarray) -> float:
        total = (u_arr + h_arr + b_arr) * sold - h_arr * qm - b_arr * mu_arr - f_arr
        total = np.where(qm > 0, total, 0.0)
        return float(total.sum())

    def score_remove_one(idx: int, sold: np.ndarray, qm: np.ndarray) -> float:
        q = int(qm[idx])
        if q <= 0:
            return -1e18
        cur = sku_profit_value(u_arr[idx], h_arr[idx], b_arr[idx], f_arr[idx], mu_arr[idx], sold[idx], q)
        # rough local approximation by reducing one unit and recomputing own sales only
        spec = spec_list[idx]
        sold_minus = expected_min(DistSpec(spec.kind, mu_arr[idx], spec.r), q - 1)
        new = sku_profit_value(u_arr[idx], h_arr[idx], b_arr[idx], f_arr[idx], mu_arr[idx], sold_minus, q - 1)
        v = max(1e-9, v_arr[idx])
        w = max(1e-9, w_arr[idx])
        return (cur - new) / (v / max(v_cap, 1e-9) + w / max(w_cap, 1e-9))

    def score_add_one(idx: int, sold: np.ndarray, qm: np.ndarray) -> float:
        q = int(qm[idx])
        if q >= q_max:
            return -1e18
        spec = spec_list[idx]
        cur = sku_profit_value(u_arr[idx], h_arr[idx], b_arr[idx], f_arr[idx], mu_arr[idx], sold[idx], q)
        sold_plus = expected_min(DistSpec(spec.kind, mu_arr[idx], spec.r), q + 1)
        new = sku_profit_value(u_arr[idx], h_arr[idx], b_arr[idx], f_arr[idx], mu_arr[idx], sold_plus, q + 1)
        v = max(1e-9, v_arr[idx])
        w = max(1e-9, w_arr[idx])
        return (new - cur) / (v / max(v_cap, 1e-9) + w / max(w_cap, 1e-9))

    # repair to capacity by removing low value-density units
    used_v, used_w = total_capacity(q_arr)
    if used_v > v_cap + 1e-9 or used_w > w_cap + 1e-9:
        scale = min(
            v_cap / max(used_v, 1e-9),
            w_cap / max(used_w, 1e-9),
            1.0,
        )
        q_arr = np.floor(q_arr * scale).astype(int)
        print(
            f"  [Stage B] scaled seed inventory by {scale:.4f} before fine repair",
            flush=True,
        )

    sold_arr = expected_sales_with_substitution_arr(q_arr)
    used_v, used_w = total_capacity(q_arr)
    repair_steps = 0
    while used_v > v_cap + 1e-9 or used_w > w_cap + 1e-9:
        candidates = [(score_remove_one(idx, sold_arr, q_arr), idx) for idx in range(n) if q_arr[idx] > 0]
        if not candidates:
            break
        _, worst_idx = min(candidates, key=lambda x: x[0])
        q_arr[worst_idx] -= 1
        sold_arr = expected_sales_with_substitution_arr(q_arr)
        used_v, used_w = total_capacity(q_arr)
        repair_steps += 1

    print(
        f"  [Stage B] repair done steps={repair_steps}, used_v={used_v:.2f}, used_w={used_w:.2f}",
        flush=True,
    )

    # local search: try +1 or -1,+1 swaps
    best_sold_arr = expected_sales_with_substitution_arr(q_arr)
    best_profit = compute_plan_profit_arr(q_arr, best_sold_arr)
    best_q_arr = q_arr.copy()
    add_pool_size = min(8, n)
    swap_pool_size = min(6, n)
    print(
        f"  [Stage B] local search start add_pool={add_pool_size}, swap_pool={swap_pool_size}",
        flush=True,
    )

    completed_iters = 0
    for it in range(local_iters):
        completed_iters = it + 1
        improved = False
        used_v, used_w = total_capacity(best_q_arr)

        if it > 0 and it % 10 == 0:
            print(
                f"  [Stage B] local search progress iter={it}/{local_iters}, profit={best_profit:.2f}",
                flush=True,
            )

        # single-add moves
        add_candidates = []
        add_order = sorted(
            range(n),
            key=lambda idx: score_add_one(idx, best_sold_arr, best_q_arr),
            reverse=True,
        )
        for idx in add_order[:add_pool_size]:
            q = int(best_q_arr[idx])
            if q >= q_max:
                continue
            delta_v, delta_w = marginal_capacity_delta(
                current_q=q,
                delta_q=1,
                unit_v=v_arr[idx],
                unit_w=w_arr[idx],
            )
            new_v = used_v + delta_v
            new_w = used_w + delta_w
            if new_v <= v_cap + 1e-9 and new_w <= w_cap + 1e-9:
                trial_q = best_q_arr.copy()
                trial_q[idx] += 1
                trial_sold = expected_sales_with_substitution_arr(trial_q)
                trial_profit = compute_plan_profit_arr(trial_q, trial_sold)
                add_candidates.append((trial_profit, trial_q, trial_sold))

        if add_candidates:
            trial_profit, trial_q, trial_sold = max(add_candidates, key=lambda x: x[0])
            if trial_profit > best_profit + 1e-8:
                best_profit = trial_profit
                best_q_arr = trial_q
                best_sold_arr = trial_sold
                improved = True

        if improved:
            continue

        # swap moves: remove one unit from a SKU and add one to another
        swap_best = None
        remove_order = sorted(range(n), key=lambda idx: score_remove_one(idx, best_sold_arr, best_q_arr))
        add_order = sorted(range(n), key=lambda idx: score_add_one(idx, best_sold_arr, best_q_arr), reverse=True)
        for idx_out in remove_order[:swap_pool_size]:
            if best_q_arr[idx_out] <= 0:
                continue
            for idx_in in add_order[:swap_pool_size]:
                if idx_in == idx_out or best_q_arr[idx_in] >= q_max:
                    continue
                delta_v_out, delta_w_out = marginal_capacity_delta(
                    current_q=int(best_q_arr[idx_out]),
                    delta_q=-1,
                    unit_v=v_arr[idx_out],
                    unit_w=w_arr[idx_out],
                )
                delta_v_in, delta_w_in = marginal_capacity_delta(
                    current_q=int(best_q_arr[idx_in]),
                    delta_q=1,
                    unit_v=v_arr[idx_in],
                    unit_w=w_arr[idx_in],
                )
                new_v = used_v + delta_v_out + delta_v_in
                new_w = used_w + delta_w_out + delta_w_in
                if new_v <= v_cap + 1e-9 and new_w <= w_cap + 1e-9:
                    trial_q = best_q_arr.copy()
                    trial_q[idx_out] -= 1
                    trial_q[idx_in] += 1
                    trial_sold = expected_sales_with_substitution_arr(trial_q)
                    trial_profit = compute_plan_profit_arr(trial_q, trial_sold)
                    if (swap_best is None) or (trial_profit > swap_best[0]):
                        swap_best = (trial_profit, trial_q, trial_sold)

        if swap_best and swap_best[0] > best_profit + 1e-8:
            best_profit = swap_best[0]
            best_q_arr = swap_best[1]
            best_sold_arr = swap_best[2]
            improved = True

        if not improved:
            break

    print(
        f"  [Stage B] local search done iters={completed_iters}, profit={best_profit:.2f}",
        flush=True,
    )

    used_v, used_w = total_capacity(best_q_arr)

    rows = []
    for idx, row in df.iterrows():
        sku = row["SKU"]
        q = int(best_q_arr[idx])
        rows.append({
            "SKU": sku,
            "q_hybrid_v2": q,
            "expected_sales_horizon_subaware": float(best_sold_arr[idx]),
            "mu_horizon": float(mu_arr[idx]),
            "presence_prob": float(presence_arr[idx]),
            "train_mu_daily": float(train_mu_daily_arr[idx]),
            "quality_proxy_avg_price": float(price_arr[idx]),
            "avg_profit": float(profit_arr[idx]),
            "v_i": float(v_arr[idx]),
            "w_i": float(w_arr[idx]),
            "u_i": float(u_arr[idx]),
            "h_i": float(h_arr[idx]),
            "b_i": float(b_arr[idx]),
            "f_i": float(f_arr[idx]),
        })

    out = pd.DataFrame(rows)
    out.attrs["used_v"] = used_v
    out.attrs["used_w"] = used_w
    out.attrs["total_profit"] = best_profit
    out.attrs["total_exp_sales"] = float(best_sold_arr.sum())
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
    n_listed = 0

    weekly_demand = test_week[sku_cols].sum(axis=0)
    stockout_flags = []

    for sku in sku_cols:
        d = float(weekly_demand.get(sku, 0.0))
        q = float(sku_to_q.get(sku, 0.0))
        demand_total += d
        fulfilled_total += min(d, q)
        if q > 0:
            listed_demand += d
            n_listed += 1
        stockout_flags.append(1.0 if d > q + 1e-9 else 0.0)

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
        "avg_stockout": float(np.mean(stockout_flags)) if stockout_flags else 0.0,
        "n_listed": int(n_listed),
        "listed_demand_share": listed_demand / demand_total if demand_total > 0 else 0.0,
        "full_day_fill_proxy": full_day_days / positive_days if positive_days > 0 else 0.0,
    }


def extract_standard_plan_records(
    plan_df: pd.DataFrame,
    window_id: int,
    scenario_name: str,
    algorithm_key: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, row in plan_df.iterrows():
        q = int(row["q_hybrid_v2"])
        if q > 0:
            records.append(
                {
                    "window": window_id,
                    "scenario": scenario_name,
                    "algorithm": algorithm_key,
                    "sku": row["SKU"],
                    "stock_qty": q,
                }
            )
    return records


def summarize_standard_results(df: pd.DataFrame, scenario_name: str, method_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "method",
                "lfr_mean",
                "lfr_std",
                "stockout_mean",
                "stockout_std",
                "time_mean",
                "n_listed_mean",
                "n_windows",
            ]
        )

    lfr_col = f"{method_name}_lfr"
    stockout_col = f"{method_name}_stockout"
    time_col = f"{method_name}_time"
    n_listed_col = f"{method_name}_n_listed"

    return pd.DataFrame(
        [
            {
                "scenario": scenario_name,
                "method": method_name,
                "lfr_mean": float(df[lfr_col].mean()),
                "lfr_std": float(df[lfr_col].std(ddof=1)) if len(df) > 1 else 0.0,
                "stockout_mean": float(df[stockout_col].mean()),
                "stockout_std": float(df[stockout_col].std(ddof=1)) if len(df) > 1 else 0.0,
                "time_mean": float(df[time_col].mean()),
                "n_listed_mean": float(df[n_listed_col].mean()),
                "n_windows": int(len(df)),
            }
        ]
    )


# -----------------------------
# Rolling experiment
# -----------------------------
def run_rolling_experiment(
    sku_params: pd.DataFrame,
    daily_demand: pd.DataFrame,
    output_dir: Path,
    scenario_name: str = "baseline",
    method_name: str = "hybrid_v2",
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
    local_iters: int = 10,
) -> None:
    ensure_dir(output_dir)

    sku_params = sku_params.sort_values("total_qty", ascending=False).head(top_n).copy()
    keep_skus = sku_params["SKU"].tolist()
    daily_demand = daily_demand[["Date"] + keep_skus].copy()

    total_days = len(daily_demand)
    train_len = train_weeks * 7
    test_len = test_weeks * 7
    n_windows = max(0, (total_days - train_len - test_len) // test_len + 1)

    print(
        f"[Hybrid v2] scenario={scenario_name}, method={method_name}, "
        f"top_n={top_n}, k_sku={k_sku}, windows={n_windows}, local_iters={local_iters}",
        flush=True,
    )

    plan_frames = []
    metric_rows = []
    sub_frames = []
    rolling_rows = []
    standard_plan_rows = []

    start = 0
    wid = 0
    while start + train_len + test_len <= total_days:
        print(f"[Window {wid + 1}/{n_windows}] start", flush=True)
        train_df = daily_demand.iloc[start:start + train_len].copy()
        test_df = daily_demand.iloc[start + train_len:start + train_len + test_len].copy()

        scored = compute_mci_scores(train_df, sku_params)
        selected = select_assortment_mci(scored, min(k_sku, len(scored)))

        t0 = time.time()
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
        elapsed = time.time() - t0

        eval_metrics = evaluate_week_static_plan(plan, test_df, q_col="q_hybrid_v2")

        plan["window_id"] = wid
        plan["train_start"] = train_df["Date"].min()
        plan["train_end"] = train_df["Date"].max()
        plan["test_start"] = test_df["Date"].min()
        plan["test_end"] = test_df["Date"].max()
        plan_frames.append(plan)
        standard_plan_rows.extend(
            extract_standard_plan_records(
                plan_df=plan,
                window_id=wid,
                scenario_name=scenario_name,
                algorithm_key=method_name,
            )
        )

        if not S.empty:
            sub_long = S.stack().reset_index()
            sub_long.columns = ["sku_from", "sku_to", "sub_rate"]
            sub_long["window_id"] = wid
            sub_frames.append(sub_long)

        metric_rows.append({
            "window_id": wid,
            "scenario": scenario_name,
            "method": method_name,
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
            "runtime_sec": float(elapsed),
            **eval_metrics,
        })
        rolling_rows.append(
            {
                "window": wid,
                "scenario": scenario_name,
                f"{method_name}_lfr": float(eval_metrics["line_fill_rate"]),
                f"{method_name}_stockout": float(eval_metrics["avg_stockout"]),
                f"{method_name}_time": float(elapsed),
                f"{method_name}_n_listed": int(eval_metrics["n_listed"]),
            }
        )

        print(
            f"[Window {wid + 1}/{n_windows}] done "
            f"lfr={eval_metrics['line_fill_rate']:.4f}, "
            f"stockout={eval_metrics['avg_stockout']:.4f}, "
            f"time={elapsed:.2f}s",
            flush=True,
        )

        start += test_len
        wid += 1

    plans_df = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    sub_df = pd.concat(sub_frames, ignore_index=True) if sub_frames else pd.DataFrame()
    rolling_df = pd.DataFrame(rolling_rows)
    standard_plans_df = pd.DataFrame(standard_plan_rows)
    standard_summary_df = summarize_standard_results(rolling_df, scenario_name, method_name)

    summary = {}
    if not metrics_df.empty:
        for col in [
            "line_fill_rate",
            "listed_demand_share",
            "full_day_fill_proxy",
            "expected_profit_horizon",
            "expected_sales_horizon",
            "avg_stockout",
            "n_stocked_positive_q",
            "used_volume",
            "used_weight",
            "runtime_sec",
        ]:
            summary[f"avg_{col}"] = float(metrics_df[col].mean())

    rolling_df.to_csv(output_dir / "rolling_results.csv", index=False)
    standard_plans_df.to_csv(output_dir / "plans.csv", index=False)
    standard_summary_df.to_csv(output_dir / "summary.csv", index=False)
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

    print(f"[Hybrid v2] finished. Results saved to: {output_dir}", flush=True)

    return rolling_df, standard_plans_df, standard_summary_df, metrics_df, plans_df, sub_df


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Li + Transchel hybrid baseline v2")
    p.add_argument("--sku_params", type=str, default=str(DEFAULT_DATA_DIR / "sku_params.csv"))
    p.add_argument("--daily_demand", type=str, default=str(DEFAULT_DATA_DIR / "daily_demand.csv"))
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--scenario_name", type=str, default="baseline")
    p.add_argument("--method_name", type=str, default="hybrid_v2")
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
    p.add_argument("--local_iters", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    sku_params, daily_demand = load_data(Path(args.sku_params), Path(args.daily_demand))
    run_rolling_experiment(
        sku_params=sku_params,
        daily_demand=daily_demand,
        output_dir=Path(args.output_dir),
        scenario_name=args.scenario_name,
        method_name=args.method_name,
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
