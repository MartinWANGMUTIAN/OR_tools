import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())

SCANNER_FILE = ROOT_DIR / "data" / "raw" / "scanner_data.csv"
MAIN_ROLLING_FILE = ROOT_DIR / "results" / "main_optimizer" / "rolling_results.csv"
MAIN_PLAN_FILE = ROOT_DIR / "results" / "main_optimizer" / "plans.csv"
HYBRID_ROLLING_FILE = ROOT_DIR / "results" / "literature" / "hybrid_v2" / "rolling_results.csv"
HYBRID_PLAN_FILE = ROOT_DIR / "results" / "literature" / "hybrid_v2" / "plans.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "literature" / "benchmark_vs_hybrid_v2"

DATE_COL = "Date"
ORDER_COL = "Transaction_ID"
SKU_COL = "SKU"
QTY_COL = "Quantity"
TRAIN_DAYS = 56
TEST_DAYS = 7
DAYFIRST = True
SCENARIO = "baseline"

MAIN_ALGO_COLS = {
    "greedy_nominal": ("greedy_nominal_lfr", "greedy_nominal_time"),
    "dp_nominal": ("dp_nominal_lfr", "dp_nominal_time"),
    "cpsat_nominal": ("cpsat_nominal_lfr", "cpsat_nominal_time"),
    "greedy_robust": ("greedy_robust_lfr", "greedy_robust_time"),
    "dp_robust": ("dp_robust_lfr", "dp_robust_time"),
    "cpsat_robust": ("cpsat_robust_lfr", "cpsat_robust_time"),
}

HYBRID_METHOD = "hybrid_v2"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_scanner_data(scanner_file: Path) -> pd.DataFrame:
    df = pd.read_csv(scanner_file)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.rename(columns={
        DATE_COL: "date",
        ORDER_COL: "order_id",
        SKU_COL: "sku",
        QTY_COL: "qty",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=DAYFIRST, errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "order_id", "sku"])
    df = df[df["qty"] > 0].copy()

    return (
        df.groupby(["date", "order_id", "sku"], as_index=False)["qty"]
        .sum()
        .sort_values(["date", "order_id", "sku"])
        .reset_index(drop=True)
    )


def build_windows_from_orders(order_lines: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.Series(sorted(order_lines["date"].dt.normalize().unique()))
    all_dates = pd.to_datetime(all_dates)

    windows = []
    window_id = 0
    start_idx = TRAIN_DAYS
    while start_idx + TEST_DAYS <= len(all_dates):
        windows.append({
            "window": window_id,
            "train_start": all_dates.iloc[start_idx - TRAIN_DAYS],
            "train_end": all_dates.iloc[start_idx - 1],
            "test_start": all_dates.iloc[start_idx],
            "test_end": all_dates.iloc[start_idx + TEST_DAYS - 1],
        })
        window_id += 1
        start_idx += TEST_DAYS

    return pd.DataFrame(windows)


def load_plans(plan_file: Path, scenario: str) -> pd.DataFrame:
    plans = pd.read_csv(plan_file)
    required_cols = {"window", "scenario", "algorithm", "sku", "stock_qty"}
    missing = required_cols - set(plans.columns)
    if missing:
        raise ValueError(f"{plan_file} 缺少字段: {missing}")

    plans["window"] = plans["window"].astype(int)
    plans["scenario"] = plans["scenario"].astype(str)
    plans["algorithm"] = plans["algorithm"].astype(str)
    plans["sku"] = plans["sku"].astype(str)
    plans["stock_qty"] = pd.to_numeric(plans["stock_qty"], errors="coerce").fillna(0)

    plans = plans[(plans["scenario"] == scenario) & (plans["stock_qty"] > 0)].copy()
    return plans


def load_main_metrics(rolling_file: Path, scenario: str) -> pd.DataFrame:
    rolling = pd.read_csv(rolling_file)
    rolling["window"] = rolling["window"].astype(int)
    rolling["scenario"] = rolling["scenario"].astype(str)
    rolling = rolling[rolling["scenario"] == scenario].copy()

    rows = []
    for _, row in rolling.iterrows():
        for algorithm, (lfr_col, time_col) in MAIN_ALGO_COLS.items():
            rows.append({
                "window": int(row["window"]),
                "scenario": scenario,
                "algorithm": algorithm,
                "aggregated_lfr": float(row[lfr_col]),
                "runtime_sec": float(row[time_col]),
            })
    return pd.DataFrame(rows)


def load_hybrid_metrics(rolling_file: Path, scenario: str) -> pd.DataFrame:
    rolling = pd.read_csv(rolling_file)
    rolling["window"] = rolling["window"].astype(int)
    rolling["scenario"] = rolling["scenario"].astype(str)
    rolling = rolling[rolling["scenario"] == scenario].copy()

    return rolling.rename(columns={
        "hybrid_v2_lfr": "aggregated_lfr",
        "hybrid_v2_time": "runtime_sec",
    })[["window", "scenario", "aggregated_lfr", "runtime_sec"]].assign(algorithm=HYBRID_METHOD)


def simulate_one_plan(order_lines_test: pd.DataFrame, plan_df: pd.DataFrame, whole_order: bool) -> dict:
    inventory = dict(zip(plan_df["sku"], plan_df["stock_qty"]))

    total_orders = 0
    full_orders = 0
    total_lines = 0
    filled_lines = 0
    total_qty = 0.0
    filled_qty = 0.0

    order_lines_test = order_lines_test.sort_values(["date", "order_id", "sku"]).copy()

    for _, grp in order_lines_test.groupby("order_id", sort=False):
        total_orders += 1
        order_full = True
        line_results = []

        for _, row in grp.iterrows():
            sku = row["sku"]
            req = float(row["qty"])
            total_lines += 1
            total_qty += req

            if inventory.get(sku, 0) >= req:
                line_results.append((sku, req, True))
            else:
                line_results.append((sku, req, False))
                order_full = False

        if whole_order:
            if order_full:
                full_orders += 1
                for sku, req, _ in line_results:
                    inventory[sku] -= req
                    filled_lines += 1
                    filled_qty += req
        else:
            for sku, req, ok in line_results:
                if ok:
                    inventory[sku] -= req
                    filled_lines += 1
                    filled_qty += req
            if order_full:
                full_orders += 1

    ofr = full_orders / total_orders if total_orders > 0 else np.nan
    lfr = filled_lines / total_lines if total_lines > 0 else np.nan
    qfr = filled_qty / total_qty if total_qty > 0 else np.nan
    return {
        "basket_ofr": ofr,
        "basket_lfr": lfr,
        "basket_qfr": qfr,
        "n_orders": total_orders,
        "n_lines": total_lines,
        "total_qty": total_qty,
    }


def run_basket_simulation(
    order_lines: pd.DataFrame,
    windows: pd.DataFrame,
    plans: pd.DataFrame,
    metrics_long: pd.DataFrame,
    whole_order: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []

    for _, w in windows.iterrows():
        window_id = int(w["window"])
        test_start = pd.to_datetime(w["test_start"])
        test_end = pd.to_datetime(w["test_end"])

        test_mask = (order_lines["date"] >= test_start) & (order_lines["date"] <= test_end)
        test_orders = order_lines.loc[test_mask].copy()
        if test_orders.empty:
            continue

        plan_w = plans[plans["window"] == window_id].copy()
        if plan_w.empty:
            continue

        for (scenario, algorithm), grp in plan_w.groupby(["scenario", "algorithm"]):
            sim = simulate_one_plan(test_orders, grp[["sku", "stock_qty"]].copy(), whole_order=whole_order)
            results.append({
                "window": window_id,
                "scenario": scenario,
                "algorithm": algorithm,
                "test_start": test_start,
                "test_end": test_end,
                **sim,
            })

    basket_window = pd.DataFrame(results)
    if basket_window.empty:
        raise ValueError("没有得到任何篮子模拟结果，请检查共同窗口和方案文件。")

    basket_window = basket_window.merge(
        metrics_long,
        on=["window", "scenario", "algorithm"],
        how="left",
    )
    basket_window["proxy_gap_lfr"] = basket_window["aggregated_lfr"] - basket_window["basket_lfr"]

    basket_summary = (
        basket_window.groupby(["scenario", "algorithm"], as_index=False)
        .agg(
            basket_ofr_mean=("basket_ofr", "mean"),
            basket_lfr_mean=("basket_lfr", "mean"),
            basket_qfr_mean=("basket_qfr", "mean"),
            aggregated_lfr_mean=("aggregated_lfr", "mean"),
            proxy_gap_lfr_mean=("proxy_gap_lfr", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            n_windows=("window", "count"),
            avg_orders=("n_orders", "mean"),
            avg_lines=("n_lines", "mean"),
        )
        .sort_values(["scenario", "basket_lfr_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return basket_window, basket_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark main methods against hybrid_v2 on shared baseline windows.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default=SCENARIO)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    order_lines = load_scanner_data(SCANNER_FILE)
    windows = build_windows_from_orders(order_lines)

    main_plans = load_plans(MAIN_PLAN_FILE, args.scenario)
    hybrid_plans = load_plans(HYBRID_PLAN_FILE, args.scenario)
    main_metrics = load_main_metrics(MAIN_ROLLING_FILE, args.scenario)
    hybrid_metrics = load_hybrid_metrics(HYBRID_ROLLING_FILE, args.scenario)

    common_windows = sorted(
        set(main_plans["window"].unique())
        & set(hybrid_plans["window"].unique())
        & set(main_metrics["window"].unique())
        & set(hybrid_metrics["window"].unique())
        & set(windows["window"].unique())
    )
    if not common_windows:
        raise ValueError("没有找到共同窗口，无法做 benchmark。")

    windows = windows[windows["window"].isin(common_windows)].copy()
    main_plans = main_plans[main_plans["window"].isin(common_windows)].copy()
    hybrid_plans = hybrid_plans[hybrid_plans["window"].isin(common_windows)].copy()
    main_metrics = main_metrics[main_metrics["window"].isin(common_windows)].copy()
    hybrid_metrics = hybrid_metrics[hybrid_metrics["window"].isin(common_windows)].copy()

    combined_plans = pd.concat([main_plans, hybrid_plans], ignore_index=True)
    combined_metrics = pd.concat([main_metrics, hybrid_metrics], ignore_index=True)

    whole_window, whole_summary = run_basket_simulation(
        order_lines=order_lines,
        windows=windows,
        plans=combined_plans,
        metrics_long=combined_metrics,
        whole_order=True,
    )
    partial_window, partial_summary = run_basket_simulation(
        order_lines=order_lines,
        windows=windows,
        plans=combined_plans,
        metrics_long=combined_metrics,
        whole_order=False,
    )

    windows[["window", "train_start", "train_end", "test_start", "test_end"]].to_csv(
        args.output_dir / "benchmark_common_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    whole_window.to_csv(args.output_dir / "benchmark_whole_window.csv", index=False, encoding="utf-8-sig")
    whole_summary.to_csv(args.output_dir / "benchmark_whole_summary.csv", index=False, encoding="utf-8-sig")
    partial_window.to_csv(args.output_dir / "benchmark_partial_window.csv", index=False, encoding="utf-8-sig")
    partial_summary.to_csv(args.output_dir / "benchmark_partial_summary.csv", index=False, encoding="utf-8-sig")
    combined_metrics.sort_values(["window", "algorithm"]).to_csv(
        args.output_dir / "benchmark_proxy_runtime_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"共同窗口数: {len(common_windows)}")
    print(f"输出目录: {args.output_dir}")
    print(f"整单汇总: {args.output_dir / 'benchmark_whole_summary.csv'}")
    print(f"部分履约汇总: {args.output_dir / 'benchmark_partial_summary.csv'}")


if __name__ == "__main__":
    main()
