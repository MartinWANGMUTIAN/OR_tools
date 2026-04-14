import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 1. 参数区
# =========================
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
SCANNER_FILE = ROOT_DIR / "data" / "raw" / "scanner_data.csv"
ROLLING_FILE = ROOT_DIR / "results" / "main_optimizer" / "rolling_results.csv"
PLAN_FILE = ROOT_DIR / "results" / "main_optimizer" / "plans.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "evaluation" / "whole_order"

OUTPUT_WINDOW_FILE = OUTPUT_DIR / "basket_window_results.csv"
OUTPUT_SUMMARY_FILE = OUTPUT_DIR / "basket_summary.csv"
OUTPUT_PROXY_FILE = OUTPUT_DIR / "basket_proxy_gap.csv"

DATE_COL = "Date"
ORDER_COL = "Transaction_ID"
SKU_COL = "SKU"
QTY_COL = "Quantity"

# 这里要和你第一步 rolling window 的设定一致
TRAIN_DAYS = 56
TEST_DAYS = 7

# 日期格式：
# 你这份数据像 02/01/2016
# 如果这是 “日/月/年”，设 dayfirst=True
# 如果这是 “月/日/年”，设 dayfirst=False
DAYFIRST = True


# =========================
# 2. 读取并清洗订单明细
# =========================
def load_scanner_data(scanner_file: str) -> pd.DataFrame:
    df = pd.read_csv(scanner_file)

    # 删掉无用索引列
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 重命名为统一格式
    df = df.rename(columns={
        DATE_COL: "date",
        ORDER_COL: "order_id",
        SKU_COL: "sku",
        QTY_COL: "qty"
    })

    # 日期
    df["date"] = pd.to_datetime(df["date"], dayfirst=DAYFIRST, errors="coerce")

    # 数量
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)

    # 基本过滤
    df = df.dropna(subset=["date", "order_id", "sku"])
    df = df[df["qty"] > 0].copy()

    # 同一订单里如果同一个 SKU 出现多次，合并
    df = (
        df.groupby(["date", "order_id", "sku"], as_index=False)["qty"]
        .sum()
        .sort_values(["date", "order_id", "sku"])
        .reset_index(drop=True)
    )

    return df


# =========================
# 3. 构造 rolling window 测试区间
# =========================
def build_windows_from_orders(order_lines: pd.DataFrame,
                              train_days: int,
                              test_days: int) -> pd.DataFrame:
    """
    根据订单日期生成 rolling window。
    默认每次向前滚动 7 天，与你现在 45 个窗口的思路一致。
    """
    all_dates = pd.Series(sorted(order_lines["date"].dt.normalize().unique()))
    all_dates = pd.to_datetime(all_dates)

    windows = []
    window_id = 0

    start_idx = train_days
    while start_idx + test_days <= len(all_dates):
        train_start = all_dates.iloc[start_idx - train_days]
        train_end = all_dates.iloc[start_idx - 1]
        test_start = all_dates.iloc[start_idx]
        test_end = all_dates.iloc[start_idx + test_days - 1]

        windows.append({
            "window": window_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        })

        window_id += 1
        start_idx += test_days   # 每次滚动 7 天

    return pd.DataFrame(windows)


# =========================
# 4. 读取方案文件
# =========================
def load_plans(plan_file: str) -> pd.DataFrame:
    """
    plans.csv 格式：
    window,scenario,algorithm,sku,stock_qty
    """
    plans = pd.read_csv(plan_file)

    required_cols = {"window", "scenario", "algorithm", "sku", "stock_qty"}
    missing = required_cols - set(plans.columns)
    if missing:
        raise ValueError(f"plans.csv 缺少字段: {missing}")

    plans["window"] = plans["window"].astype(int)
    plans["scenario"] = plans["scenario"].astype(str)
    plans["algorithm"] = plans["algorithm"].astype(str)
    plans["sku"] = plans["sku"].astype(str)
    plans["stock_qty"] = pd.to_numeric(plans["stock_qty"], errors="coerce").fillna(0)

    plans = plans[plans["stock_qty"] > 0].copy()
    return plans


# =========================
# 5. 单窗口 + 单算法 回放
# =========================
def simulate_one_plan(order_lines_test: pd.DataFrame,
                      plan_df: pd.DataFrame) -> dict:
    """
    输入：
      - 测试期订单行明细
      - 某个 window/algorithm 对应的方案（sku, stock_qty）

    输出：
      - order fill rate
      - line fill rate
      - qty fill rate（可选）
    """

    # 初始库存
    inventory = dict(zip(plan_df["sku"], plan_df["stock_qty"]))

    total_orders = 0
    full_orders = 0

    total_lines = 0
    filled_lines = 0

    total_qty = 0.0
    filled_qty = 0.0

    # 按订单顺序回放
    order_lines_test = order_lines_test.sort_values(["date", "order_id", "sku"]).copy()

    for order_id, grp in order_lines_test.groupby("order_id", sort=False):
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

        # 整单逻辑：只有全部行都能满足，才扣库存、计入填充
        if order_full:
            full_orders += 1
            for sku, req, _ in line_results:
                inventory[sku] -= req
                filled_lines += 1
                filled_qty += req
        # 整单不成功：库存不动，该订单所有行均不计入 filled

    ofr = full_orders / total_orders if total_orders > 0 else np.nan
    lfr = filled_lines / total_lines if total_lines > 0 else np.nan
    qfr = filled_qty / total_qty if total_qty > 0 else np.nan

    return {
        "basket_ofr": ofr,
        "basket_lfr": lfr,
        "basket_qfr": qfr,
        "n_orders": total_orders,
        "n_lines": total_lines,
        "total_qty": total_qty
    }


# =========================
# 6. 主流程：逐窗口逐算法跑
# =========================
def run_basket_simulation(scanner_file: str,
                          rolling_file: str,
                          plan_file: str,
                          output_window_file: str,
                          output_summary_file: str,
                          output_proxy_file: str):
    Path(output_window_file).parent.mkdir(parents=True, exist_ok=True)

    # 读订单
    order_lines = load_scanner_data(scanner_file)

    # 生成 window 区间
    windows = build_windows_from_orders(order_lines, TRAIN_DAYS, TEST_DAYS)

    # 读第一步结果
    rolling = pd.read_csv(rolling_file)
    rolling["window"] = rolling["window"].astype(int)
    rolling["scenario"] = rolling["scenario"].astype(str)

    # 读方案
    plans = load_plans(plan_file)

    # 只保留 rolling 里存在的 window/scenario
    valid_ws = rolling[["window", "scenario"]].drop_duplicates()
    plans = plans.merge(valid_ws, on=["window", "scenario"], how="inner")

    results = []

    # 循环每个 window
    for _, w in windows.iterrows():
        window_id = int(w["window"])
        test_start = pd.to_datetime(w["test_start"])
        test_end = pd.to_datetime(w["test_end"])

        test_mask = (
            (order_lines["date"] >= test_start) &
            (order_lines["date"] <= test_end)
        )
        test_orders = order_lines.loc[test_mask].copy()

        if test_orders.empty:
            continue

        # 找该窗口所有方案
        plan_w = plans[plans["window"] == window_id].copy()
        if plan_w.empty:
            continue

        for (scenario, algorithm), grp in plan_w.groupby(["scenario", "algorithm"]):
            sim = simulate_one_plan(test_orders, grp[["sku", "stock_qty"]].copy())

            results.append({
                "window": window_id,
                "scenario": scenario,
                "algorithm": algorithm,
                "test_start": test_start,
                "test_end": test_end,
                **sim
            })

    basket_window = pd.DataFrame(results)

    if basket_window.empty:
        raise ValueError("没有得到任何篮子模拟结果。请检查 plans.csv 的 window/scenario 是否和 rolling_results.csv 对齐。")

    # 存窗口级结果
    basket_window.to_csv(output_window_file, index=False, encoding="utf-8-sig")

    # 汇总
    basket_summary = (
        basket_window
        .groupby(["scenario", "algorithm"], as_index=False)
        .agg(
            basket_ofr_mean=("basket_ofr", "mean"),
            basket_lfr_mean=("basket_lfr", "mean"),
            basket_qfr_mean=("basket_qfr", "mean"),
            n_windows=("window", "count"),
            avg_orders=("n_orders", "mean"),
            avg_lines=("n_lines", "mean")
        )
        .sort_values(["scenario", "basket_lfr_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    basket_summary.to_csv(output_summary_file, index=False, encoding="utf-8-sig")

    # 与第一步 rolling_results 合并，算 proxy gap
    metric_map = {
        "greedy_nominal": "greedy_nominal_lfr",
        "dp_nominal": "dp_nominal_lfr",
        "cpsat_nominal": "cpsat_nominal_lfr",
        "greedy_robust": "greedy_robust_lfr",
        "dp_robust": "dp_robust_lfr",
        "cpsat_robust": "cpsat_robust_lfr",
    }

    proxy_rows = []
    for _, row in basket_window.iterrows():
        algo = row["algorithm"]
        if algo not in metric_map:
            continue

        col = metric_map[algo]
        temp = rolling[
            (rolling["window"] == row["window"]) &
            (rolling["scenario"] == row["scenario"])
        ]

        if temp.empty or col not in temp.columns:
            agg_lfr = np.nan
        else:
            agg_lfr = temp.iloc[0][col]

        proxy_rows.append({
            "window": row["window"],
            "scenario": row["scenario"],
            "algorithm": row["algorithm"],
            "aggregated_lfr": agg_lfr,
            "basket_lfr": row["basket_lfr"],
            "basket_ofr": row["basket_ofr"],
            "proxy_gap_lfr": agg_lfr - row["basket_lfr"] if pd.notna(agg_lfr) else np.nan
        })

    proxy_df = pd.DataFrame(proxy_rows)
    proxy_df.to_csv(output_proxy_file, index=False, encoding="utf-8-sig")

    print("篮子模拟完成。")
    print(f"窗口级结果已保存: {output_window_file}")
    print(f"汇总结果已保存: {output_summary_file}")
    print(f"代理误差结果已保存: {output_proxy_file}")


# =========================
# 7. 入口
# =========================
if __name__ == "__main__":
    run_basket_simulation(
        scanner_file=SCANNER_FILE,
        rolling_file=ROLLING_FILE,
        plan_file=PLAN_FILE,
        output_window_file=OUTPUT_WINDOW_FILE,
        output_summary_file=OUTPUT_SUMMARY_FILE,
        output_proxy_file=OUTPUT_PROXY_FILE
    )
