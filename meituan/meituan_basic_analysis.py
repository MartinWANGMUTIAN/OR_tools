from __future__ import annotations

import ast
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "meituan"
OUTPUT_DIR = DATA_DIR / "analysis_output"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gbk", header=1, low_memory=False)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def parse_time(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"0": pd.NA, "nan": pd.NA, "NaT": pd.NA, "": pd.NA})
    return pd.to_datetime(text, errors="coerce")


def duration_minutes(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds().div(60)


def series_summary(series: pd.Series) -> dict[str, float]:
    valid = series.dropna()
    return {
        "count": int(valid.count()),
        "mean": round(valid.mean(), 2),
        "median": round(valid.median(), 2),
        "p90": round(valid.quantile(0.9), 2),
        "max": round(valid.max(), 2),
    }


def parse_list_len(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def save_order_volume_plot(df: pd.DataFrame) -> None:
    hourly = df.groupby("order_hour").size()
    weekend = (
        df.groupby(["order_hour", "is_weekend"])["order_id"]
        .count()
        .unstack(fill_value=0)
        .rename(columns={0: "weekday", 1: "weekend"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hourly.plot(kind="bar", color="#2E86AB", ax=axes[0])
    axes[0].set_title("Orders by Hour")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Order Count")

    weekend.plot(kind="bar", stacked=True, ax=axes[1], color=["#7FB069", "#D95D39"])
    axes[1].set_title("Hourly Orders by Weekend Flag")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Order Count")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "order_volume_by_hour.png", dpi=200)
    plt.close(fig)


def save_duration_plot(df: pd.DataFrame) -> None:
    duration_cols = [
        "assign_to_grab_mins",
        "grab_to_fetch_mins",
        "fetch_to_arrive_mins",
        "push_to_arrive_mins",
    ]
    plot_df = df[duration_cols].copy()
    upper_bounds = plot_df.quantile(0.99)
    for col in duration_cols:
        plot_df[col] = plot_df[col].clip(lower=0, upper=upper_bounds[col])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([plot_df[col].dropna() for col in duration_cols], labels=duration_cols)
    ax.set_title("Distribution of Core Durations")
    ax.set_ylabel("Minutes")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delivery_duration_boxplot.png", dpi=200)
    plt.close(fig)


def save_location_plot(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        sample["sender_lng"],
        sample["sender_lat"],
        s=8,
        alpha=0.35,
        label="pickup",
        color="#457B9D",
    )
    ax.scatter(
        sample["recipient_lng"],
        sample["recipient_lat"],
        s=8,
        alpha=0.35,
        label="dropoff",
        color="#E76F51",
    )
    ax.set_title("Pickup and Dropoff Location Sample")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pickup_dropoff_scatter.png", dpi=200)
    plt.close(fig)


def save_wave_plot(wave_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(wave_df["wave_duration_mins"].dropna(), bins=40, color="#6A994E", alpha=0.85)
    axes[0].set_title("Wave Duration Distribution")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("Count")

    axes[1].hist(wave_df["wave_order_count"].dropna(), bins=30, color="#BC4749", alpha=0.85)
    axes[1].set_title("Orders per Wave")
    axes[1].set_xlabel("Order Count")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wave_metrics.png", dpi=200)
    plt.close(fig)


def save_workload_plot(dispatch_rider_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dispatch_rider_df["current_order_count"], bins=30, color="#8338EC", alpha=0.8)
    ax.set_title("Current Orders per Rider at Dispatch Time")
    ax.set_xlabel("Current Order Count")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "courier_workload_hist.png", dpi=200)
    plt.close(fig)


def build_summary(
    orders_df: pd.DataFrame, wave_df: pd.DataFrame, dispatch_rider_df: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        {"metric": "total_orders", "value": int(len(orders_df))},
        {"metric": "unique_couriers", "value": int(orders_df["courier_id"].nunique())},
        {"metric": "unique_pois", "value": int(orders_df["poi_id"].nunique())},
        {"metric": "grabbed_rate", "value": round(orders_df["is_courier_grabbed"].mean(), 4)},
        {"metric": "prebook_rate", "value": round(orders_df["is_prebook"].mean(), 4)},
        {"metric": "weekend_rate", "value": round(orders_df["is_weekend"].mean(), 4)},
        {"metric": "dispatch_events", "value": int(len(dispatch_rider_df))},
        {"metric": "wave_records", "value": int(len(wave_df))},
        {
            "metric": "avg_orders_per_wave",
            "value": round(wave_df["wave_order_count"].mean(), 2),
        },
        {
            "metric": "avg_current_orders_per_rider_dispatch",
            "value": round(dispatch_rider_df["current_order_count"].mean(), 2),
        },
    ]

    duration_map = {
        "assign_to_grab_mins": "assign_to_grab",
        "grab_to_fetch_mins": "grab_to_fetch",
        "fetch_to_arrive_mins": "fetch_to_arrive",
        "push_to_arrive_mins": "push_to_arrive",
    }
    for col, prefix in duration_map.items():
        stats = series_summary(orders_df[col])
        for key, value in stats.items():
            rows.append({"metric": f"{prefix}_{key}", "value": value})

    return pd.DataFrame(rows)


def write_highlights(
    orders_df: pd.DataFrame, wave_df: pd.DataFrame, dispatch_rider_df: pd.DataFrame
) -> None:
    peak_hour = int(orders_df.groupby("order_hour").size().idxmax())
    peak_orders = int(orders_df.groupby("order_hour").size().max())
    busiest_da = int(orders_df.groupby("da_id").size().idxmax())
    busiest_da_orders = int(orders_df.groupby("da_id").size().max())

    lines = [
        "# Meituan Basic Analysis",
        "",
        f"- Total orders: {len(orders_df):,}",
        f"- Courier acceptance rate: {orders_df['is_courier_grabbed'].mean():.2%}",
        f"- Prebook rate: {orders_df['is_prebook'].mean():.2%}",
        f"- Peak order hour: {peak_hour}:00 with {peak_orders:,} orders",
        f"- Busiest business area id: {busiest_da} with {busiest_da_orders:,} orders",
        f"- Median assign to grab time: {orders_df['assign_to_grab_mins'].median():.2f} minutes",
        f"- Median grab to fetch time: {orders_df['grab_to_fetch_mins'].median():.2f} minutes",
        f"- Median fetch to arrive time: {orders_df['fetch_to_arrive_mins'].median():.2f} minutes",
        f"- Median push to arrive time: {orders_df['push_to_arrive_mins'].median():.2f} minutes",
        f"- Average wave duration: {wave_df['wave_duration_mins'].mean():.2f} minutes",
        f"- Average orders per wave: {wave_df['wave_order_count'].mean():.2f}",
        (
            "- Average rider current workload at dispatch: "
            f"{dispatch_rider_df['current_order_count'].mean():.2f} orders"
        ),
    ]
    (OUTPUT_DIR / "analysis_highlights.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders_df = load_csv(DATA_DIR / "processed_all_waybill_info_meituan.csv")
    wave_df = load_csv(DATA_DIR / "processed_courier_wave_info_meituan.csv")
    dispatch_rider_df = load_csv(DATA_DIR / "processed_dispatch_rider_meituan.csv")
    dispatch_waybill_df = load_csv(DATA_DIR / "processed_dispatch_waybill_meituan.csv")

    orders_df["dt"] = pd.to_datetime(orders_df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    numeric_cols = [
        "order_id",
        "waybill_id",
        "courier_id",
        "da_id",
        "is_courier_grabbed",
        "is_weekend",
        "is_prebook",
        "poi_id",
        "sender_lng",
        "sender_lat",
        "recipient_lng",
        "recipient_lat",
    ]
    for col in numeric_cols:
        orders_df[col] = pd.to_numeric(orders_df[col], errors="coerce")

    time_cols = [
        "estimate_arrived_time",
        "dispatch_time",
        "grab_time",
        "fetch_time",
        "estimate_meal_prepare_time",
        "arrive_time",
        "order_push_time",
        "platform_order_time",
    ]
    for col in time_cols:
        orders_df[col] = parse_time(orders_df[col])

    orders_df["order_hour"] = orders_df["dispatch_time"].dt.hour
    orders_df["assign_to_grab_mins"] = duration_minutes(orders_df["grab_time"], orders_df["dispatch_time"])
    orders_df["grab_to_fetch_mins"] = duration_minutes(orders_df["fetch_time"], orders_df["grab_time"])
    orders_df["fetch_to_arrive_mins"] = duration_minutes(orders_df["arrive_time"], orders_df["fetch_time"])
    orders_df["push_to_arrive_mins"] = duration_minutes(orders_df["arrive_time"], orders_df["order_push_time"])
    orders_df["promise_delta_mins"] = duration_minutes(
        orders_df["arrive_time"], orders_df["estimate_arrived_time"]
    )

    wave_df["wave_start_time"] = parse_time(wave_df["wave_start_time"])
    wave_df["wave_end_time"] = parse_time(wave_df["wave_end_time"])
    wave_df["wave_duration_mins"] = duration_minutes(wave_df["wave_end_time"], wave_df["wave_start_time"])
    wave_df["wave_order_count"] = wave_df["order_ids"].apply(parse_list_len)

    dispatch_rider_df["dispatch_time"] = parse_time(dispatch_rider_df["dispatch_time"])
    dispatch_rider_df["current_order_count"] = dispatch_rider_df["courier_waybills"].apply(parse_list_len)

    dispatch_waybill_df["dispatch_time"] = parse_time(dispatch_waybill_df["dispatch_time"])

    summary_df = build_summary(orders_df, wave_df, dispatch_rider_df)
    summary_df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    hourly_summary = (
        orders_df.groupby("order_hour")
        .agg(
            order_count=("order_id", "count"),
            acceptance_rate=("is_courier_grabbed", "mean"),
            median_push_to_arrive_mins=("push_to_arrive_mins", "median"),
        )
        .reset_index()
    )
    hourly_summary.to_csv(OUTPUT_DIR / "hourly_summary.csv", index=False)

    courier_summary = (
        dispatch_rider_df.groupby("courier_id")
        .agg(
            dispatch_records=("courier_id", "size"),
            avg_current_order_count=("current_order_count", "mean"),
        )
        .sort_values("dispatch_records", ascending=False)
        .reset_index()
    )
    courier_summary.to_csv(OUTPUT_DIR / "courier_summary.csv", index=False)

    save_order_volume_plot(orders_df.dropna(subset=["order_hour"]))
    save_duration_plot(orders_df)
    save_location_plot(orders_df.dropna(subset=["sender_lng", "sender_lat", "recipient_lng", "recipient_lat"]))
    save_wave_plot(wave_df)
    save_workload_plot(dispatch_rider_df)
    write_highlights(orders_df, wave_df, dispatch_rider_df)


if __name__ == "__main__":
    main()
