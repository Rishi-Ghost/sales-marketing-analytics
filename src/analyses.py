"""
Core analytics: RFM segmentation, Cohort analysis, Marketing attribution & ROI,
and helper aggregations for the dashboard.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"


def load_processed():
    data = {}
    for p in PROCESSED.glob("*.parquet"):
        data[p.stem] = pd.read_parquet(p)
    return data


def compute_rfm(orders_clean: pd.DataFrame, as_of=None, quantiles=5):
    """
    Classic RFM segmentation.
    Recency = days since last order, Frequency = # orders, Monetary = total spend.
    """
    if as_of is None:
        as_of = orders_clean["order_date"].max()
    as_of = pd.to_datetime(as_of)

    rfm = orders_clean.groupby("customer_id").agg(
        recency=("order_date", lambda x: (as_of - pd.to_datetime(x).max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_total", "sum"),
    ).reset_index()

    # Scores (1 = worst, quantiles = best). Handle ties with rank.
    rfm["R_score"] = pd.qcut(rfm["recency"], q=quantiles, labels=False, duplicates="drop")
    rfm["R_score"] = quantiles - rfm["R_score"]  # invert: low recency = high score
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=quantiles, labels=False, duplicates="drop") + 1
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=quantiles, labels=False, duplicates="drop") + 1

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal"
        if r >= 4 and f <= 2:
            return "Promising / New"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r <= 2 and f <= 2:
            return "Lost / Hibernating"
        if m >= 4:
            return "Big Spenders"
        return "Need Attention"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def compute_cohorts(orders_clean: pd.DataFrame, customers: pd.DataFrame):
    """
    Monthly acquisition cohort retention (orders) and revenue.
    """
    orders = orders_clean.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["order_month"] = orders["order_date"].dt.to_period("M")

    cust = customers[["customer_id", "registration_date"]].copy()
    cust["registration_date"] = pd.to_datetime(cust["registration_date"])
    cust["cohort"] = cust["registration_date"].dt.to_period("M")

    df = orders.merge(cust, on="customer_id", how="left")
    df = df.dropna(subset=["cohort"])

    # Period number relative to cohort
    df["period_number"] = (df["order_month"] - df["cohort"]).apply(lambda x: x.n)

    # Cohort size
    cohort_sizes = df.groupby("cohort")["customer_id"].nunique()

    # Retention matrix (unique customers per cohort-period)
    retention = df.groupby(["cohort", "period_number"])["customer_id"].nunique().unstack(fill_value=0)
    retention = retention.divide(cohort_sizes, axis=0)

    # Revenue matrix
    revenue = df.groupby(["cohort", "period_number"])["order_total"].sum().unstack(fill_value=0)

    return {
        "retention": retention,
        "revenue": revenue,
        "cohort_sizes": cohort_sizes,
        "raw": df,
    }


def channel_roi(fact_sales: pd.DataFrame, fact_spend: pd.DataFrame, dim_channel: pd.DataFrame):
    """Revenue, spend, ROI by channel."""
    rev = fact_sales.merge(dim_channel, on="channel_id", how="left")
    rev_by_ch = rev.groupby("channel_name").agg(
        revenue=("line_total", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        gross_profit=("gross_profit", "sum"),
    ).reset_index()

    spend = fact_spend.merge(dim_channel, on="channel_id", how="left")
    spend_by_ch = spend.groupby("channel_name")["spend"].sum().reset_index()

    roi = rev_by_ch.merge(spend_by_ch, on="channel_name", how="left").fillna({"spend": 0})
    roi["roi"] = np.where(roi["spend"] > 0, (roi["revenue"] - roi["spend"]) / roi["spend"], np.nan)
    roi["roas"] = np.where(roi["spend"] > 0, roi["revenue"] / roi["spend"], np.nan)
    roi = roi.sort_values("revenue", ascending=False)
    return roi


def attribution_summary(fact_attr: pd.DataFrame, dim_channel: pd.DataFrame):
    """Simple last-touch + assist attribution revenue by channel."""
    attr = fact_attr.merge(dim_channel, on="channel_id", how="left")
    summary = attr.groupby(["channel_name", "attribution_type"]).agg(
        attributed_revenue=("revenue", "sum"),
        events=("order_id", "count"),
    ).reset_index()
    pivot = summary.pivot_table(
        index="channel_name",
        columns="attribution_type",
        values="attributed_revenue",
        fill_value=0,
    ).reset_index()
    return summary, pivot


def daily_revenue_series(fact_sales: pd.DataFrame, dim_date: pd.DataFrame):
    """Daily revenue time series for forecasting."""
    fs = fact_sales.merge(dim_date[["date_key", "date"]], on="date_key", how="left")
    daily = fs.groupby("date").agg(
        revenue=("line_total", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        gross_profit=("gross_profit", "sum"),
    ).reset_index().sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def segment_growth(orders_clean: pd.DataFrame, customers: pd.DataFrame, dim_date=None):
    """Monthly revenue & customer count by segment – growth/decline signals."""
    o = orders_clean.merge(
        customers[["customer_id", "segment"]], on="customer_id", how="left"
    )
    o["order_date"] = pd.to_datetime(o["order_date"])
    o["year_month"] = o["order_date"].dt.to_period("M").astype(str)
    monthly = o.groupby(["year_month", "segment"]).agg(
        revenue=("order_total", "sum"),
        customers=("customer_id", "nunique"),
        orders=("order_id", "nunique"),
    ).reset_index()
    return monthly


if __name__ == "__main__":
    data = load_processed()
    print("Loaded:", list(data.keys()))
    rfm = compute_rfm(data["orders_clean"])
    print("\nRFM segment counts:")
    print(rfm["segment"].value_counts())
    cohorts = compute_cohorts(data["orders_clean"], data["dim_customer"])
    print("\nCohort retention shape:", cohorts["retention"].shape)
    roi = channel_roi(data["fact_sales"], data["fact_spend"], data["dim_channel"])
    print("\nChannel ROI sample:")
    print(roi.head())
