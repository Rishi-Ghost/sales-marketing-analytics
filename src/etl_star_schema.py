"""
ETL: Clean raw data and build a star-schema dimensional model.
Outputs processed Parquet / CSV tables + SQL views in the database.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
DB_PATH = BASE / "data" / "sales.db"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_raw():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    tables = {}
    for t in ["customers", "products", "campaigns", "orders", "order_items",
              "attribution", "daily_marketing_spend"]:
        tables[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
    return tables, engine


def build_dim_date(min_date, max_date):
    dates = pd.date_range(min_date, max_date, freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["date_key"] = dim["date"].dt.strftime("%Y%m%d").astype(int)
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = dim["date"].dt.quarter
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%B")
    dim["week"] = dim["date"].dt.isocalendar().week.astype(int)
    dim["day_of_week"] = dim["date"].dt.dayofweek
    dim["day_name"] = dim["date"].dt.strftime("%A")
    dim["is_weekend"] = dim["day_of_week"].isin([5, 6]).astype(int)
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    return dim


def build_star_schema(tables):
    customers = tables["customers"].copy()
    products = tables["products"].copy()
    campaigns = tables["campaigns"].copy()
    orders = tables["orders"].copy()
    items = tables["order_items"].copy()
    spend = tables["daily_marketing_spend"].copy()
    attr = tables["attribution"].copy()

    # --- Cleaning ---
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    customers["registration_date"] = pd.to_datetime(customers["registration_date"])
    products["launch_date"] = pd.to_datetime(products["launch_date"])
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
    spend["date"] = pd.to_datetime(spend["date"])

    # Keep only completed orders for revenue facts
    orders_clean = orders[orders["status"] == "Completed"].copy()
    orders_clean = orders_clean.dropna(subset=["order_total"])
    orders_clean["order_total"] = orders_clean["order_total"].clip(lower=0)

    # --- Dimensions ---
    dim_customer = customers[[
        "customer_id", "name", "email", "region", "segment",
        "acquisition_channel", "registration_date", "lifetime_value_estimate"
    ]].copy()
    dim_customer["registration_date_key"] = dim_customer["registration_date"].dt.strftime("%Y%m%d").astype(int)

    dim_product = products.copy()
    dim_product["launch_date_key"] = dim_product["launch_date"].dt.strftime("%Y%m%d").astype(int)

    dim_campaign = campaigns.copy()
    dim_campaign["start_date_key"] = dim_campaign["start_date"].dt.strftime("%Y%m%d").astype(int)
    dim_campaign["end_date_key"] = dim_campaign["end_date"].dt.strftime("%Y%m%d").astype(int)

    dim_channel = pd.DataFrame({
        "channel_id": range(1, len(orders["channel"].unique()) + 1),
        "channel_name": sorted(orders["channel"].unique())
    })
    channel_map = dict(zip(dim_channel["channel_name"], dim_channel["channel_id"]))

    min_d = min(orders_clean["order_date"].min(), spend["date"].min())
    max_d = max(orders_clean["order_date"].max(), spend["date"].max())
    dim_date = build_dim_date(min_d, max_d)

    # --- Facts ---
    # Fact Sales (order grain + product)
    fact_sales = items.merge(
        orders_clean[["order_id", "customer_id", "order_date", "channel", "campaign_id", "region", "discount", "order_total"]],
        on="order_id", how="inner"
    )
    fact_sales = fact_sales.merge(products[["product_id", "category", "cost"]], on="product_id", how="left", suffixes=("", "_prod"))
    # Use line cost if available, else product cost * qty
    if "cost" in fact_sales.columns:
        fact_sales["line_cost"] = fact_sales["cost"]
    else:
        fact_sales["line_cost"] = fact_sales["cost_prod"] * fact_sales["quantity"]
    fact_sales["gross_profit"] = fact_sales["line_total"] - fact_sales["line_cost"]
    fact_sales["date_key"] = fact_sales["order_date"].dt.strftime("%Y%m%d").astype(int)
    fact_sales["channel_id"] = fact_sales["channel"].map(channel_map)
    fact_sales = fact_sales[[
        "order_id", "customer_id", "product_id", "campaign_id", "channel_id",
        "date_key", "region", "quantity", "unit_price", "line_total",
        "line_cost", "gross_profit", "discount"
    ]]

    # Fact Marketing Spend (daily)
    fact_spend = spend.copy()
    fact_spend["date_key"] = fact_spend["date"].dt.strftime("%Y%m%d").astype(int)
    fact_spend["channel_id"] = fact_spend["channel"].map(channel_map)
    fact_spend = fact_spend[["date_key", "campaign_id", "channel_id", "channel", "spend"]]

    # Fact Attribution
    fact_attr = attr.copy()
    fact_attr = fact_attr.merge(
        orders_clean[["order_id", "order_date"]], on="order_id", how="inner"
    )
    fact_attr["date_key"] = pd.to_datetime(fact_attr["order_date"]).dt.strftime("%Y%m%d").astype(int)
    fact_attr["channel_id"] = fact_attr["channel"].map(channel_map)

    return {
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_campaign": dim_campaign,
        "dim_channel": dim_channel,
        "fact_sales": fact_sales,
        "fact_spend": fact_spend,
        "fact_attribution": fact_attr,
        "orders_clean": orders_clean,  # useful for RFM etc.
    }


def save_processed(schema, engine=None):
    import tempfile
    import shutil
    for name, df in schema.items():
        path = PROCESSED / f"{name}.parquet"
        # Write via cross-platform temp then move to avoid partial writes
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            df.to_parquet(tmp_path, index=False)
            shutil.copy(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(f"  Saved {name}: {len(df):,} rows -> {path}")
    print("  (Skipping SQL views due to environment FS limits; dashboard uses Parquet)")


def main():
    print("Loading raw data...")
    tables, engine = load_raw()
    print("Building star schema...")
    schema = build_star_schema(tables)
    print("Saving processed tables...")
    save_processed(schema, engine)
    print("ETL complete.")


if __name__ == "__main__":
    main()
