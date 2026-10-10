"""
Generate realistic synthetic sales & marketing data for the analytics system.
Produces:
- CSV files (customers, products, campaigns)
- SQLite database with transactional data
- Marketing spend / attribution events
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Paths
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE / "data" / "sales.db"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Config
N_CUSTOMERS = 5000
N_PRODUCTS = 120
N_CAMPAIGNS = 45
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2026, 9, 15)  # ~ current project time
CHANNELS = ["Organic Search", "Paid Search", "Social Media", "Email", "Affiliate", "Direct", "Display Ads"]
REGIONS = ["North America", "Europe", "APAC", "LATAM"]
SEGMENTS = ["Enterprise", "SMB", "Consumer"]
PRODUCT_CATEGORIES = ["Software", "Hardware", "Services", "Accessories", "Subscriptions"]


def random_dates(n, start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return [start + timedelta(days=random.randint(0, delta)) for _ in range(n)]


def generate_customers(n=N_CUSTOMERS):
    customers = []
    for i in range(1, n + 1):
        reg_date = random_dates(1)[0]
        customers.append({
            "customer_id": i,
            "name": f"Customer_{i:04d}",
            "email": f"customer{i}@example.com",
            "region": random.choice(REGIONS),
            "segment": random.choice(SEGMENTS),
            "acquisition_channel": random.choice(CHANNELS),
            "registration_date": reg_date.date(),
            "lifetime_value_estimate": round(np.random.lognormal(6.5, 1.1), 2),
        })
    return pd.DataFrame(customers)


def generate_products(n=N_PRODUCTS):
    products = []
    for i in range(1, n + 1):
        cat = random.choice(PRODUCT_CATEGORIES)
        base_price = {
            "Software": (50, 800),
            "Hardware": (80, 2500),
            "Services": (100, 5000),
            "Accessories": (10, 150),
            "Subscriptions": (9, 299),
        }[cat]
        price = round(random.uniform(*base_price), 2)
        products.append({
            "product_id": i,
            "product_name": f"{cat}_{i:03d}",
            "category": cat,
            "unit_price": price,
            "cost": round(price * random.uniform(0.35, 0.65), 2),
            "launch_date": random_dates(1, START_DATE, START_DATE + timedelta(days=400))[0].date(),
        })
    return pd.DataFrame(products)


def generate_campaigns(n=N_CAMPAIGNS):
    campaigns = []
    for i in range(1, n + 1):
        start = random_dates(1, START_DATE, END_DATE - timedelta(days=60))[0]
        duration = random.randint(14, 90)
        end = min(start + timedelta(days=duration), END_DATE)
        channel = random.choice(CHANNELS)
        budget = round(random.uniform(2000, 80000), 2)
        campaigns.append({
            "campaign_id": i,
            "campaign_name": f"{channel.replace(' ', '')}_Camp_{i:02d}",
            "channel": channel,
            "start_date": start.date(),
            "end_date": end.date(),
            "budget": budget,
            "target_segment": random.choice(SEGMENTS + ["All"]),
        })
    return pd.DataFrame(campaigns)


def generate_transactions(customers, products, campaigns, n_orders=35000):
    """Generate order header + line items with realistic seasonality & trends."""
    orders = []
    order_items = []
    attribution = []  # simple last-touch + multi-touch proxy

    # Seasonal multipliers (monthly)
    seasonality = {
        1: 0.85, 2: 0.9, 3: 1.05, 4: 1.0, 5: 1.1, 6: 1.15,
        7: 1.05, 8: 0.95, 9: 1.1, 10: 1.2, 11: 1.35, 12: 1.5
    }

    # Growth trend: ~18% YoY base
    def trend_factor(d):
        years = (d - START_DATE).days / 365.25
        return 1.0 + 0.18 * years + 0.03 * np.sin(2 * np.pi * years)

    order_id = 1
    for _ in range(n_orders):
        cust = customers.sample(1).iloc[0]
        # Prefer later dates slightly (growth)
        days_offset = int(np.random.beta(2, 1.2) * (END_DATE - START_DATE).days)
        order_date = START_DATE + timedelta(days=days_offset)
        if order_date > END_DATE:
            order_date = END_DATE

        # Channel preference influenced by acquisition + campaign
        channel = random.choices(
            CHANNELS,
            weights=[0.22, 0.18, 0.15, 0.12, 0.08, 0.15, 0.10]
        )[0]

        # Pick active campaign for that channel if any
        active_camps = campaigns[
            (campaigns["channel"] == channel) &
            (campaigns["start_date"] <= order_date.date()) &
            (campaigns["end_date"] >= order_date.date())
        ]
        campaign_id = int(active_camps.sample(1)["campaign_id"].iloc[0]) if len(active_camps) > 0 and random.random() < 0.55 else None

        # Number of line items
        n_items = random.choices([1, 2, 3, 4], weights=[0.55, 0.25, 0.15, 0.05])[0]
        order_total = 0.0
        for _ in range(n_items):
            prod = products.sample(1).iloc[0]
            qty = random.choices([1, 2, 3, 5], weights=[0.7, 0.2, 0.07, 0.03])[0]
            # Apply seasonality + trend + some noise
            price_mult = seasonality[order_date.month] * trend_factor(order_date) * random.uniform(0.92, 1.08)
            unit_price = round(prod["unit_price"] * price_mult, 2)
            line_total = round(unit_price * qty, 2)
            order_total += line_total

            order_items.append({
                "order_id": order_id,
                "product_id": int(prod["product_id"]),
                "quantity": qty,
                "unit_price": unit_price,
                "line_total": line_total,
                "cost": round(prod["cost"] * qty, 2),
            })

        # Discount occasionally
        discount = 0.0
        if random.random() < 0.18:
            discount = round(order_total * random.uniform(0.05, 0.25), 2)
            order_total = round(order_total - discount, 2)

        orders.append({
            "order_id": order_id,
            "customer_id": int(cust["customer_id"]),
            "order_date": order_date.date(),
            "channel": channel,
            "campaign_id": campaign_id,
            "region": cust["region"],
            "order_total": order_total,
            "discount": discount,
            "status": random.choices(["Completed", "Returned", "Cancelled"], weights=[0.92, 0.05, 0.03])[0],
        })

        # Simple attribution event (last-touch + assist)
        if campaign_id:
            attribution.append({
                "order_id": order_id,
                "campaign_id": campaign_id,
                "channel": channel,
                "attribution_type": "last_touch",
                "attribution_weight": 1.0,
                "revenue": order_total,
            })
            # occasional multi-touch assist
            if random.random() < 0.3:
                other_ch = random.choice([c for c in CHANNELS if c != channel])
                attribution.append({
                    "order_id": order_id,
                    "campaign_id": None,
                    "channel": other_ch,
                    "attribution_type": "assist",
                    "attribution_weight": 0.3,
                    "revenue": round(order_total * 0.3, 2),
                })

        order_id += 1

    return pd.DataFrame(orders), pd.DataFrame(order_items), pd.DataFrame(attribution)


def generate_daily_marketing_spend(campaigns):
    """Daily spend per campaign (simplified)."""
    rows = []
    for _, camp in campaigns.iterrows():
        start = pd.to_datetime(camp["start_date"])
        end = pd.to_datetime(camp["end_date"])
        days = (end - start).days + 1
        if days <= 0:
            continue
        daily_base = camp["budget"] / days
        for d in pd.date_range(start, end, freq="D"):
            # Spend variation
            spend = round(daily_base * random.uniform(0.7, 1.4), 2)
            rows.append({
                "date": d.date(),
                "campaign_id": int(camp["campaign_id"]),
                "channel": camp["channel"],
                "spend": spend,
            })
    return pd.DataFrame(rows)


def main():
    print("Generating customers...")
    customers = generate_customers()
    print("Generating products...")
    products = generate_products()
    print("Generating campaigns...")
    campaigns = generate_campaigns()
    print("Generating transactions (this may take a minute)...")
    orders, order_items, attribution = generate_transactions(customers, products, campaigns)
    print("Generating daily marketing spend...")
    daily_spend = generate_daily_marketing_spend(campaigns)

    # Save CSVs
    customers.to_csv(RAW / "customers.csv", index=False)
    products.to_csv(RAW / "products.csv", index=False)
    campaigns.to_csv(RAW / "campaigns.csv", index=False)
    orders.to_csv(RAW / "orders.csv", index=False)
    order_items.to_csv(RAW / "order_items.csv", index=False)
    attribution.to_csv(RAW / "attribution.csv", index=False)
    daily_spend.to_csv(RAW / "daily_marketing_spend.csv", index=False)

    print(f"CSVs written to {RAW}")

    # SQLite database
    engine = create_engine(f"sqlite:///{DB_PATH}")
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    products.to_sql("products", engine, if_exists="replace", index=False)
    campaigns.to_sql("campaigns", engine, if_exists="replace", index=False)
    orders.to_sql("orders", engine, if_exists="replace", index=False)
    order_items.to_sql("order_items", engine, if_exists="replace", index=False)
    attribution.to_sql("attribution", engine, if_exists="replace", index=False)
    daily_spend.to_sql("daily_marketing_spend", engine, if_exists="replace", index=False)

    # Quick indexes
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_cust ON orders(customer_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_spend_date ON daily_marketing_spend(date)"))

    print(f"SQLite DB created at {DB_PATH}")
    print("\nSummary:")
    print(f"  Customers : {len(customers):,}")
    print(f"  Products  : {len(products):,}")
    print(f"  Campaigns : {len(campaigns):,}")
    print(f"  Orders    : {len(orders):,}")
    print(f"  Line items: {len(order_items):,}")
    print(f"  Spend days: {len(daily_spend):,}")
    print("Done.")


if __name__ == "__main__":
    main()
