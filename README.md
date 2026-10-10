# Real-Time Sales / Marketing Performance Analytics + Forecasting System

A complete, end-to-end analytics platform demonstrating modern data engineering and data science skills:

- **Multi-source ingestion** (CSV + SQLite)
- **Dimensional modeling** (star schema)
- **Advanced analytics** (RFM segmentation, cohort retention, marketing attribution)
- **Time-series forecasting** (Prophet with scenario planning)
- **Interactive dashboard** answering key business questions

## Business Questions Answered

| Question | Dashboard Page |
|----------|----------------|
| Which channels give the best ROI? | Channels & ROI |
| Which customer segments are growing / declining? | Segments & Growth |
| What will next quarter’s revenue look like under different scenarios? | Forecasting |
| How do acquisition cohorts retain over time? | Cohort Retention |

## Architecture

```
data/raw/          ← CSVs + SQLite (source of truth)
data/processed/    ← Star-schema Parquet tables
src/
  generate_data.py     # Synthetic data generator (realistic seasonality + growth)
  etl_star_schema.py   # Clean + dimensional model
  analyses.py          # RFM, cohorts, ROI, attribution
  forecasting.py       # Prophet + scenarios
app/
  dashboard.py         # Streamlit live dashboard
models/                # Saved forecasts
```

### Star Schema

**Dimensions:** `dim_date`, `dim_customer`, `dim_product`, `dim_campaign`, `dim_channel`  
**Facts:** `fact_sales`, `fact_spend`, `fact_attribution`

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-generate data
python src/generate_data.py

# 3. Run ETL
python src/etl_star_schema.py

# 4. (Optional) Refresh analyses / forecast
python src/analyses.py
python src/forecasting.py

# 5. Launch dashboard
streamlit run app/dashboard.py
```

## Key Metrics & Models

- **RFM**: Recency / Frequency / Monetary scoring → Champions, Loyal, At Risk, etc.
- **Cohorts**: Monthly acquisition cohorts with retention heat-map
- **Attribution**: Last-touch + assist revenue by channel
- **ROI / ROAS**: Revenue vs marketing spend by channel
- **Forecast**: Facebook Prophet (multiplicative seasonality) + 4 demand scenarios for next ~90 days

## Sample Insights (from generated data)

- Organic Search typically shows highest ROAS (low cost, high volume)
- Paid channels have higher absolute spend but solid returns
- Clear Champions vs At-Risk segments visible in RFM
- Strong seasonal peaks (Q4) captured by the model

## Extending the System

- Replace synthetic generator with real CSVs / warehouse extracts
- Add Airflow / Prefect for scheduled ETL + retrain
- Swap Streamlit for Power BI / Tableau / Looker by pointing at the Parquet / SQL layer
- Add LSTM or neural forecasting if longer history or more features available

---

Built as a portfolio-ready demonstration of messy real-world data → reliable forecasts → strategy-influencing insights.
