"""
Live Sales & Marketing Performance Analytics Dashboard
Run: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyses import (
    load_processed, compute_rfm, compute_cohorts, channel_roi,
    attribution_summary, daily_revenue_series, segment_growth
)
from forecasting import full_forecast_pipeline

st.set_page_config(
    page_title="Sales & Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware CSS: works in both Light and Dark mode (no forced colors)
st.markdown("""
<style>
    /* Metric cards – subtle background that works in both themes */
    [data-testid="stMetric"] {
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(128, 128, 128, 0.08);
    }
    /* Do NOT force text colors – let Streamlit theme control light/dark text */
</style>
""", unsafe_allow_html=True)


def get_plotly_layout():
    """Return Plotly layout that matches the current Streamlit theme (light or dark)."""
    base = "light"
    try:
        # Streamlit 1.33+ 
        if hasattr(st, "context") and hasattr(st.context, "theme"):
            base = getattr(st.context.theme, "type", None) or "light"
        else:
            base = st.get_option("theme.base") or "light"
    except Exception:
        base = "light"

    is_dark = str(base).lower() == "dark"
    if is_dark:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa", size=12),
            title_font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.2)"),
        )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a1a", size=12),
        title_font=dict(color="#1a1a1a"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)"),
    )


@st.cache_data(ttl=3600)
def get_data():
    data = load_processed()
    daily = daily_revenue_series(data["fact_sales"], data["dim_date"])
    rfm = compute_rfm(data["orders_clean"])
    cohorts = compute_cohorts(data["orders_clean"], data["dim_customer"])
    roi = channel_roi(data["fact_sales"], data["fact_spend"], data["dim_channel"])
    attr_sum, attr_pivot = attribution_summary(data["fact_attribution"], data["dim_channel"])
    seg_growth = segment_growth(data["orders_clean"], data["dim_customer"])
    fc, summary, _, last_date = full_forecast_pipeline(daily)
    return {
        "data": data,
        "daily": daily,
        "rfm": rfm,
        "cohorts": cohorts,
        "roi": roi,
        "attr_sum": attr_sum,
        "attr_pivot": attr_pivot,
        "seg_growth": seg_growth,
        "forecast": fc,
        "forecast_summary": summary,
        "last_date": last_date,
    }


def kpi_row(daily, roi, rfm):
    total_rev = daily["revenue"].sum()
    total_orders = daily["orders"].sum()
    avg_order = total_rev / total_orders if total_orders else 0
    best_ch = roi.iloc[0]["channel_name"] if len(roi) else "N/A"
    best_roi = roi.iloc[0]["roas"] if len(roi) and pd.notna(roi.iloc[0]["roas"]) else 0
    champions = (rfm["segment"] == "Champions").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"${total_rev/1e6:.1f}M")
    c2.metric("Total Orders", f"{total_orders:,.0f}")
    c3.metric("Avg Order Value", f"${avg_order:,.0f}")
    c4.metric("Best Channel (ROAS)", f"{best_ch}", f"{best_roi:.1f}x")
    c5.metric("Champion Customers", f"{champions:,}")


def page_overview(ctx):
    st.header("📈 Executive Overview")
    kpi_row(ctx["daily"], ctx["roi"], ctx["rfm"])

    st.subheader("Revenue Trend")
    daily = ctx["daily"].copy()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month").agg(revenue=("revenue", "sum"), orders=("orders", "sum")).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["revenue"], name="Revenue", marker_color="#2563eb"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["orders"], name="Orders", line=dict(color="#f59e0b", width=2)), secondary_y=True)
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"), **get_plotly_layout())
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Channel")
        fig2 = px.bar(ctx["roi"], x="channel_name", y="revenue", color="roas",
                      color_continuous_scale="Blues", labels={"channel_name": "Channel", "revenue": "Revenue"})
        fig2.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), **get_plotly_layout())
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Customer Segments (RFM)")
        seg_counts = ctx["rfm"]["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig3 = px.pie(seg_counts, names="segment", values="count", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), **get_plotly_layout())
        st.plotly_chart(fig3, use_container_width=True)


def page_channels(ctx):
    st.header("📡 Channel Performance & ROI")
    st.markdown("**Which channels give the best ROI?**")

    roi = ctx["roi"].copy()
    roi["roi_pct"] = roi["roi"] * 100

    display_roi = roi[["channel_name", "revenue", "spend", "gross_profit", "orders", "roas", "roi"]].copy()
    st.dataframe(
        display_roi.style.format({
            "revenue": "${:,.0f}",
            "spend": "${:,.0f}",
            "gross_profit": "${:,.0f}",
            "orders": "{:,.0f}",
            "roas": "{:.1f}x",
            "roi": "{:.0%}",
        }, na_rep="—"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(roi, x="spend", y="revenue", size="orders", color="channel_name",
                         hover_name="channel_name", labels={"spend": "Marketing Spend", "revenue": "Revenue"},
                         title="Spend vs Revenue (bubble = orders)")
        fig.update_layout(height=400, **get_plotly_layout())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(roi.sort_values("roas", ascending=True), y="channel_name", x="roas",
                      orientation="h", title="Return on Ad Spend (ROAS)",
                      color="roas", color_continuous_scale="Tealgrn")
        fig2.update_layout(height=400, **get_plotly_layout())
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Attribution (Last-touch vs Assist)")
    if not ctx["attr_pivot"].empty:
        pivot = ctx["attr_pivot"].copy()
        num_cols = pivot.select_dtypes(include="number").columns
        fmt = {c: "{:,.0f}" for c in num_cols}
        st.dataframe(pivot.style.format(fmt), use_container_width=True)


def page_segments(ctx):
    st.header("👥 Customer Segments & Growth")
    st.markdown("**Which customer segments are growing / declining?**")

    rfm = ctx["rfm"]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("RFM Segment Distribution")
        st.dataframe(
            rfm.groupby("segment").agg(
                customers=("customer_id", "count"),
                avg_monetary=("monetary", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_recency=("recency", "mean"),
            ).round(1).sort_values("customers", ascending=False),
            use_container_width=True,
        )
    with col2:
        fig = px.scatter(rfm, x="frequency", y="monetary", color="segment",
                         size="recency", hover_data=["customer_id"],
                         title="RFM Map (size = recency days)",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=400, **get_plotly_layout())
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Revenue by Segment")
    sg = ctx["seg_growth"].copy()
    fig2 = px.area(sg, x="year_month", y="revenue", color="segment",
                   title="Segment Revenue Over Time", groupnorm="")
    fig2.update_layout(height=400, xaxis_tickangle=-45, **get_plotly_layout())
    st.plotly_chart(fig2, use_container_width=True)

    # Growth rates last 3 months vs prior 3
    st.subheader("Recent Growth / Decline Signals")
    recent = sg[sg["year_month"] >= "2026-04"]
    if len(recent) > 0:
        pivot = recent.pivot_table(index="segment", columns="year_month", values="revenue", aggfunc="sum").fillna(0)
        st.dataframe(pivot.style.format("${:,.0f}", na_rep="—"), use_container_width=True)


def page_cohorts(ctx):
    st.header("🔁 Cohort Analysis")
    ret = ctx["cohorts"]["retention"]
    # Show recent cohorts only for readability
    recent_cohorts = ret.tail(18)
    st.markdown("Retention rate by acquisition cohort (rows) and months since acquisition (columns)")
    fig = px.imshow(
        recent_cohorts.values,
        labels=dict(x="Months since acquisition", y="Cohort", color="Retention"),
        x=[str(c) for c in recent_cohorts.columns],
        y=[str(i) for i in recent_cohorts.index],
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), **get_plotly_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cohort Sizes")
    sizes = ctx["cohorts"]["cohort_sizes"].tail(18).reset_index()
    sizes.columns = ["cohort", "customers"]
    # Period objects are not JSON-serializable for Plotly — convert to string
    sizes["cohort"] = sizes["cohort"].astype(str)
    fig2 = px.bar(sizes, x="cohort", y="customers", title="New Customers per Cohort Month")
    fig2.update_layout(height=300, xaxis_tickangle=-45, **get_plotly_layout())
    st.plotly_chart(fig2, use_container_width=True)


def page_forecast(ctx):
    st.header("🔮 Revenue Forecasting & Scenarios")
    st.markdown("**What will next quarter’s revenue look like under different scenarios?**")

    summary = ctx["forecast_summary"]
    cols = st.columns(len(summary))
    for i, (name, vals) in enumerate(summary.items()):
        cols[i].metric(name, f"${vals['revenue']/1e6:.2f}M", f"{vals['multiplier']:.0%} of base")

    fc = ctx["forecast"].copy()
    last = ctx["last_date"]
    # Plot last 180 days history + forecast
    plot_df = fc[fc["ds"] >= (last - pd.Timedelta(days=180))].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["ds"], y=plot_df["actual"], name="Actual", line=dict(color="#1e40af", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["ds"], y=plot_df["forecast_base"], name="Forecast (base)", line=dict(color="#059669", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=plot_df["ds"], y=plot_df["upper"], fill=None, mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["ds"], y=plot_df["lower"], fill="tonexty", mode="lines", line=dict(width=0),
        name="80% interval", fillcolor="rgba(5,150,105,0.15)"
    ))

    # Scenario lines (future only)
    future = plot_df[plot_df["ds"] > last]
    for col, color, label in [
        ("forecast_optimistic", "#16a34a", "Optimistic"),
        ("forecast_conservative", "#d97706", "Conservative"),
        ("forecast_severe", "#dc2626", "Severe"),
    ]:
        if col in future.columns:
            fig.add_trace(go.Scatter(x=future["ds"], y=future[col], name=label,
                                     line=dict(color=color, width=1.5, dash="dot")))

    fig.update_layout(
        height=480, title="Revenue: Actual + 90-day Forecast with Scenarios",
        xaxis_title="Date", yaxis_title="Daily Revenue ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        **get_plotly_layout(),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Scenarios apply demand multipliers to the Prophet baseline forecast. "
        "Use these for planning capacity, marketing budget, and risk buffers."
    )


def main():
    st.title("📊 Real-Time Sales & Marketing Performance Analytics")
    st.caption("Multi-source data → Star schema → RFM / Cohorts / Attribution → Prophet forecasting")

    with st.spinner("Loading data & models..."):
        ctx = get_data()

    pages = {
        "Overview": page_overview,
        "Channels & ROI": page_channels,
        "Segments & Growth": page_segments,
        "Cohort Retention": page_cohorts,
        "Forecasting": page_forecast,
    }

    choice = st.sidebar.radio("Navigate", list(pages.keys()), index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data through:** {ctx['last_date'].date()}")
    st.sidebar.markdown(f"**Customers:** {len(ctx['data']['dim_customer']):,}")
    st.sidebar.markdown(f"**Orders (completed):** {len(ctx['data']['orders_clean']):,}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Tech: Python · Pandas · Prophet · Plotly · Streamlit · SQLite + Parquet star schema")

    pages[choice](ctx)


if __name__ == "__main__":
    main()
