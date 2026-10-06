import uuid
import pandas as pd
import plotly.express as px
import streamlit as st

from .engine import (
    prepare_ecommerce_data,
    calculate_ecommerce_kpis,
    sales_by_category,
    sales_by_product,
    sales_by_channel,
    orders_by_channel,
    customer_segments,
    daily_sales_trend,
    payment_method_analysis,
    order_status_analysis,
    geographic_sales,
    generate_ecommerce_insights,
)


def _key(prefix):
    return f"ecommerce_{prefix}_{uuid.uuid4().hex[:12]}"


def _money(value):
    return f"{value:,.2f}"


def render_ecommerce_dashboard(df):
    data = prepare_ecommerce_data(df)
    kpis = calculate_ecommerce_kpis(data)

    st.subheader("🛒 E-commerce Analytics")

    mapping = data.attrs.get("ecommerce_mapping", {})
    if mapping:
        st.markdown("### Detected Fields")
        st.dataframe(
            pd.DataFrame(
                [{"Analytics Field": k, "Source Column": v} for k, v in mapping.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    cards = [
        ("Orders", kpis.get("Total Orders")),
        ("Customers", kpis.get("Unique Customers")),
        ("Revenue", kpis.get("Revenue")),
        ("AOV", kpis.get("Average Order Value")),
        ("Profit", kpis.get("Profit")),
        ("Margin", kpis.get("Profit Margin")),
        ("Units Sold", kpis.get("Units Sold")),
        ("Avg Rating", kpis.get("Average Rating")),
    ]

    available = [(label, value) for label, value in cards if value is not None]
    if available:
        cols = st.columns(min(4, len(available)))
        for i, (label, value) in enumerate(available):
            with cols[i % len(cols)]:
                if label in {"Revenue", "AOV", "Profit"}:
                    st.metric(label, _money(value))
                elif label == "Margin":
                    st.metric(label, f"{value:.2f}%")
                elif label == "Avg Rating":
                    st.metric(label, f"{value:.2f}")
                elif isinstance(value, float):
                    st.metric(label, f"{value:,.2f}")
                else:
                    st.metric(label, f"{value:,}")

    st.markdown("---")

    category = sales_by_category(data)
    if not category.empty:
        st.markdown("### Sales by Category")
        fig = px.bar(category.head(20), x="category", y="revenue", title="Revenue by Category")
        st.plotly_chart(fig, use_container_width=True, key=_key("category"))

    channel = sales_by_channel(data)
    if not channel.empty:
        st.markdown("### Sales by Channel")
        fig = px.bar(channel, x="channel", y="revenue", title="Revenue by Channel")
        st.plotly_chart(fig, use_container_width=True, key=_key("channel"))

    orders_channel = orders_by_channel(data)
    if not orders_channel.empty:
        st.markdown("### Orders by Channel")
        fig = px.pie(orders_channel, names="channel", values="Orders", title="Order Distribution by Channel")
        st.plotly_chart(fig, use_container_width=True, key=_key("orders_channel"))

    product = sales_by_product(data)
    if not product.empty:
        st.markdown("### Top Products")
        fig = px.bar(
            product.head(15),
            x="revenue",
            y="product",
            orientation="h",
            title="Top Products by Revenue",
        )
        st.plotly_chart(fig, use_container_width=True, key=_key("product"))

    segment = customer_segments(data)
    if not segment.empty:
        st.markdown("### Customer Segments")
        value_col = "Revenue" if "Revenue" in segment.columns else segment.columns[-1]
        fig = px.bar(
            segment,
            x="customer_segment",
            y=value_col,
            title="Customer Segment Performance",
        )
        st.plotly_chart(fig, use_container_width=True, key=_key("segment"))

    trend = daily_sales_trend(data)
    if not trend.empty:
        st.markdown("### Sales Trend")
        fig = px.line(trend, x="Date", y="Revenue", title="Daily Revenue Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True, key=_key("trend"))

    payment = payment_method_analysis(data)
    if not payment.empty:
        st.markdown("### Payment Methods")
        value_col = payment.columns[-1]
        fig = px.bar(
            payment,
            x="payment_method",
            y=value_col,
            title="Payment Method Analysis",
        )
        st.plotly_chart(fig, use_container_width=True, key=_key("payment"))

    status = order_status_analysis(data)
    if not status.empty:
        st.markdown("### Order Status")
        fig = px.bar(
            status,
            x="order_status",
            y="Orders",
            title="Order Status Distribution",
        )
        st.plotly_chart(fig, use_container_width=True, key=_key("status"))

    geo = geographic_sales(data)
    if not geo.empty:
        geo_col = geo.columns[0]
        st.markdown(f"### Sales by {geo_col.replace('_', ' ').title()}")
        fig = px.bar(
            geo.head(20),
            x=geo_col,
            y="revenue",
            title=f"Revenue by {geo_col.replace('_', ' ').title()}",
        )
        st.plotly_chart(fig, use_container_width=True, key=_key("geo"))

    insights = generate_ecommerce_insights(data)
    if insights:
        st.markdown("### AI-Style Business Insights")
        for insight in insights:
            st.info(insight)

    st.markdown("### Raw Data")
    st.dataframe(data, use_container_width=True, height=420)
