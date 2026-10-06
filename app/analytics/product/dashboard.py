from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.product.engine import (
    detect_product_columns,
    prepare_product_data,
    calculate_product_kpis,
    revenue_by_product,
    units_by_product,
    product_performance,
    revenue_by_category,
    category_performance,
    price_vs_performance,
    ratings_and_reviews,
    product_trend,
    product_summary,
    generate_product_insights,
)


def _chart_key(prefix: str, df: pd.DataFrame) -> str:
    return f"product_{prefix}_{id(df)}"


def _show_chart(fig, key: str):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def render_product_dashboard(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Product Analytics requires a non-empty dataset.")
        return

    data, columns = prepare_product_data(df)

    st.subheader("📦 Product Analytics")

    st.caption(
        "Analyze product performance, revenue, units, pricing, "
        "categories, ratings, reviews, and product trends."
    )

    # =========================================================
    # FIELD DETECTION
    # =========================================================

    with st.expander(
        "🔎 Detected Product Fields",
        expanded=True,
    ):
        detected_rows = []

        for field, column in columns.items():
            if column:
                detected_rows.append(
                    {
                        "Product Field": field.replace("_", " ").title(),
                        "Dataset Column": str(column),
                    }
                )

        if detected_rows:
            st.dataframe(
                pd.DataFrame(detected_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No standard product fields were detected."
            )

    # =========================================================
    # KPIs
    # =========================================================

    kpis = calculate_product_kpis(
        data,
        columns,
    )

    st.markdown("### 📊 Product KPIs")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Products",
        f"{kpis['products']:,}",
    )

    c2.metric(
        "Units Sold",
        f"{kpis['units']:,.0f}",
    )

    c3.metric(
        "Revenue",
        _format_money(kpis["revenue"]),
    )

    c4.metric(
        "Profit",
        _format_money(kpis["profit"]),
    )

    c5.metric(
        "Profit Margin",
        f"{kpis['profit_margin']:.2f}%",
    )

    c6, c7, c8, c9, c10 = st.columns(5)

    c6.metric(
        "Average Price",
        _format_money(kpis["average_price"]),
    )

    c7.metric(
        "Orders",
        f"{kpis['orders']:,.0f}",
    )

    c8.metric(
        "Average Rating",
        f"{kpis['average_rating']:.2f}",
    )

    c9.metric(
        "Reviews",
        f"{kpis['reviews']:,.0f}",
    )

    c10.metric(
        "Active Users",
        f"{kpis['active_users']:,.0f}",
    )

    # =========================================================
    # PRODUCT REVENUE
    # =========================================================

    revenue_product = revenue_by_product(
        data,
        columns,
    )

    if not revenue_product.empty:
        st.markdown("### 💰 Revenue by Product")

        product_column = (
            columns.get("product")
            or columns.get("product_id")
        )

        fig = px.bar(
            revenue_product.head(15),
            x=product_column,
            y="Revenue",
            title="Top Products by Revenue",
        )

        _show_chart(
            fig,
            _chart_key(
                "revenue_product",
                revenue_product,
            ),
        )

    # =========================================================
    # CATEGORY ANALYSIS
    # =========================================================

    category_perf = category_performance(
        data,
        columns,
    )

    if not category_perf.empty:
        st.markdown("### 🗂️ Category Performance")

        category_column = columns.get("category")

        if category_column:
            fig = px.bar(
                category_perf,
                x=category_column,
                y="Revenue"
                if "Revenue" in category_perf.columns
                else category_perf.columns[-1],
                title="Revenue by Category",
            )

            _show_chart(
                fig,
                _chart_key(
                    "category_revenue",
                    category_perf,
                ),
            )

        st.dataframe(
            category_perf,
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # UNITS
    # =========================================================

    units_product = units_by_product(
        data,
        columns,
    )

    if not units_product.empty:
        st.markdown("### 📦 Units Sold by Product")

        product_column = (
            columns.get("product")
            or columns.get("product_id")
        )

        fig = px.bar(
            units_product.head(15),
            x=product_column,
            y="Units",
            title="Top Products by Units Sold",
        )

        _show_chart(
            fig,
            _chart_key(
                "units_product",
                units_product,
            ),
        )

    # =========================================================
    # PRICE VS PERFORMANCE
    # =========================================================

    price_perf = price_vs_performance(
        data,
        columns,
    )

    if not price_perf.empty:
        st.markdown("### 💵 Price vs Product Performance")

        if "Revenue" in price_perf.columns:
            y_column = "Revenue"
        elif "Units" in price_perf.columns:
            y_column = "Units"
        else:
            y_column = "Average Price"

        fig = px.scatter(
            price_perf,
            x="Average Price",
            y=y_column,
            hover_name=(
                columns.get("product")
                or columns.get("product_id")
            ),
            title="Price vs Performance",
            size=(
                "Units"
                if "Units" in price_perf.columns
                else None
            ),
        )

        _show_chart(
            fig,
            _chart_key(
                "price_performance",
                price_perf,
            ),
        )

    # =========================================================
    # RATINGS
    # =========================================================

    ratings = ratings_and_reviews(
        data,
        columns,
    )

    if not ratings.empty:
        st.markdown("### ⭐ Ratings & Reviews")

        product_column = (
            columns.get("product")
            or columns.get("product_id")
        )

        if "Average Rating" in ratings.columns:
            fig = px.bar(
                ratings.head(15),
                x=product_column,
                y="Average Rating",
                title="Product Ratings",
            )

            _show_chart(
                fig,
                _chart_key(
                    "ratings",
                    ratings,
                ),
            )

        st.dataframe(
            ratings.head(25),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # TREND
    # =========================================================

    trend = product_trend(
        data,
        columns,
    )

    if not trend.empty:
        st.markdown("### 📈 Product Revenue Trend")

        fig = px.line(
            trend,
            x="Period",
            y="Revenue",
            markers=True,
            title="Monthly Product Revenue",
        )

        _show_chart(
            fig,
            _chart_key(
                "trend",
                trend,
            ),
        )

    # =========================================================
    # PRODUCT PERFORMANCE TABLE
    # =========================================================

    performance = product_performance(
        data,
        columns,
    )

    if not performance.empty:
        st.markdown("### 🏆 Product Performance")

        st.dataframe(
            performance.head(50),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    st.markdown("### 📋 Product Summary")

    summary = product_summary(
        data,
        columns,
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    # =========================================================
    # AI-STYLE BUSINESS INSIGHTS
    # =========================================================

    st.markdown("### 🧠 Product Business Insights")

    insights = generate_product_insights(
        data,
        columns,
    )

    for insight in insights:
        st.info(insight)

    # =========================================================
    # RAW DATA
    # =========================================================

    with st.expander(
        "🔍 View Product Dataset",
        expanded=False,
    ):
        st.dataframe(
            data.head(100),
            use_container_width=True,
        )
