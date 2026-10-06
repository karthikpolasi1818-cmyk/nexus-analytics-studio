import streamlit as st
import plotly.express as px

from .engine import (
    prepare_sales_data,
    calculate_sales_kpis,
    revenue_by_region,
    revenue_by_product,
    revenue_by_category,
    revenue_over_time,
)


def render_sales_dashboard(df):

    st.markdown(
        "## 📈 Sales Analytics Dashboard"
    )

    df, mapping = prepare_sales_data(
        df
    )

    # =====================================================
    # COLUMN DETECTION
    # =====================================================

    with st.expander(
        "🔍 Detected Sales Fields",
        expanded=False
    ):

        for field, column in mapping.items():

            if column:

                st.write(
                    f"**{field.title()}** → "
                    f"`{column}`"
                )

            else:

                st.write(
                    f"**{field.title()}** → "
                    "Not detected"
                )

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    kpis = calculate_sales_kpis(
        df,
        mapping
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Revenue",
        f"₹{kpis['revenue']:,.0f}"
    )

    c2.metric(
        "Profit",
        f"₹{kpis['profit']:,.0f}"
    )

    c3.metric(
        "Orders",
        f"{kpis['orders']:,}"
    )

    c4.metric(
        "Quantity",
        f"{kpis['quantity']:,.0f}"
    )

    c5.metric(
        "Profit Margin",
        f"{kpis['margin']:.2f}%"
    )

    st.divider()

    # =====================================================
    # REVENUE TREND
    # =====================================================

    trend = revenue_over_time(
        df,
        mapping
    )

    if trend is not None:

        st.markdown(
            "### 📈 Revenue Trend"
        )

        revenue_column = mapping[
            "revenue"
        ]

        fig = px.line(
            trend,
            x="period",
            y=revenue_column,
            markers=True,
            title="Monthly Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # REGION + PRODUCT
    # =====================================================

    col1, col2 = st.columns(2)

    region_data = revenue_by_region(
        df,
        mapping
    )

    if region_data is not None:

        with col1:

            st.markdown(
                "### 🌎 Revenue by Region"
            )

            region_column = mapping[
                "region"
            ]

            revenue_column = mapping[
                "revenue"
            ]

            fig = px.bar(
                region_data,
                x=region_column,
                y=revenue_column,
                title="Regional Revenue"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    product_data = revenue_by_product(
        df,
        mapping
    )

    if product_data is not None:

        with col2:

            st.markdown(
                "### 🏆 Top Products"
            )

            product_column = mapping[
                "product"
            ]

            revenue_column = mapping[
                "revenue"
            ]

            fig = px.bar(
                product_data,
                x=revenue_column,
                y=product_column,
                orientation="h",
                title="Top 10 Products"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # CATEGORY ANALYSIS
    # =====================================================

    category_data = revenue_by_category(
        df,
        mapping
    )

    if category_data is not None:

        st.markdown(
            "### 📦 Revenue by Category"
        )

        category_column = mapping[
            "category"
        ]

        revenue_column = mapping[
            "revenue"
        ]

        fig = px.pie(
            category_data,
            names=category_column,
            values=revenue_column,
            title="Category Contribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # DATA TABLE
    # =====================================================

    with st.expander(
        "📋 View Sales Dataset"
    ):

        st.dataframe(
            df,
            use_container_width=True
        )