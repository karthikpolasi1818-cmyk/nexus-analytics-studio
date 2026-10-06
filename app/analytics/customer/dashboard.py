import streamlit as st
import plotly.express as px

from .engine import (
    prepare_customer_data,
    calculate_customer_kpis,
    revenue_by_segment,
    customers_by_segment,
    revenue_by_region,
    top_customers,
    customer_summary,
    generate_customer_insights,
)


def render_customer_dashboard(df):

    st.markdown(
        "## 👥 Customer Analytics Dashboard"
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    df, mapping = prepare_customer_data(
        df
    )

    # =====================================================
    # DETECTED FIELDS
    # =====================================================

    with st.expander(
        "🔍 Detected Customer Fields",
        expanded=False
    ):

        for field, column in mapping.items():

            label = (
                field
                .replace("_", " ")
                .title()
            )

            if column:

                st.write(
                    f"**{label}** → `{column}`"
                )

            else:

                st.write(
                    f"**{label}** → Not detected"
                )

    # =====================================================
    # KPIs
    # =====================================================

    kpis = calculate_customer_kpis(
        df,
        mapping
    )

    st.markdown(
        "### 📊 Customer Performance"
    )

    row1 = st.columns(4)

    with row1[0]:

        st.metric(
            "Customers",
            f"{kpis['total_customers']:,}",
            help=kpis[
                "customer_count_type"
            ],
        )

    with row1[1]:

        st.metric(
            "Revenue",
            f"₹{kpis['total_revenue']:,.0f}",
        )

    with row1[2]:

        st.metric(
            "Orders",
            f"{kpis['total_orders']:,}",
        )

    with row1[3]:

        st.metric(
            "Avg Customer Value",
            f"₹{kpis['average_customer_value']:,.2f}",
        )

    row2 = st.columns(4)

    with row2[0]:

        st.metric(
            "Avg Order Value",
            f"₹{kpis['average_order_value']:,.2f}",
        )

    with row2[1]:

        retention_value = (
            f"{kpis['retention_rate']:.2f}%"
            if kpis["retention_rate"]
            is not None
            else "N/A"
        )

        st.metric(
            "Retention Rate",
            retention_value,
        )

    with row2[2]:

        churn_value = (
            f"{kpis['churn_rate']:.2f}%"
            if kpis["churn_rate"]
            is not None
            else "N/A"
        )

        st.metric(
            "Churn Rate",
            churn_value,
        )

    with row2[3]:

        clv_value = (
            f"₹{kpis['average_clv']:,.2f}"
            if kpis["average_clv"]
            is not None
            else "N/A"
        )

        st.metric(
            "Avg CLV",
            clv_value,
        )

    st.divider()

    # =====================================================
    # SEGMENT ANALYSIS
    # =====================================================

    segment_revenue = revenue_by_segment(
        df,
        mapping
    )

    segment_customers = customers_by_segment(
        df,
        mapping
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # REVENUE BY SEGMENT
    # -----------------------------------------------------

    if segment_revenue is not None:

        with col1:

            st.markdown(
                "### 💰 Revenue by Segment"
            )

            segment_column = (
                mapping["segment"]
            )

            revenue_column = (
                mapping["revenue"]
            )

            fig = px.bar(
                segment_revenue,
                x=segment_column,
                y=revenue_column,
                title=(
                    "Customer Revenue by Segment"
                ),
                text_auto=".2s",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # -----------------------------------------------------
    # CUSTOMERS BY SEGMENT
    # -----------------------------------------------------

    if segment_customers is not None:

        with col2:

            st.markdown(
                "### 👥 Customers by Segment"
            )

            segment_column = (
                mapping["segment"]
            )

            fig = px.pie(
                segment_customers,
                names=segment_column,
                values="Customers",
                title=(
                    "Customer Distribution"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # REGION ANALYSIS
    # =====================================================

    region_data = revenue_by_region(
        df,
        mapping
    )

    if region_data is not None:

        st.markdown(
            "### 🌎 Customer Revenue by Region"
        )

        region_column = (
            mapping["region"]
        )

        revenue_column = (
            mapping["revenue"]
        )

        fig = px.bar(
            region_data,
            x=region_column,
            y=revenue_column,
            title=(
                "Revenue by Customer Region"
            ),
            text_auto=".2s",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # TOP CUSTOMERS
    # =====================================================

    top_data = top_customers(
        df,
        mapping
    )

    if top_data is not None:

        st.markdown(
            "### 🏆 Top Customers"
        )

        fig = px.bar(
            top_data,
            x="Revenue",
            y=mapping["customer_id"],
            orientation="h",
            title=(
                "Top 10 Customers by Revenue"
            ),
            text_auto=".2s",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No dedicated Customer ID was detected. "
            "Individual customer ranking is therefore "
            "not available for this dataset."
        )

    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    summary = customer_summary(
        df,
        mapping
    )

    if summary is not None:

        st.markdown(
            "### 📋 Customer Summary"
        )

        if "Revenue" in summary.columns:

            summary = summary.sort_values(
                "Revenue",
                ascending=False
            )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    st.markdown(
        "### 🧠 Customer Business Insights"
    )

    insights = generate_customer_insights(
        df,
        mapping
    )

    if insights:

        for insight in insights:

            st.info(
                f"💡 {insight}"
            )

    else:

        st.info(
            "Not enough customer-related fields "
            "were detected to generate insights."
        )