import streamlit as st
import plotly.express as px

from .engine import (
    prepare_marketing_data,
    calculate_marketing_kpis,
    revenue_by_channel,
    spend_by_channel,
    roas_by_channel,
    conversions_by_channel,
    campaign_performance,
    marketing_funnel,
    generate_marketing_insights,
)


def render_marketing_dashboard(df):

    st.markdown(
        "## 📣 Marketing Analytics Dashboard"
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    df, mapping = prepare_marketing_data(
        df
    )

    # =====================================================
    # DETECTED FIELDS
    # =====================================================

    with st.expander(
        "🔍 Detected Marketing Fields",
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

    kpis = calculate_marketing_kpis(
        df,
        mapping
    )

    st.markdown(
        "### 📊 Marketing Performance"
    )

    row1 = st.columns(4)

    with row1[0]:

        st.metric(
            "Marketing Spend",
            f"₹{kpis['spend']:,.0f}",
        )

    with row1[1]:

        st.metric(
            "Revenue",
            f"₹{kpis['revenue']:,.0f}",
        )

    with row1[2]:

        st.metric(
            "Conversions",
            f"{kpis['conversions']:,.0f}",
        )

    with row1[3]:

        st.metric(
            "ROAS",
            f"{kpis['roas']:.2f}x",
        )

    row2 = st.columns(4)

    with row2[0]:

        st.metric(
            "Impressions",
            f"{kpis['impressions']:,.0f}",
        )

    with row2[1]:

        st.metric(
            "Clicks",
            f"{kpis['clicks']:,.0f}",
        )

    with row2[2]:

        st.metric(
            "CTR",
            f"{kpis['ctr']:.2f}%",
        )

    with row2[3]:

        st.metric(
            "CAC",
            f"₹{kpis['cac']:,.2f}",
        )

    row3 = st.columns(2)

    with row3[0]:

        st.metric(
            "Conversion Rate",
            f"{kpis['conversion_rate']:.2f}%",
        )

    with row3[1]:

        st.metric(
            "Marketing ROI",
            f"{kpis['roi']:.2f}%",
        )

    st.divider()

    # =====================================================
    # CHANNEL DATA
    # =====================================================

    revenue_channel = (
        revenue_by_channel(
            df,
            mapping
        )
    )

    spend_channel = (
        spend_by_channel(
            df,
            mapping
        )
    )

    roas_channel = (
        roas_by_channel(
            df,
            mapping
        )
    )

    conversions_channel = (
        conversions_by_channel(
            df,
            mapping
        )
    )

    # =====================================================
    # REVENUE + SPEND
    # =====================================================

    col1, col2 = st.columns(2)

    if revenue_channel is not None:

        with col1:

            st.markdown(
                "### 💰 Revenue by Channel"
            )

            fig = px.bar(
                revenue_channel,
                x=mapping["channel"],
                y=mapping["revenue"],
                title=(
                    "Attributed Revenue by Channel"
                ),
                text_auto=".2s",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if spend_channel is not None:

        with col2:

            st.markdown(
                "### 💸 Spend by Channel"
            )

            fig = px.bar(
                spend_channel,
                x=mapping["channel"],
                y=mapping["spend"],
                title=(
                    "Marketing Spend by Channel"
                ),
                text_auto=".2s",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # ROAS + CONVERSIONS
    # =====================================================

    col3, col4 = st.columns(2)

    if roas_channel is not None:

        with col3:

            st.markdown(
                "### 📈 ROAS by Channel"
            )

            fig = px.bar(
                roas_channel,
                x=mapping["channel"],
                y="ROAS",
                title=(
                    "Return on Ad Spend by Channel"
                ),
                text_auto=".2f",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if conversions_channel is not None:

        with col4:

            st.markdown(
                "### 🎯 Conversions by Channel"
            )

            fig = px.bar(
                conversions_channel,
                x=mapping["channel"],
                y=mapping["conversions"],
                title=(
                    "Conversions by Channel"
                ),
                text_auto=".2s",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # FUNNEL
    # =====================================================

    funnel = marketing_funnel(
        df,
        mapping
    )

    if funnel is not None:

        st.markdown(
            "### 🔽 Marketing Funnel"
        )

        fig = px.funnel(
            funnel,
            x="Count",
            y="Stage",
            title=(
                "Impressions → Clicks → Conversions"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CAMPAIGN PERFORMANCE
    # =====================================================

    campaign_data = campaign_performance(
        df,
        mapping
    )

    if campaign_data is not None:

        st.markdown(
            "### 🏆 Campaign Performance"
        )

        table_columns = []

        campaign_column = (
            mapping["campaign"]
        )

        if campaign_column:

            table_columns.append(
                campaign_column
            )

        for column in [
            "Impressions",
            "Clicks",
            "Conversions",
            "Spend",
            "Revenue",
            "CTR",
            "Conversion Rate",
            "ROAS",
            "ROI",
            "CAC",
        ]:

            if column in campaign_data.columns:

                table_columns.append(
                    column
                )

        display_data = (
            campaign_data[
                table_columns
            ]
            .copy()
        )

        if "ROAS" in display_data.columns:

            display_data = (
                display_data
                .sort_values(
                    "ROAS",
                    ascending=False
                )
            )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )

        # =================================================
        # CAMPAIGN SPEND VS REVENUE
        # =================================================

        if (
            "Spend" in campaign_data.columns
            and "Revenue"
            in campaign_data.columns
        ):

            st.markdown(
                "### 💰 Campaign Spend vs Revenue"
            )

            chart_columns = [
                campaign_column,
                "Spend",
                "Revenue",
            ]

            chart_df = (
                campaign_data[
                    chart_columns
                ]
                .melt(
                    id_vars=campaign_column,
                    var_name="Metric",
                    value_name="Amount",
                )
            )

            fig = px.bar(
                chart_df,
                x=campaign_column,
                y="Amount",
                color="Metric",
                barmode="group",
                title=(
                    "Campaign Spend vs Revenue"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # CAMPAIGN ROAS
        # =================================================

        if "ROAS" in campaign_data.columns:

            st.markdown(
                "### 📊 Campaign ROAS"
            )

            roas_chart = (
                campaign_data[
                    [
                        campaign_column,
                        "ROAS",
                    ]
                ]
                .dropna()
                .sort_values(
                    "ROAS",
                    ascending=True
                )
            )

            fig = px.bar(
                roas_chart,
                x="ROAS",
                y=campaign_column,
                orientation="h",
                title=(
                    "ROAS by Campaign"
                ),
                text_auto=".2f",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    st.markdown(
        "### 🧠 Marketing Business Insights"
    )

    insights = generate_marketing_insights(
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
            "Not enough marketing fields were "
            "detected to generate business insights."
        )

    # =====================================================
    # DATASET
    # =====================================================

    with st.expander(
        "📋 View Marketing Dataset",
        expanded=False
    ):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )