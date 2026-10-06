import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.fraud.engine import (
    prepare_fraud_data,
    calculate_fraud_kpis,
    fraud_by_type,
    fraud_by_merchant,
    fraud_by_country,
    risk_distribution,
    daily_fraud_trend,
    high_risk_transactions,
    generate_fraud_insights,
)


def render_fraud_dashboard(df):

    render_id = uuid.uuid4().hex[:12]

    work, fields = prepare_fraud_data(df)
    kpis = calculate_fraud_kpis(work)

    st.markdown("## 🛡️ Fraud Analytics")

    st.caption(
        "Transaction risk, fraud patterns, high-risk activity, "
        "and fraud concentration analysis."
    )

    # =========================================================
    # DETECTED FIELDS
    # =========================================================

    with st.expander(
        "Detected Fraud Fields",
        expanded=False
    ):

        detected = {
            logical.replace("_", " ").title(): source
            for logical, source in fields.items()
            if source
        }

        if detected:

            st.dataframe(
                pd.DataFrame(
                    list(detected.items()),
                    columns=[
                        "Fraud Field",
                        "Source Column"
                    ],
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.warning(
                "No fraud-specific fields were detected."
            )

    # =========================================================
    # KPI CARDS
    # =========================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{kpis['transactions']:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{kpis['fraud_transactions']:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{kpis['fraud_rate']:.2f}%"
    )

    c4.metric(
        "Flagged Amount",
        (
            "N/A"
            if kpis["fraud_amount"] is None
            else f"{kpis['fraud_amount']:,.2f}"
        )
    )

    if kpis["total_amount"] is not None:

        c5, c6 = st.columns(2)

        c5.metric(
            "Total Transaction Value",
            f"{kpis['total_amount']:,.2f}"
        )

        c6.metric(
            "Average Transaction",
            f"{kpis['average_transaction']:,.2f}"
        )

    # =========================================================
    # UNIQUE STREAMLIT KEYS
    # =========================================================

    def chart_key(name):

        return (
            f"fraud_"
            f"{render_id}_"
            f"{name}"
        )

    # =========================================================
    # FRAUD BY TRANSACTION TYPE
    # =========================================================

    transaction_type_data = fraud_by_type(work)

    if not transaction_type_data.empty:

        st.markdown(
            "### Fraud by Transaction Type"
        )

        # IMPORTANT:
        # The engine creates "Fraud_Rate".
        # Do NOT change this to "Fraud Rate".

        fig = px.bar(
            transaction_type_data,
            x="transaction_type",
            y="Fraud_Rate",
            title="Fraud Rate by Transaction Type",
            labels={
                "transaction_type": "Transaction Type",
                "Fraud_Rate": "Fraud Rate (%)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "fraud_by_type"
            ),
        )

    # =========================================================
    # MERCHANT RISK
    # =========================================================

    merchant_data = fraud_by_merchant(work)

    if not merchant_data.empty:

        st.markdown(
            "### Merchant Risk"
        )

        fig = px.bar(
            merchant_data.head(15),
            x="merchant",
            y="Fraud_Transactions",
            title="Top Merchants by Fraud Transactions",
            labels={
                "merchant": "Merchant",
                "Fraud_Transactions": (
                    "Fraud Transactions"
                ),
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "merchant_fraud"
            ),
        )

    # =========================================================
    # COUNTRY RISK
    # =========================================================

    country_data = fraud_by_country(work)

    if not country_data.empty:

        st.markdown(
            "### Country Risk"
        )

        fig = px.bar(
            country_data.head(15),
            x="country",
            y="Fraud Rate",
            title="Fraud Rate by Country",
            labels={
                "country": "Country",
                "Fraud Rate": "Fraud Rate (%)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "country_fraud"
            ),
        )

    # =========================================================
    # RISK SCORE
    # =========================================================

    risk_data = risk_distribution(work)

    if not risk_data.empty:

        st.markdown(
            "### Risk Score Distribution"
        )

        fig = px.histogram(
            risk_data,
            x="risk_score",
            nbins=20,
            title=(
                "Transaction Risk "
                "Score Distribution"
            ),
            labels={
                "risk_score": "Risk Score"
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "risk_distribution"
            ),
        )

    # =========================================================
    # DAILY FRAUD TREND
    # =========================================================

    trend_data = daily_fraud_trend(work)

    if not trend_data.empty:

        st.markdown(
            "### Fraud Trend"
        )

        fig = px.line(
            trend_data,
            x="Date",
            y=[
                "Transactions",
                "Fraud_Transactions"
            ],
            title=(
                "Daily Transaction "
                "and Fraud Trend"
            ),
            labels={
                "Date": "Date",
                "value": "Transactions",
                "variable": "Metric",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "daily_fraud_trend"
            ),
        )

    # =========================================================
    # HIGH-RISK TRANSACTIONS
    # =========================================================

    high_risk_data = high_risk_transactions(
        work
    )

    if not high_risk_data.empty:

        st.markdown(
            "### High-Risk Transactions"
        )

        st.dataframe(
            high_risk_data.head(100),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # INSIGHTS
    # =========================================================

    insights = generate_fraud_insights(
        work
    )

    if insights:

        st.markdown(
            "### Fraud Insights"
        )

        for insight in insights:

            st.info(
                insight
            )

    # =========================================================
    # PREPARED DATA
    # =========================================================

    with st.expander(
        "View Prepared Fraud Data",
        expanded=False
    ):

        st.dataframe(
            work,
            use_container_width=True,
            hide_index=True,
        )