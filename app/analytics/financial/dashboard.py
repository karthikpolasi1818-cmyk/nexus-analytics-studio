import streamlit as st
import plotly.express as px

from .engine import (
    prepare_financial_data,
    calculate_financial_kpis,
    revenue_by_category,
    expenses_by_category,
    profit_by_category,
    revenue_by_department,
    expenses_by_department,
    financial_trend,
    budget_analysis,
    financial_summary,
    generate_financial_insights,
)


def money(value):
    """Format financial values for dashboard KPI cards."""
    if value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"

    if abs(value) >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.2f}K"

    return f"₹{value:,.2f}"


def _chart_key(prefix, df):
    """
    Generate a unique Plotly key for each dashboard instance.

    The same Financial dashboard can be rendered multiple times when
    an Excel file contains multiple sheets. Using the dataframe object
    identity prevents StreamlitDuplicateElementId errors between sheets.
    """
    return f"financial_{prefix}_{id(df)}"


def _show_chart(fig, key):
    """Render a Plotly chart with a guaranteed explicit key."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


def render_financial_dashboard(df):
    st.markdown("## 💰 Financial Analytics")

    st.caption(
        "Financial performance, profitability, expenses, "
        "budget utilization and cash-flow analysis."
    )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    prepared_df, mapping = prepare_financial_data(df)

    # --------------------------------------------------
    # DETECTED FINANCIAL FIELDS
    # --------------------------------------------------

    st.markdown("### 🔍 Detected Financial Fields")

    detected = {
        key.replace("_", " ").title(): value
        for key, value in mapping.items()
        if value is not None
    }

    if not detected:
        st.warning(
            "No recognizable financial fields were detected."
        )
        return

    st.json(detected)

    # --------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------

    kpis = calculate_financial_kpis(prepared_df)

    st.markdown("### 📊 Financial KPIs")

    row1 = st.columns(4)

    row1[0].metric(
        "Revenue",
        money(kpis["Revenue"]),
    )

    row1[1].metric(
        "Expenses",
        money(kpis["Expenses"]),
    )

    row1[2].metric(
        "Profit",
        money(kpis["Profit"]),
    )

    row1[3].metric(
        "Profit Margin",
        f"{kpis['Profit Margin']:.2f}%",
    )

    row2 = st.columns(4)

    row2[0].metric(
        "Cost",
        money(kpis["Cost"]),
    )

    row2[1].metric(
        "Budget",
        money(kpis["Budget"]),
    )

    row2[2].metric(
        "Cash Flow",
        money(kpis["Cash Flow"]),
    )

    row2[3].metric(
        "Expense Ratio",
        f"{kpis['Expense Ratio']:.2f}%",
    )

    # --------------------------------------------------
    # BUDGET KPI
    # --------------------------------------------------

    if kpis["Budget"] != 0:
        st.markdown("### 🎯 Budget Performance")

        budget_cols = st.columns(2)

        budget_cols[0].metric(
            "Budget Variance",
            money(kpis["Budget Variance"]),
        )

        budget_cols[1].metric(
            "Budget Utilization",
            f"{kpis['Budget Utilization']:.2f}%",
        )

    # --------------------------------------------------
    # FINANCIAL TREND
    # --------------------------------------------------

    trend = financial_trend(prepared_df)

    if not trend.empty:
        st.markdown("### 📈 Financial Trend")

        metrics = [
            column
            for column in [
                "Revenue",
                "Expenses",
                "Profit",
            ]
            if column in trend.columns
        ]

        if metrics:
            fig = px.line(
                trend,
                x="Period",
                y=metrics,
                markers=True,
                title="Revenue, Expenses and Profit Over Time",
            )

            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Amount",
                legend_title="Metric",
            )

            _show_chart(
                fig,
                _chart_key("trend", prepared_df),
            )

    # --------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------

    category_revenue = revenue_by_category(
        prepared_df
    )

    category_expenses = expenses_by_category(
        prepared_df
    )

    category_profit = profit_by_category(
        prepared_df
    )

    if not category_revenue.empty:
        st.markdown("### 🗂️ Revenue by Category")

        fig = px.bar(
            category_revenue,
            x="Category",
            y="Revenue",
            title="Revenue by Financial Category",
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Revenue",
        )

        _show_chart(
            fig,
            _chart_key("revenue_category", prepared_df),
        )

    if not category_expenses.empty:
        st.markdown("### 💸 Expenses by Category")

        fig = px.bar(
            category_expenses,
            x="Category",
            y="Expenses",
            title="Expenses by Category",
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Expenses",
        )

        _show_chart(
            fig,
            _chart_key("expenses_category", prepared_df),
        )

    if not category_profit.empty:
        st.markdown("### 📊 Profit by Category")

        fig = px.bar(
            category_profit,
            x="Category",
            y="Profit",
            title="Profit by Category",
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Profit",
        )

        _show_chart(
            fig,
            _chart_key("profit_category", prepared_df),
        )

    # --------------------------------------------------
    # DEPARTMENT ANALYSIS
    # --------------------------------------------------

    department_revenue = revenue_by_department(
        prepared_df
    )

    department_expenses = expenses_by_department(
        prepared_df
    )

    if not department_revenue.empty:
        st.markdown("### 🏢 Revenue by Department")

        fig = px.bar(
            department_revenue,
            x="Department",
            y="Revenue",
            title="Revenue by Department",
        )

        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Revenue",
        )

        _show_chart(
            fig,
            _chart_key("revenue_department", prepared_df),
        )

    if not department_expenses.empty:
        st.markdown("### 🏢 Expenses by Department")

        fig = px.bar(
            department_expenses,
            x="Department",
            y="Expenses",
            title="Expenses by Department",
        )

        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Expenses",
        )

        _show_chart(
            fig,
            _chart_key("expenses_department", prepared_df),
        )

    # --------------------------------------------------
    # BUDGET VS ACTUAL
    # --------------------------------------------------

    budget_df = budget_analysis(prepared_df)

    if not budget_df.empty:
        st.markdown("### 🎯 Budget vs Actual")

        dimensions = [
            column
            for column in [
                "Period",
                "Category",
                "Department",
            ]
            if column in budget_df.columns
        ]

        if (
            dimensions
            and {"Budget", "Actual"}.issubset(
                budget_df.columns
            )
        ):
            dimension = dimensions[0]

            fig = px.bar(
                budget_df,
                x=dimension,
                y=["Budget", "Actual"],
                barmode="group",
                title=f"Budget vs Actual by {dimension}",
            )

            fig.update_layout(
                xaxis_title=dimension,
                yaxis_title="Amount",
                legend_title="Metric",
            )

            _show_chart(
                fig,
                _chart_key("budget_actual", prepared_df),
            )

        st.dataframe(
            budget_df,
            use_container_width=True,
        )

    # --------------------------------------------------
    # FINANCIAL SUMMARY
    # --------------------------------------------------

    st.markdown("### 📋 Financial Summary")

    summary = financial_summary(
        prepared_df
    )

    st.dataframe(
        summary,
        use_container_width=True,
    )

    # --------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------

    st.markdown("### 🧠 Financial Business Insights")

    insights = generate_financial_insights(
        prepared_df
    )

    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.info(
            "Not enough financial information was available "
            "to generate insights."
        )

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------

    with st.expander("📄 View Financial Dataset"):
        st.dataframe(
            prepared_df,
            use_container_width=True,
        )
