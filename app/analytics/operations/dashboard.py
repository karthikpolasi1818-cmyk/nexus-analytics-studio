from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.operations.engine import (
    prepare_operations_data,
    calculate_operations_kpis,
    output_by_operation,
    cost_by_department,
    efficiency_by_operation,
    downtime_by_operation,
    quality_by_operation,
    operational_trend,
    operation_performance,
    generate_operations_insights,
)


def _chart_key(prefix, df):
    return f"operations_{prefix}_{id(df)}"


def _show_chart(fig, key):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


def render_operations_dashboard(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning(
            "Operations Analytics requires a non-empty dataset."
        )
        return

    data, columns = prepare_operations_data(df)

    st.subheader("⚙️ Operations Analytics")

    st.caption(
        "Analyze operational output, efficiency, productivity, "
        "downtime, quality, utilization, and process performance."
    )

    # =========================================================
    # DETECTED FIELDS
    # =========================================================

    with st.expander(
        "🔎 Detected Operations Fields",
        expanded=True,
    ):
        rows = [
            {
                "Operations Field": field.replace(
                    "_", " "
                ).title(),
                "Dataset Column": str(column),
            }
            for field, column in columns.items()
            if column
        ]

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No standard operations fields were detected."
            )

    # =========================================================
    # KPIs
    # =========================================================

    kpis = calculate_operations_kpis(
        data,
        columns,
    )

    st.markdown("### 📊 Operations KPIs")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Operations",
        f"{kpis['operations']:,}",
    )

    c2.metric(
        "Output",
        f"{kpis['output']:,.0f}",
    )

    c3.metric(
        "Operating Cost",
        f"{kpis['cost']:,.2f}",
    )

    c4.metric(
        "Efficiency",
        f"{kpis['efficiency']:.2f}%",
    )

    c5.metric(
        "Quality Rate",
        f"{kpis['quality_rate']:.2f}%",
    )

    c6, c7, c8, c9, c10 = st.columns(5)

    c6.metric(
        "Downtime",
        f"{kpis['downtime']:,.2f}",
    )

    c7.metric(
        "Cycle Time",
        f"{kpis['cycle_time']:.2f}",
    )

    c8.metric(
        "Productivity",
        f"{kpis['productivity']:.2f}",
    )

    c9.metric(
        "Utilization",
        f"{kpis['utilization']:.2f}%",
    )

    c10.metric(
        "Defects",
        f"{kpis['defects']:,.0f}",
    )

    # =========================================================
    # OUTPUT BY OPERATION
    # =========================================================

    output = output_by_operation(
        data,
        columns,
    )

    if not output.empty:
        st.markdown("### 🏭 Output by Operation")

        operation = columns.get("operation")

        fig = px.bar(
            output.head(15),
            x=operation,
            y="Output",
            title="Output by Operation",
        )

        _show_chart(
            fig,
            _chart_key("output", output),
        )

    # =========================================================
    # COST BY DEPARTMENT
    # =========================================================

    cost = cost_by_department(
        data,
        columns,
    )

    if not cost.empty:
        st.markdown("### 💰 Operating Cost by Department")

        department = columns.get("department")

        fig = px.bar(
            cost,
            x=department,
            y="Cost",
            title="Operating Cost by Department",
        )

        _show_chart(
            fig,
            _chart_key("cost_department", cost),
        )

        st.dataframe(
            cost,
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # EFFICIENCY
    # =========================================================

    efficiency = efficiency_by_operation(
        data,
        columns,
    )

    if not efficiency.empty:
        st.markdown("### ⚡ Efficiency by Operation")

        operation = columns.get("operation")

        fig = px.bar(
            efficiency.head(15),
            x=operation,
            y="Efficiency",
            title="Operational Efficiency",
        )

        _show_chart(
            fig,
            _chart_key("efficiency", efficiency),
        )

    # =========================================================
    # DOWNTIME
    # =========================================================

    downtime = downtime_by_operation(
        data,
        columns,
    )

    if not downtime.empty:
        st.markdown("### ⏱️ Downtime Analysis")

        operation = columns.get("operation")

        fig = px.bar(
            downtime.head(15),
            x=operation,
            y="Downtime",
            title="Downtime by Operation",
        )

        _show_chart(
            fig,
            _chart_key("downtime", downtime),
        )

    # =========================================================
    # QUALITY
    # =========================================================

    quality = quality_by_operation(
        data,
        columns,
    )

    if not quality.empty:
        st.markdown("### ✅ Quality Analysis")

        operation = columns.get("operation")

        fig = px.bar(
            quality.head(15),
            x=operation,
            y="Quality Rate",
            title="Quality Rate by Operation",
        )

        _show_chart(
            fig,
            _chart_key("quality", quality),
        )

    # =========================================================
    # MONTHLY TREND
    # =========================================================

    trend = operational_trend(
        data,
        columns,
    )

    if not trend.empty:
        st.markdown("### 📈 Operational Trend")

        fig = px.line(
            trend,
            x="Period",
            y="Output",
            markers=True,
            title="Monthly Output Trend",
        )

        _show_chart(
            fig,
            _chart_key("trend", trend),
        )

        trend_display = trend.copy()

        if "Efficiency" in trend_display.columns:
            trend_display["Efficiency"] = trend_display[
                "Efficiency"
            ].round(2)

        st.dataframe(
            trend_display,
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # OPERATION PERFORMANCE
    # =========================================================

    performance = operation_performance(
        data,
        columns,
    )

    if not performance.empty:
        st.markdown("### 🏆 Operation Performance")

        st.dataframe(
            performance.head(50),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # BUSINESS INSIGHTS
    # =========================================================

    st.markdown("### 🧠 Operations Business Insights")

    insights = generate_operations_insights(
        data,
        columns,
    )

    for insight in insights:
        st.info(insight)

    # =========================================================
    # RAW DATA
    # =========================================================

    with st.expander(
        "🔍 View Operations Dataset",
        expanded=False,
    ):
        st.dataframe(
            data.head(100),
            use_container_width=True,
        )
