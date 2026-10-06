import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.manufacturing.engine import (
    prepare_manufacturing_data,
    calculate_manufacturing_kpis,
    production_by_product,
    production_by_line,
    machine_performance,
    downtime_by_machine,
    defects_by_product,
    quality_by_line,
    production_trend,
    cost_by_line,
    generate_manufacturing_insights,
)


def render_manufacturing_dashboard(df):
    render_id = uuid.uuid4().hex[:12]

    work, fields = prepare_manufacturing_data(df)
    kpis = calculate_manufacturing_kpis(work)

    st.markdown("## 🏭 Manufacturing Analytics")
    st.caption(
        "Production performance, machine utilization, quality, downtime, "
        "defects, costs, and manufacturing trends."
    )

    with st.expander("Detected Manufacturing Fields", expanded=False):
        detected = {
            logical.replace("_", " ").title(): source
            for logical, source in fields.items()
            if source
        }

        if detected:
            st.dataframe(
                pd.DataFrame(
                    list(detected.items()),
                    columns=["Manufacturing Field", "Source Column"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No manufacturing-specific fields were detected.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Production", "N/A" if kpis["units_produced"] is None else f"{kpis['units_produced']:,.0f}")
    c2.metric("Target Achievement", "N/A" if kpis["target_achievement"] is None else f"{kpis['target_achievement']:.2f}%")
    c3.metric("Quality Rate", "N/A" if kpis["quality_rate"] is None else f"{kpis['quality_rate']:.2f}%")
    c4.metric("Defects", "N/A" if kpis["defects"] is None else f"{kpis['defects']:,.0f}")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Downtime", "N/A" if kpis["downtime"] is None else f"{kpis['downtime']:,.2f}")
    c6.metric("Maintenance Cost", "N/A" if kpis["maintenance_cost"] is None else f"{kpis['maintenance_cost']:,.2f}")
    c7.metric("Production Cost", "N/A" if kpis["production_cost"] is None else f"{kpis['production_cost']:,.2f}")
    c8.metric("Avg Cycle Time", "N/A" if kpis["cycle_time"] is None else f"{kpis['cycle_time']:.2f}")

    def chart_key(name):
        return f"manufacturing_{render_id}_{name}"

    product = production_by_product(work)
    if not product.empty:
        st.markdown("### Production by Product")
        fig = px.bar(
            product.head(20),
            x="product",
            y="Units Produced",
            title="Production Volume by Product",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("product"))

    line = production_by_line(work)
    if not line.empty:
        st.markdown("### Production by Line")
        fig = px.bar(
            line,
            x="line",
            y="Units Produced",
            title="Production Volume by Production Line",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("line"))

    machine = machine_performance(work)
    if not machine.empty:
        st.markdown("### Machine Performance")
        available_y = [
            col for col in ["Units Produced", "Defects", "Downtime", "Quality Rate"]
            if col in machine.columns
        ]
        if available_y:
            fig = px.bar(
                machine.head(20),
                x="machine",
                y=available_y[0],
                title=f"Machine Performance — {available_y[0]}",
            )
            st.plotly_chart(fig, use_container_width=True, key=chart_key("machine"))

    downtime = downtime_by_machine(work)
    if not downtime.empty:
        st.markdown("### Machine Downtime")
        fig = px.bar(
            downtime.head(20),
            x="machine",
            y="Downtime",
            title="Downtime by Machine",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("downtime"))

    defects = defects_by_product(work)
    if not defects.empty:
        st.markdown("### Defects by Product")
        fig = px.bar(
            defects.head(20),
            x="product",
            y="Defects",
            title="Defects by Product",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("defects"))

    quality = quality_by_line(work)
    if not quality.empty:
        st.markdown("### Quality by Production Line")
        fig = px.bar(
            quality,
            x="line",
            y="Quality Rate",
            title="Average Quality Rate by Production Line",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("quality"))

    trend = production_trend(work)
    if not trend.empty:
        st.markdown("### Production Trend")
        y_columns = ["Units_Produced"]
        if "Defects" in trend.columns:
            y_columns.append("Defects")
        fig = px.line(
            trend,
            x="Date",
            y=y_columns,
            title="Manufacturing Production Trend",
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key("trend"))

    costs = cost_by_line(work)
    if not costs.empty:
        st.markdown("### Manufacturing Cost by Line")
        value_columns = [
            col for col in ["Production Cost", "Maintenance Cost"]
            if col in costs.columns
        ]
        if value_columns:
            fig = px.bar(
                costs,
                x="line",
                y=value_columns,
                barmode="group",
                title="Production and Maintenance Cost by Line",
            )
            st.plotly_chart(fig, use_container_width=True, key=chart_key("cost"))

    insights = generate_manufacturing_insights(work)
    if insights:
        st.markdown("### Manufacturing Insights")
        for insight in insights:
            st.info(insight)

    with st.expander("View Prepared Manufacturing Data", expanded=False):
        st.dataframe(
            work,
            use_container_width=True,
            hide_index=True,
        )
