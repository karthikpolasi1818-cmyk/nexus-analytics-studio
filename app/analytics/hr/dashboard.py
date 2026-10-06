import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.hr.engine import (
    prepare_hr_data,
    calculate_hr_kpis,
    employees_by_department,
    salary_by_department,
    performance_by_department,
    attrition_by_department,
    gender_distribution,
    role_distribution,
    attendance_by_department,
    generate_hr_insights,
)


def _safe(value):
    return value == value


def _fmt_number(value):
    if value is None or not _safe(value):
        return "N/A"
    return f"{value:,.0f}"


def _fmt_percent(value):
    if value is None or not _safe(value):
        return "N/A"
    return f"{value:.2f}%"


def render_hr_dashboard(df):
    # IMPORTANT:
    # A single UUID is created for EVERY dashboard render.
    # This prevents duplicate Streamlit element keys when the same HR
    # dashboard is rendered for multiple uploaded files / Excel sheets.
    render_id = uuid.uuid4().hex[:12]

    work, fields = prepare_hr_data(df)
    kpis = calculate_hr_kpis(work)

    st.markdown("## 👥 HR Analytics")
    st.caption(
        "Workforce intelligence covering headcount, compensation, "
        "performance, attendance, demographics, and attrition."
    )

    with st.expander("Detected HR Fields", expanded=False):
        detected = {
            logical.replace("_", " ").title(): source
            for logical, source in fields.items()
            if source
        }
        if detected:
            st.dataframe(
                pd.DataFrame(
                    list(detected.items()),
                    columns=["HR Field", "Source Column"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No HR-specific fields were detected in this dataset.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{kpis['employees']:,}")
    c2.metric("Average Salary", _fmt_number(kpis["average_salary"]))
    c3.metric("Avg Performance", _fmt_number(kpis["average_performance"]))
    c4.metric("Attrition Rate", _fmt_percent(kpis["attrition_rate"]))

    if kpis.get("average_attendance") is not None:
        st.metric("Average Attendance", _fmt_percent(kpis["average_attendance"]))

    def chart_key(name):
        # UUID makes every chart instance unique even if render_hr_dashboard()
        # is called repeatedly for the same dataframe.
        return f"hr_{render_id}_{name}"

    dep = employees_by_department(work)
    if not dep.empty:
        st.markdown("### Headcount by Department")
        fig = px.bar(
            dep,
            x="department",
            y="Employees",
            title="Employees by Department",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("headcount_department"),
        )

    sal = salary_by_department(work)
    if not sal.empty:
        st.markdown("### Salary Analysis")
        fig = px.bar(
            sal,
            x="department",
            y="Average Salary",
            title="Average Salary by Department",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("salary_department"),
        )

    perf = performance_by_department(work)
    if not perf.empty:
        st.markdown("### Performance Analysis")
        fig = px.bar(
            perf,
            x="department",
            y="Average Performance",
            title="Average Performance by Department",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("performance_department"),
        )

    attr = attrition_by_department(work)
    if not attr.empty:
        st.markdown("### Attrition Analysis")
        fig = px.bar(
            attr,
            x="department",
            y="Attrition Rate",
            title="Attrition Rate by Department",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("attrition_department"),
        )

    gend = gender_distribution(work)
    if not gend.empty:
        st.markdown("### Gender Demographics")
        fig = px.pie(
            gend,
            names="Gender",
            values="Employees",
            title="Employee Gender Distribution",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("gender_distribution"),
        )

    roles = role_distribution(work)
    if not roles.empty:
        st.markdown("### Role Distribution")
        fig = px.bar(
            roles.head(20),
            x="Job Role",
            y="Employees",
            title="Employee Distribution by Job Role",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("role_distribution"),
        )

    att = attendance_by_department(work)
    if not att.empty:
        st.markdown("### Attendance Analysis")
        fig = px.bar(
            att,
            x="department",
            y="Average Attendance",
            title="Average Attendance by Department",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key("attendance_department"),
        )

    insights = generate_hr_insights(work)
    if insights:
        st.markdown("### HR Insights")
        for insight in insights:
            st.info(insight)

    with st.expander("View Prepared HR Data", expanded=False):
        st.dataframe(work, use_container_width=True, hide_index=True)
