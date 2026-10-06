import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.healthcare.engine import (
    prepare_healthcare_data,
    calculate_healthcare_kpis,
    patients_by_department,
    diagnosis_distribution,
    procedure_distribution,
    charges_by_department,
    stay_by_department,
    readmission_by_department,
    age_distribution,
    gender_distribution,
    outcome_distribution,
    generate_healthcare_insights,
)


def render_healthcare_dashboard(df):
    render_id = uuid.uuid4().hex[:12]

    work, fields = prepare_healthcare_data(df)
    kpis = calculate_healthcare_kpis(work)

    st.markdown("## 🏥 Healthcare Analytics")

    st.caption(
        "Healthcare operations and patient-record analytics "
        "covering utilization, length of stay, charges, "
        "readmissions, outcomes, and demographics."
    )

    st.warning(
        "This workspace is for analytics and operational reporting. "
        "It does not provide medical diagnosis or treatment advice."
    )

    with st.expander(
        "Detected Healthcare Fields",
        expanded=False,
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
                        "Healthcare Field",
                        "Source Column",
                    ],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "No healthcare-specific fields were detected."
            )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Patients",
        f"{kpis['patients']:,}",
    )

    c2.metric(
        "Average Age",
        "N/A"
        if kpis["average_age"] is None
        else f"{kpis['average_age']:.1f}",
    )

    c3.metric(
        "Average Stay",
        "N/A"
        if kpis["average_stay"] is None
        else f"{kpis['average_stay']:.2f} days",
    )

    c4.metric(
        "Readmission Rate",
        f"{kpis['readmission_rate']:.2f}%",
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Total Charges",
        "N/A"
        if kpis["total_charges"] is None
        else f"{kpis['total_charges']:,.2f}",
    )

    c6.metric(
        "Avg Charges",
        "N/A"
        if kpis["average_charges"] is None
        else f"{kpis['average_charges']:,.2f}",
    )

    c7.metric(
        "Mortality Indicator",
        f"{kpis['mortality_rate']:.2f}%",
    )

    def chart_key(name):
        return (
            f"healthcare_"
            f"{render_id}_"
            f"{name}"
        )

    # =========================================================
    # PATIENTS BY DEPARTMENT
    # =========================================================

    dep = patients_by_department(work)

    if not dep.empty:
        st.markdown(
            "### Patients by Department"
        )

        fig = px.bar(
            dep,
            x="department",
            y="Patients",
            title="Patient Records by Department",
            labels={
                "department": "Department",
                "Patients": "Patients",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "patients_department"
            ),
        )

    # =========================================================
    # DIAGNOSIS
    # =========================================================

    diagnosis = diagnosis_distribution(work)

    if not diagnosis.empty:
        st.markdown(
            "### Diagnosis Distribution"
        )

        fig = px.bar(
            diagnosis.head(15),
            x="Diagnosis",
            y="Patients",
            title="Top Recorded Diagnoses",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "diagnosis"
            ),
        )

    # =========================================================
    # PROCEDURES
    # =========================================================

    procedures = procedure_distribution(work)

    if not procedures.empty:
        st.markdown(
            "### Procedure Distribution"
        )

        fig = px.bar(
            procedures.head(15),
            x="Procedure",
            y="Patients",
            title="Top Recorded Procedures",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "procedures"
            ),
        )

    # =========================================================
    # CHARGES
    # =========================================================

    charges = charges_by_department(work)

    if not charges.empty:
        st.markdown(
            "### Average Charges by Department"
        )

        fig = px.bar(
            charges,
            x="department",
            y="Average Charges",
            title="Average Healthcare Charges by Department",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "charges_department"
            ),
        )

    # =========================================================
    # LENGTH OF STAY
    # =========================================================

    stay = stay_by_department(work)

    if not stay.empty:
        st.markdown(
            "### Length of Stay"
        )

        fig = px.bar(
            stay,
            x="department",
            y="Average Stay",
            title="Average Length of Stay by Department",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "stay_department"
            ),
        )

    # =========================================================
    # READMISSION
    # =========================================================

    readmission = readmission_by_department(work)

    if not readmission.empty:
        st.markdown(
            "### Readmission Analysis"
        )

        fig = px.bar(
            readmission,
            x="department",
            y="Readmission Rate",
            title="Readmission Rate by Department",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "readmission_department"
            ),
        )

    # =========================================================
    # AGE
    # =========================================================

    ages = age_distribution(work)

    if not ages.empty:
        st.markdown(
            "### Patient Age Distribution"
        )

        fig = px.histogram(
            ages,
            x="age",
            nbins=20,
            title="Patient Age Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "age_distribution"
            ),
        )

    # =========================================================
    # GENDER
    # =========================================================

    gender = gender_distribution(work)

    if not gender.empty:
        st.markdown(
            "### Patient Demographics"
        )

        fig = px.pie(
            gender,
            names="Gender",
            values="Patients",
            title="Gender Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "gender_distribution"
            ),
        )

    # =========================================================
    # OUTCOME
    # =========================================================

    outcomes = outcome_distribution(work)

    if not outcomes.empty:
        st.markdown(
            "### Patient Outcomes"
        )

        fig = px.bar(
            outcomes,
            x="Outcome",
            y="Patients",
            title="Recorded Patient Outcomes",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key(
                "outcome_distribution"
            ),
        )

    # =========================================================
    # INSIGHTS
    # =========================================================

    insights = generate_healthcare_insights(
        work
    )

    if insights:
        st.markdown(
            "### Healthcare Analytics Insights"
        )

        for insight in insights:
            st.info(insight)

    # =========================================================
    # DATA
    # =========================================================

    with st.expander(
        "View Prepared Healthcare Data",
        expanded=False,
    ):
        st.dataframe(
            work,
            use_container_width=True,
            hide_index=True,
        )
