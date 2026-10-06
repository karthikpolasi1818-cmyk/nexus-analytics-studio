import pandas as pd
import numpy as np


ALIASES = {
    "patient_id": ["patient_id", "patientid", "patient_number", "patient_no"],
    "patient_name": ["patient_name", "patient", "name"],
    "age": ["age", "patient_age"],
    "gender": ["gender", "sex"],
    "admission_date": ["admission_date", "admit_date", "date_admitted", "admission"],
    "discharge_date": ["discharge_date", "discharge", "date_discharged"],
    "department": ["department", "dept", "ward", "unit"],
    "diagnosis": ["diagnosis", "condition", "disease", "primary_diagnosis"],
    "procedure": ["procedure", "treatment", "procedure_type"],
    "length_of_stay": ["length_of_stay", "los", "stay_days", "hospital_stay"],
    "charges": ["charges", "hospital_charges", "total_charges", "cost", "bill_amount"],
    "insurance": ["insurance", "insurance_provider", "payer"],
    "readmission": ["readmission", "readmitted", "readmission_flag"],
    "mortality": ["mortality", "death", "mortality_flag", "died"],
    "outcome": ["outcome", "disposition", "patient_outcome"],
    "satisfaction": ["satisfaction", "patient_satisfaction", "satisfaction_score"],
}


def _norm(value):
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _find_column(df, aliases):
    normalized = {_norm(c): c for c in df.columns}

    for alias in aliases:
        key = _norm(alias)
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        ncol = _norm(col)
        for alias in aliases:
            na = _norm(alias)
            if len(na) >= 4 and (
                ncol.startswith(na) or ncol.endswith(na)
            ):
                return col

    return None


def detect_healthcare_fields(df):
    return {
        key: _find_column(df, aliases)
        for key, aliases in ALIASES.items()
    }


def _binary_flag(series, positive_values):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        ).fillna(0).gt(0).astype(int)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(positive_values)
        .astype(int)
    )


def prepare_healthcare_data(df):
    work = df.copy()
    fields = detect_healthcare_fields(work)

    for logical, source in fields.items():
        if source and logical not in work.columns:
            work[logical] = work[source]

    numeric_columns = [
        "age",
        "length_of_stay",
        "charges",
        "satisfaction",
    ]

    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(
                work[column],
                errors="coerce"
            )

    for column in [
        "admission_date",
        "discharge_date",
    ]:
        if column in work.columns:
            work[column] = pd.to_datetime(
                work[column],
                errors="coerce"
            )

    if (
        "length_of_stay" not in work.columns
        and "admission_date" in work.columns
        and "discharge_date" in work.columns
    ):
        work["length_of_stay"] = (
            work["discharge_date"]
            - work["admission_date"]
        ).dt.days

    if "readmission" in work.columns:
        work["readmission_flag"] = _binary_flag(
            work["readmission"],
            {
                "yes", "y", "true", "1",
                "readmitted", "readmission"
            }
        )

    else:
        work["readmission_flag"] = 0

    if "mortality" in work.columns:
        work["mortality_flag"] = _binary_flag(
            work["mortality"],
            {
                "yes", "y", "true", "1",
                "dead", "died", "death"
            }
        )

    else:
        work["mortality_flag"] = 0

    return work, fields


def calculate_healthcare_kpis(df):
    patients = len(df)

    result = {
        "patients": patients,
        "average_age": None,
        "average_stay": None,
        "total_charges": None,
        "average_charges": None,
        "readmission_rate": 0.0,
        "mortality_rate": 0.0,
        "average_satisfaction": None,
    }

    if "age" in df.columns:
        result["average_age"] = float(
            df["age"].mean()
        )

    if "length_of_stay" in df.columns:
        result["average_stay"] = float(
            df["length_of_stay"].mean()
        )

    if "charges" in df.columns:
        result["total_charges"] = float(
            df["charges"].sum()
        )
        result["average_charges"] = float(
            df["charges"].mean()
        )

    if patients:
        if "readmission_flag" in df.columns:
            result["readmission_rate"] = float(
                df["readmission_flag"].mean() * 100
            )

        if "mortality_flag" in df.columns:
            result["mortality_rate"] = float(
                df["mortality_flag"].mean() * 100
            )

    if "satisfaction" in df.columns:
        result["average_satisfaction"] = float(
            df["satisfaction"].mean()
        )

    return result


def patients_by_department(df):
    if "department" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(
            "department",
            dropna=False
        )
        .size()
        .reset_index(
            name="Patients"
        )
        .sort_values(
            "Patients",
            ascending=False
        )
    )


def diagnosis_distribution(df):
    if "diagnosis" not in df.columns:
        return pd.DataFrame()

    return (
        df["diagnosis"]
        .astype(str)
        .value_counts()
        .rename_axis("Diagnosis")
        .reset_index(
            name="Patients"
        )
    )


def procedure_distribution(df):
    if "procedure" not in df.columns:
        return pd.DataFrame()

    return (
        df["procedure"]
        .astype(str)
        .value_counts()
        .rename_axis("Procedure")
        .reset_index(
            name="Patients"
        )
    )


def charges_by_department(df):
    if (
        "department" not in df.columns
        or "charges" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby(
            "department",
            dropna=False
        )["charges"]
        .mean()
        .reset_index(
            name="Average Charges"
        )
        .sort_values(
            "Average Charges",
            ascending=False
        )
    )


def stay_by_department(df):
    if (
        "department" not in df.columns
        or "length_of_stay" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby(
            "department",
            dropna=False
        )["length_of_stay"]
        .mean()
        .reset_index(
            name="Average Stay"
        )
        .sort_values(
            "Average Stay",
            ascending=False
        )
    )


def readmission_by_department(df):
    if "department" not in df.columns:
        return pd.DataFrame()

    if "readmission_flag" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(
            "department",
            dropna=False
        )["readmission_flag"]
        .mean()
        .mul(100)
        .reset_index(
            name="Readmission Rate"
        )
        .sort_values(
            "Readmission Rate",
            ascending=False
        )
    )


def age_distribution(df):
    if "age" not in df.columns:
        return pd.DataFrame()

    return df[["age"]].dropna()


def gender_distribution(df):
    if "gender" not in df.columns:
        return pd.DataFrame()

    return (
        df["gender"]
        .astype(str)
        .value_counts()
        .rename_axis("Gender")
        .reset_index(
            name="Patients"
        )
    )


def outcome_distribution(df):
    if "outcome" not in df.columns:
        return pd.DataFrame()

    return (
        df["outcome"]
        .astype(str)
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(
            name="Patients"
        )
    )


def generate_healthcare_insights(df):
    insights = []
    kpis = calculate_healthcare_kpis(df)

    insights.append(
        f"Analyzed {kpis['patients']:,} patient records."
    )

    if kpis["average_stay"] is not None:
        insights.append(
            f"Average length of stay is "
            f"{kpis['average_stay']:.2f} days."
        )

    if kpis["total_charges"] is not None:
        insights.append(
            f"Total recorded healthcare charges are "
            f"{kpis['total_charges']:,.2f}."
        )

    if "readmission_flag" in df.columns:
        insights.append(
            f"Observed readmission rate is "
            f"{kpis['readmission_rate']:.2f}%."
        )

    if "mortality_flag" in df.columns:
        insights.append(
            f"Observed mortality indicator rate is "
            f"{kpis['mortality_rate']:.2f}%."
        )

    dep = patients_by_department(df)

    if not dep.empty:
        row = dep.iloc[0]
        insights.append(
            f"{row['department']} has the largest "
            f"number of recorded patient visits "
            f"with {int(row['Patients']):,}."
        )

    return insights


def healthcare_summary(df):
    return {
        "kpis": calculate_healthcare_kpis(df),
        "patients_by_department": patients_by_department(df),
        "diagnosis_distribution": diagnosis_distribution(df),
        "procedure_distribution": procedure_distribution(df),
        "charges_by_department": charges_by_department(df),
        "stay_by_department": stay_by_department(df),
        "readmission_by_department": readmission_by_department(df),
        "age_distribution": age_distribution(df),
        "gender_distribution": gender_distribution(df),
        "outcome_distribution": outcome_distribution(df),
        "insights": generate_healthcare_insights(df),
    }
