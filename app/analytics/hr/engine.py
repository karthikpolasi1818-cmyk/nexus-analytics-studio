import pandas as pd
import numpy as np


ALIASES = {
    "employee_id": ["employee_id", "employeeid", "emp_id", "empid", "employee_number"],
    "employee_name": ["employee_name", "name", "employee", "emp_name"],
    "department": ["department", "dept", "division", "business_unit"],
    "job_role": ["job_role", "jobrole", "role", "job_title", "position", "designation"],
    "gender": ["gender", "sex"],
    "age": ["age"],
    "salary": ["salary", "base_salary", "annual_salary", "compensation", "pay"],
    "bonus": ["bonus", "annual_bonus", "incentive"],
    "performance": ["performance", "performance_score", "rating", "performance_rating"],
    "attendance": ["attendance", "attendance_rate", "attendance_percentage"],
    "absent_days": ["absent_days", "absence_days", "absences", "days_absent"],
    "leave_days": ["leave_days", "leaves", "days_leave", "paid_leave_days"],
    "experience": ["experience", "experience_years", "years_experience"],
    "joining_date": ["joining_date", "join_date", "date_joined", "hire_date"],
    "exit_date": ["exit_date", "termination_date", "date_left"],
    "attrition": ["attrition", "attrition_flag", "left", "churn"],
    "status": ["status", "employee_status"],
    "location": ["location", "city", "office", "work_location"],
}


def _norm(value):
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _find_column(df, aliases):
    normalized = {_norm(c): c for c in df.columns}

    # Exact match first.
    for alias in aliases:
        key = _norm(alias)
        if key in normalized:
            return normalized[key]

    # Only allow prefix/suffix matching to avoid collisions such as
    # customer_segment being mistaken for customer_id.
    for col in df.columns:
        ncol = _norm(col)
        for alias in aliases:
            na = _norm(alias)
            if len(na) >= 4 and (ncol.startswith(na) or ncol.endswith(na)):
                return col

    return None


def detect_hr_fields(df):
    return {
        key: _find_column(df, aliases)
        for key, aliases in ALIASES.items()
    }


def prepare_hr_data(df):
    work = df.copy()
    fields = detect_hr_fields(work)

    for logical, source in fields.items():
        if source and logical not in work.columns:
            work[logical] = work[source]

    for col in ["salary", "bonus", "age", "performance", "attendance",
                "absent_days", "leave_days", "experience"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in ["joining_date", "exit_date"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")

    return work, fields


def _attrition_rate(df):
    if "attrition" not in df.columns:
        return None
    s = df["attrition"]
    if pd.api.types.is_numeric_dtype(s):
        return float(pd.to_numeric(s, errors="coerce").mean() * 100)

    text = s.astype(str).str.strip().str.lower()
    yes = text.isin(["yes", "y", "true", "1", "left", "exited", "attrition"])
    valid = text.ne("") & text.ne("nan")
    if valid.sum() == 0:
        return None
    return float(yes[valid].mean() * 100)


def calculate_hr_kpis(df):
    out = {
        "employees": int(len(df)),
        "average_salary": None,
        "average_performance": None,
        "attrition_rate": _attrition_rate(df),
        "average_attendance": None,
    }

    if "salary" in df.columns:
        out["average_salary"] = float(df["salary"].mean())
    if "performance" in df.columns:
        out["average_performance"] = float(df["performance"].mean())
    if "attendance" in df.columns:
        out["average_attendance"] = float(df["attendance"].mean())

    return out


def employees_by_department(df):
    if "department" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("department", dropna=False)
        .size()
        .reset_index(name="Employees")
        .sort_values("Employees", ascending=False)
    )


def salary_by_department(df):
    if "department" not in df.columns or "salary" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("department", dropna=False)["salary"]
        .mean()
        .reset_index(name="Average Salary")
        .sort_values("Average Salary", ascending=False)
    )


def performance_by_department(df):
    if "department" not in df.columns or "performance" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("department", dropna=False)["performance"]
        .mean()
        .reset_index(name="Average Performance")
        .sort_values("Average Performance", ascending=False)
    )


def attrition_by_department(df):
    if "department" not in df.columns or "attrition" not in df.columns:
        return pd.DataFrame()

    work = df[["department", "attrition"]].copy()
    text = work["attrition"].astype(str).str.strip().str.lower()
    work["_left"] = text.isin(["yes", "y", "true", "1", "left", "exited", "attrition"]).astype(int)

    return (
        work.groupby("department", dropna=False)["_left"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
        .sort_values("Attrition Rate", ascending=False)
    )


def gender_distribution(df):
    if "gender" not in df.columns:
        return pd.DataFrame()
    return (
        df["gender"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Gender")
        .reset_index(name="Employees")
    )


def role_distribution(df):
    if "job_role" not in df.columns:
        return pd.DataFrame()
    return (
        df["job_role"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Job Role")
        .reset_index(name="Employees")
    )


def attendance_by_department(df):
    if "department" not in df.columns or "attendance" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("department", dropna=False)["attendance"]
        .mean()
        .reset_index(name="Average Attendance")
        .sort_values("Average Attendance", ascending=False)
    )


def generate_hr_insights(df):
    insights = []

    kpis = calculate_hr_kpis(df)

    if kpis["employees"]:
        insights.append(f"Workforce contains {kpis['employees']:,} employee records.")

    if kpis["average_salary"] is not None:
        insights.append(f"Average salary is {kpis['average_salary']:,.0f}.")

    if kpis["average_performance"] is not None:
        insights.append(f"Average performance score is {kpis['average_performance']:.2f}.")

    if kpis["attrition_rate"] is not None:
        insights.append(f"Overall attrition rate is {kpis['attrition_rate']:.2f}%.")

    dep = employees_by_department(df)
    if not dep.empty:
        top = dep.iloc[0]
        insights.append(
            f"{top['department']} has the largest recorded workforce "
            f"with {int(top['Employees']):,} employees."
        )

    att = attrition_by_department(df)
    if not att.empty:
        top = att.iloc[0]
        insights.append(
            f"{top['department']} has the highest observed attrition rate "
            f"at {top['Attrition Rate']:.2f}%."
        )

    return insights


def hr_summary(df):
    return {
        "kpis": calculate_hr_kpis(df),
        "headcount_by_department": employees_by_department(df),
        "salary_by_department": salary_by_department(df),
        "performance_by_department": performance_by_department(df),
        "attrition_by_department": attrition_by_department(df),
        "gender_distribution": gender_distribution(df),
        "role_distribution": role_distribution(df),
        "attendance_by_department": attendance_by_department(df),
        "insights": generate_hr_insights(df),
    }
