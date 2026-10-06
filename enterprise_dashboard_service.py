from __future__ import annotations

from typing import Any
import numpy as np
import re
import pandas as pd


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {_norm(c): str(c) for c in df.columns}

    # exact normalized match
    for alias in aliases:
        a = _norm(alias)
        if a in normalized:
            return normalized[a]

    # token-aware partial match
    for alias in aliases:
        tokens = set(_norm(alias).split())
        for norm_col, original in normalized.items():
            col_tokens = set(norm_col.split())
            if tokens and tokens.issubset(col_tokens):
                return original

    # substring fallback
    for alias in aliases:
        a = _norm(alias)
        for norm_col, original in normalized.items():
            if a and a in norm_col:
                return original

    return None


def _numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if not column:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _sum(df, col):
    s = _numeric(df, col).dropna()
    return float(s.sum()) if not s.empty else 0.0


def _mean(df, col):
    s = _numeric(df, col).dropna()
    return float(s.mean()) if not s.empty else 0.0


def _count_true(df, col, positive=("yes", "true", "1", "y")):
    if not col:
        return 0
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return int(s.fillna(False).sum())
    return int(s.astype(str).str.strip().str.lower().isin(positive).sum())


def _ratio(a, b):
    return float(a / b * 100) if b else 0.0


def _safe_records(frame: pd.DataFrame, label_col: str, value_col: str, limit=12):
    if label_col not in frame.columns or value_col not in frame.columns:
        return []
    work = frame[[label_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    work[label_col] = work[label_col].astype(str)
    work = work.groupby(label_col, as_index=False)[value_col].sum()
    work = work.sort_values(value_col, ascending=False).head(limit)
    return [
        {"label": str(row[label_col]), "value": float(row[value_col])}
        for _, row in work.iterrows()
    ]


def _group_sum(df, group_col, value_col, limit=12):
    if not group_col or not value_col:
        return []
    return _safe_records(df, group_col, value_col, limit)


def _trend(df, date_col, value_columns, label_names=None):
    if not date_col or not value_columns:
        return []

    work = df.copy()
    dates = pd.to_datetime(work[date_col], errors="coerce")
    valid = dates.notna()
    work = work.loc[valid].copy()
    dates = dates.loc[valid]

    if work.empty:
        return []

    work["_period"] = dates.dt.to_period("M").astype(str)
    group = work.groupby("_period", as_index=False)

    result = []
    for period, part in group:
        item = {"period": str(period)}
        for col, label in zip(value_columns, label_names or value_columns):
            values = pd.to_numeric(part[col], errors="coerce")
            item[str(label)] = float(values.sum()) if values.notna().any() else 0.0
        result.append(item)

    return result


def _auto_charts(df: pd.DataFrame, limit=12):
    charts = []

    date_col = _find(df, ["date", "datetime", "timestamp", "production date", "admission date", "joining date"])
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    categorical_cols = [
        str(c) for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and 1 < df[c].nunique(dropna=True) <= 30
    ]

    if date_col and numeric_cols:
        preferred = numeric_cols[:3]
        trend = _trend(df, date_col, preferred)
        for col in preferred:
            if trend:
                charts.append({
                    "type": "line",
                    "title": f"{col} Trend",
                    "label_key": "period",
                    "value_key": str(col),
                    "data": trend,
                })

    for cat in categorical_cols[:6]:
        for num_col in numeric_cols[:2]:
            records = _group_sum(df, cat, str(num_col))
            if records:
                charts.append({
                    "type": "bar",
                    "title": f"{num_col} by {cat}",
                    "label_key": "label",
                    "value_key": "value",
                    "data": records,
                })

    # Numeric distributions as compact ranked bins
    for col in numeric_cols[:2]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) >= 10 and series.nunique() > 5:
            try:
                bins = pd.qcut(series, q=6, duplicates="drop")
                counts = bins.value_counts().sort_index()
                data = [
                    {"label": str(idx), "value": int(value)}
                    for idx, value in counts.items()
                ]
                charts.append({
                    "type": "bar",
                    "title": f"{col} Distribution",
                    "label_key": "label",
                    "value_key": "value",
                    "data": data,
                })
            except Exception:
                pass

    # De-duplicate titles and cap response size.
    unique = []
    seen = set()
    for chart in charts:
        if chart["title"] not in seen:
            unique.append(chart)
            seen.add(chart["title"])

    return unique[:limit]


def _sales(df):
    sales = _find(df, ["sales", "revenue", "total sales", "amount", "net sales"])
    profit = _find(df, ["profit", "net profit", "gross profit"])
    order = _find(df, ["order id", "order_id", "order"])
    customer = _find(df, ["customer id", "customer_id", "customer"])
    quantity = _find(df, ["quantity", "units sold", "units"])
    date = _find(df, ["date", "order date", "sales date"])
    category = _find(df, ["category", "product category"])
    region = _find(df, ["region", "area", "zone"])
    product = _find(df, ["product", "product name"])
    channel = _find(df, ["channel", "sales channel"])

    revenue = _sum(df, sales)
    profit_value = _sum(df, profit)
    orders = int(df[order].nunique()) if order else len(df)
    customers = int(df[customer].nunique()) if customer else 0

    kpis = {
        "Revenue": revenue,
        "Profit": profit_value,
        "Orders": orders,
        "Customers": customers,
        "Quantity": _sum(df, quantity),
        "Profit Margin": _ratio(profit_value, revenue),
        "Average Order Value": revenue / orders if orders else 0,
    }

    tables = []
    if category and sales:
        tables.append({"title": "Category Performance", "data": _group_sum(df, category, sales)})
    if region and sales:
        tables.append({"title": "Regional Performance", "data": _group_sum(df, region, sales)})
    if product and sales:
        tables.append({"title": "Product Performance", "data": _group_sum(df, product, sales)})
    if channel and sales:
        tables.append({"title": "Channel Performance", "data": _group_sum(df, channel, sales)})

    trend = _trend(df, date, [x for x in [sales, profit] if x], ["Revenue", "Profit"])
    if trend:
        tables.append({"title": "Monthly Trend", "data": trend})

    return kpis, tables


def _manufacturing(df):
    units = _find(df, ["units produced", "units_produced", "production", "output"])
    target = _find(df, ["target units", "target_units", "production target"])
    defects = _find(df, ["defects", "defect count", "defect"])
    quality = _find(df, ["quality rate", "quality_rate", "quality"])
    downtime = _find(df, ["downtime", "downtime hours", "downtime_hours"])
    maint = _find(df, ["maintenance cost", "maintenance_cost"])
    cost = _find(df, ["production cost", "production_cost", "manufacturing cost"])
    energy = _find(df, ["energy", "energy consumption", "energy_consumption"])
    labor = _find(df, ["labor hours", "labor_hours"])
    cycle = _find(df, ["cycle time", "cycle_time"])
    date = _find(df, ["production date", "date"])
    product = _find(df, ["product"])
    machine = _find(df, ["machine", "machine id", "machine_id"])
    line = _find(df, ["line", "production line"])
    shift = _find(df, ["shift"])
    status = _find(df, ["status"])

    unit_total = _sum(df, units)
    defect_total = _sum(df, defects)
    target_total = _sum(df, target)
    downtime_total = _sum(df, downtime)
    cost_total = _sum(df, cost)

    if quality:
        quality_rate = _mean(df, quality)
    else:
        quality_rate = _ratio(unit_total - defect_total, unit_total)

    kpis = {
        "Units Produced": unit_total,
        "Target Units": target_total,
        "Defects": defect_total,
        "Defect Rate": _ratio(defect_total, unit_total),
        "Quality Rate": quality_rate,
        "Downtime": downtime_total,
        "Maintenance Cost": _sum(df, maint),
        "Production Cost": cost_total,
        "Energy": _sum(df, energy),
        "Labor Hours": _sum(df, labor),
        "Average Cycle Time": _mean(df, cycle),
        "Target Achievement": _ratio(unit_total, target_total),
    }

    tables = []
    if product and units:
        tables.append({"title": "Production by Product", "data": _group_sum(df, product, units)})
    if machine and units:
        tables.append({"title": "Machine Production", "data": _group_sum(df, machine, units)})
    if line and units:
        tables.append({"title": "Production by Line", "data": _group_sum(df, line, units)})
    if shift and units:
        tables.append({"title": "Production by Shift", "data": _group_sum(df, shift, units)})
    if status and units:
        tables.append({"title": "Production by Status", "data": _group_sum(df, status, units)})
    if date and units:
        trend = _trend(df, date, [x for x in [units, defects, downtime] if x], ["Units Produced", "Defects", "Downtime"])
        if trend:
            tables.append({"title": "Production Trend", "data": trend})

    return kpis, tables


def _hr(df):
    employee = _find(df, ["employee id", "employee_id", "employee"])
    salary = _find(df, ["salary", "base salary"])
    bonus = _find(df, ["bonus"])
    absent = _find(df, ["absent days", "absent_days", "absence days", "absences"])
    leave = _find(df, ["leave days", "leave_days"])
    performance = _find(df, ["performance", "performance score"])
    attendance = _find(df, ["attendance", "attendance rate"])
    attrition = _find(df, ["attrition", "left", "exited"])
    department = _find(df, ["department"])
    role = _find(df, ["job role", "job_role", "role"])
    location = _find(df, ["location"])
    date = _find(df, ["joining date", "joining_date", "date"])

    employees = int(df[employee].nunique()) if employee else len(df)
    attrition_count = _count_true(df, attrition, ("yes", "true", "1", "y", "left", "exited"))

    kpis = {
        "Employees": employees,
        "Attrition": attrition_count,
        "Attrition Rate": _ratio(attrition_count, employees),
        "Average Salary": _mean(df, salary),
        "Total Salary Cost": _sum(df, salary),
        "Total Bonus": _sum(df, bonus),
        "Average Performance": _mean(df, performance),
        "Average Attendance": _mean(df, attendance),
        "Absence Days": _sum(df, absent),
        "Leave Days": _sum(df, leave),
    }

    tables = []
    for label, col, value in [
        ("Salary by Department", department, salary),
        ("Employees by Department", department, employee),
        ("Performance by Department", department, performance),
        ("Salary by Job Role", role, salary),
        ("Employees by Location", location, employee),
    ]:
        if col and value:
            if label.startswith("Employees"):
                counts = df.groupby(col).size().sort_values(ascending=False).head(12)
                data = [{"label": str(k), "value": int(v)} for k, v in counts.items()]
            else:
                data = _group_sum(df, col, value)
            tables.append({"title": label, "data": data})

    if date and salary:
        trend = _trend(df, date, [salary], ["Salary"])
        if trend:
            tables.append({"title": "Joining / Salary Trend", "data": trend})

    return kpis, tables


def _healthcare(df):
    patient = _find(df, ["patient id", "patient_id", "patient"])
    charges = _find(df, ["charges", "cost", "bill", "billing"])
    stay = _find(df, ["length of stay", "length_of_stay", "los"])
    age = _find(df, ["age"])
    satisfaction = _find(df, ["satisfaction", "satisfaction score"])
    readmission = _find(df, ["readmission", "readmitted"])
    mortality = _find(df, ["mortality", "death", "deceased"])
    department = _find(df, ["department", "ward"])
    diagnosis = _find(df, ["diagnosis", "condition"])
    procedure = _find(df, ["procedure", "treatment"])
    insurance = _find(df, ["insurance", "payer"])
    date = _find(df, ["admission date", "admission_date", "date"])

    patients = int(df[patient].nunique()) if patient else len(df)
    readmitted = _count_true(df, readmission)
    deaths = _count_true(df, mortality)

    kpis = {
        "Patients": patients,
        "Total Charges": _sum(df, charges),
        "Average Charges": _mean(df, charges),
        "Average Length of Stay": _mean(df, stay),
        "Readmissions": readmitted,
        "Readmission Rate": _ratio(readmitted, patients),
        "Mortality": deaths,
        "Mortality Rate": _ratio(deaths, patients),
        "Average Age": _mean(df, age),
        "Satisfaction": _mean(df, satisfaction),
    }

    tables = []
    for label, col, value in [
        ("Charges by Department", department, charges),
        ("Patients by Department", department, patient),
        ("Charges by Diagnosis", diagnosis, charges),
        ("Procedures", procedure, patient),
        ("Insurance Performance", insurance, charges),
    ]:
        if col and value:
            if value == patient:
                counts = df.groupby(col).size().sort_values(ascending=False).head(12)
                data = [{"label": str(k), "value": int(v)} for k, v in counts.items()]
            else:
                data = _group_sum(df, col, value)
            tables.append({"title": label, "data": data})

    if date and charges:
        trend = _trend(df, date, [charges], ["Charges"])
        if trend:
            tables.append({"title": "Healthcare Charges Trend", "data": trend})

    return kpis, tables


def _marketing(df):
    spend = _find(df, ["spend", "marketing spend", "ad spend", "cost"])
    revenue = _find(df, ["revenue", "sales", "generated revenue"])
    clicks = _find(df, ["clicks", "click"])
    impressions = _find(df, ["impressions", "impression"])
    conversions = _find(df, ["conversions", "conversion"])
    channel = _find(df, ["channel", "marketing channel"])
    campaign = _find(df, ["campaign", "campaign name"])
    date = _find(df, ["date", "campaign date"])

    spend_total = _sum(df, spend)
    revenue_total = _sum(df, revenue)
    clicks_total = _sum(df, clicks)
    impressions_total = _sum(df, impressions)
    conversion_total = _sum(df, conversions)

    kpis = {
        "Spend": spend_total,
        "Revenue": revenue_total,
        "ROAS": revenue_total / spend_total if spend_total else 0,
        "CTR": _ratio(clicks_total, impressions_total),
        "Conversion Rate": _ratio(conversion_total, clicks_total),
        "Clicks": clicks_total,
        "Impressions": impressions_total,
        "Conversions": conversion_total,
    }

    tables = []
    if channel and revenue:
        tables.append({"title": "Revenue by Channel", "data": _group_sum(df, channel, revenue)})
    if channel and spend:
        tables.append({"title": "Spend by Channel", "data": _group_sum(df, channel, spend)})
    if campaign and revenue:
        tables.append({"title": "Campaign Revenue", "data": _group_sum(df, campaign, revenue)})
    if date and revenue:
        trend = _trend(df, date, [x for x in [spend, revenue, conversions] if x], ["Spend", "Revenue", "Conversions"])
        if trend:
            tables.append({"title": "Marketing Trend", "data": trend})

    return kpis, tables


def _generic(df):
    numeric = list(df.select_dtypes(include=np.number).columns)
    kpis = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Numeric Columns": len(numeric),
        "Categorical Columns": sum(
            not pd.api.types.is_numeric_dtype(df[c])
            for c in df.columns
        ),
    }
    tables = []
    for col in numeric[:6]:
        tables.append({
            "title": f"{col} Summary",
            "data": [
                {"label": "Sum", "value": _sum(df, str(col))},
                {"label": "Average", "value": _mean(df, str(col))},
            ],
        })
    return kpis, tables


def build_enterprise_dashboard(
    df: pd.DataFrame,
    module: str,
) -> dict[str, Any]:
    module_lower = str(module).lower()

    if "manufacturing" in module_lower:
        kpis, tables = _manufacturing(df)
    elif module_lower.startswith("hr"):
        kpis, tables = _hr(df)
    elif "healthcare" in module_lower:
        kpis, tables = _healthcare(df)
    elif "marketing" in module_lower:
        kpis, tables = _marketing(df)
    elif "sales" in module_lower or "e-commerce" in module_lower or "ecommerce" in module_lower:
        kpis, tables = _sales(df)
    else:
        kpis, tables = _generic(df)

    charts = []
    for table in tables:
        if table.get("data"):
            charts.append({
                "type": "bar",
                "title": table["title"],
                "label_key": "label" if "label" in table["data"][0] else "period",
                "value_key": "value" if "value" in table["data"][0] else next(
                    (k for k in table["data"][0] if k not in {"label", "period"}), "value"
                ),
                "data": table["data"],
            })

    # Add automatic charts only when specialized charts are insufficient.
    if len(charts) < 4:
        auto = _auto_charts(df, limit=12)
        seen = {x["title"] for x in charts}
        for chart in auto:
            if chart["title"] not in seen:
                charts.append(chart)
                seen.add(chart["title"])

    quality_flags = []
    if df.isna().sum().sum():
        quality_flags.append("Missing values are present.")
    if df.duplicated().sum():
        quality_flags.append("Duplicate rows are present.")
    if not quality_flags:
        quality_flags.append("No basic missing-value or duplicate-row flags detected.")

    return {
        "domain": module,
        "kpis": kpis,
        "tables": tables,
        "charts": charts[:16],
        "insights": quality_flags,
        "columns_detected": {
            "count": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
        },
    }
