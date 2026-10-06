from __future__ import annotations

import pandas as pd


COLUMN_ALIASES = {
    "operation": ["operation", "process", "process_name", "activity"],
    "department": ["department", "dept", "business_unit"],
    "date": ["date", "operation_date", "transaction_date", "timestamp"],
    "output": ["output", "production", "units_produced", "quantity_produced"],
    "input": ["input", "input_units", "raw_material", "material_used"],
    "cost": ["cost", "operating_cost", "operation_cost", "total_cost"],
    "revenue": ["revenue", "sales", "income", "output_value"],
    "labor_hours": ["labor_hours", "labour_hours", "work_hours", "hours_worked"],
    "downtime": ["downtime", "downtime_hours", "idle_hours", "stoppage_hours"],
    "cycle_time": ["cycle_time", "processing_time", "production_time", "turnaround_time"],
    "defects": ["defects", "defect_count", "defective_units", "rejected_units"],
    "quality_rate": ["quality_rate", "quality", "yield_rate", "first_pass_yield"],
    "efficiency": ["efficiency", "operational_efficiency", "productivity"],
    "target": ["target", "planned_output", "target_output", "planned_units"],
    "actual": ["actual", "actual_output", "completed_units"],
    "employees": ["employees", "workers", "headcount", "staff"],
}


def _normalize(value) -> str:
    return (
        str(value).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(columns, aliases):
    normalized = {_normalize(c): c for c in columns}

    for alias in aliases:
        key = _normalize(alias)
        if key in normalized:
            return normalized[key]

    for column in columns:
        normalized_column = _normalize(column)
        for alias in aliases:
            key = _normalize(alias)
            if (
                normalized_column.startswith(f"{key}_")
                or normalized_column.endswith(f"_{key}")
            ):
                return column

    return None


def detect_operations_columns(df: pd.DataFrame) -> dict:
    return {
        field: find_column(df.columns, aliases)
        for field, aliases in COLUMN_ALIASES.items()
    }


def prepare_operations_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = df.copy()
    columns = detect_operations_columns(data)

    numeric_fields = [
        "output", "input", "cost", "revenue", "labor_hours",
        "downtime", "cycle_time", "defects", "quality_rate",
        "efficiency", "target", "actual", "employees",
    ]

    for field in numeric_fields:
        column = columns.get(field)
        if column:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    date_column = columns.get("date")
    if date_column:
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )

    # Derive quality rate from output and defects when needed.
    if not columns.get("quality_rate"):
        output_column = columns.get("output") or columns.get("actual")
        defect_column = columns.get("defects")

        if output_column and defect_column:
            denominator = data[output_column].fillna(0)
            data["_derived_quality_rate"] = (
                ((denominator - data[defect_column].fillna(0))
                 / denominator.replace(0, pd.NA))
                * 100
            )
            data["_derived_quality_rate"] = pd.to_numeric(
                data["_derived_quality_rate"],
                errors="coerce",
            )
            columns["quality_rate"] = "_derived_quality_rate"

    # Derive efficiency from actual/target when needed.
    if not columns.get("efficiency"):
        actual_column = columns.get("actual") or columns.get("output")
        target_column = columns.get("target")

        if actual_column and target_column:
            data["_derived_efficiency"] = (
                data[actual_column].fillna(0)
                / data[target_column].replace(0, pd.NA)
                * 100
            )
            data["_derived_efficiency"] = pd.to_numeric(
                data["_derived_efficiency"],
                errors="coerce",
            )
            columns["efficiency"] = "_derived_efficiency"

    return data, columns


def _sum(df, column):
    if not column or column not in df.columns:
        return 0.0
    return float(
        pd.to_numeric(df[column], errors="coerce")
        .fillna(0)
        .sum()
    )


def _mean(df, column):
    if not column or column not in df.columns:
        return 0.0
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()
    return float(values.mean()) if not values.empty else 0.0


def calculate_operations_kpis(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> dict:
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation_column = columns.get("operation")
    department_column = columns.get("department")

    operations = (
        int(data[operation_column].nunique())
        if operation_column
        else len(data)
    )

    departments = (
        int(data[department_column].nunique())
        if department_column
        else 0
    )

    output_column = columns.get("output") or columns.get("actual")

    output = _sum(data, output_column)
    cost = _sum(data, columns.get("cost"))
    revenue = _sum(data, columns.get("revenue"))
    labor_hours = _sum(data, columns.get("labor_hours"))
    downtime = _sum(data, columns.get("downtime"))
    defects = _sum(data, columns.get("defects"))

    quality = _mean(data, columns.get("quality_rate"))
    efficiency = _mean(data, columns.get("efficiency"))
    cycle_time = _mean(data, columns.get("cycle_time"))

    productivity = (
        output / labor_hours
        if labor_hours
        else 0.0
    )

    utilization = (
        (labor_hours - downtime) / labor_hours * 100
        if labor_hours
        else 0.0
    )

    return {
        "operations": operations,
        "departments": departments,
        "output": output,
        "cost": cost,
        "revenue": revenue,
        "labor_hours": labor_hours,
        "downtime": downtime,
        "defects": defects,
        "quality_rate": quality,
        "efficiency": efficiency,
        "cycle_time": cycle_time,
        "productivity": productivity,
        "utilization": utilization,
    }


def output_by_operation(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation = columns.get("operation")
    output = columns.get("output") or columns.get("actual")

    if not operation or not output:
        return pd.DataFrame()

    return (
        data.groupby(operation, dropna=False)[output]
        .sum()
        .reset_index(name="Output")
        .sort_values("Output", ascending=False)
    )


def cost_by_department(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    department = columns.get("department")
    cost = columns.get("cost")

    if not department or not cost:
        return pd.DataFrame()

    return (
        data.groupby(department, dropna=False)[cost]
        .sum()
        .reset_index(name="Cost")
        .sort_values("Cost", ascending=False)
    )


def efficiency_by_operation(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation = columns.get("operation")
    efficiency = columns.get("efficiency")

    if not operation or not efficiency:
        return pd.DataFrame()

    return (
        data.groupby(operation, dropna=False)[efficiency]
        .mean()
        .reset_index(name="Efficiency")
        .sort_values("Efficiency", ascending=False)
    )


def downtime_by_operation(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation = columns.get("operation")
    downtime = columns.get("downtime")

    if not operation or not downtime:
        return pd.DataFrame()

    return (
        data.groupby(operation, dropna=False)[downtime]
        .sum()
        .reset_index(name="Downtime")
        .sort_values("Downtime", ascending=False)
    )


def quality_by_operation(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation = columns.get("operation")
    quality = columns.get("quality_rate")

    if not operation or not quality:
        return pd.DataFrame()

    return (
        data.groupby(operation, dropna=False)[quality]
        .mean()
        .reset_index(name="Quality Rate")
        .sort_values("Quality Rate", ascending=False)
    )


def operational_trend(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    date = columns.get("date")
    output = columns.get("output") or columns.get("actual")

    if not date or not output:
        return pd.DataFrame()

    data = data.dropna(subset=[date]).copy()
    data["Period"] = data[date].dt.to_period("M").astype(str)

    result = (
        data.groupby("Period")
        .agg(
            Output=(output, "sum"),
            Downtime=(
                columns["downtime"],
                "sum",
            ) if columns.get("downtime") else (output, lambda x: 0),
            Efficiency=(
                columns["efficiency"],
                "mean",
            ) if columns.get("efficiency") else (output, lambda x: 0),
        )
        .reset_index()
    )

    return result.sort_values("Period")


def operation_performance(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected

    operation = columns.get("operation")
    if not operation:
        return pd.DataFrame()

    aggregation = {}

    output = columns.get("output") or columns.get("actual")
    if output:
        aggregation["Output"] = (output, "sum")

    if columns.get("cost"):
        aggregation["Cost"] = (columns["cost"], "sum")

    if columns.get("downtime"):
        aggregation["Downtime"] = (
            columns["downtime"],
            "sum",
        )

    if columns.get("quality_rate"):
        aggregation["Quality Rate"] = (
            columns["quality_rate"],
            "mean",
        )

    if columns.get("efficiency"):
        aggregation["Efficiency"] = (
            columns["efficiency"],
            "mean",
        )

    if columns.get("cycle_time"):
        aggregation["Cycle Time"] = (
            columns["cycle_time"],
            "mean",
        )

    if not aggregation:
        return pd.DataFrame()

    return (
        data.groupby(operation, dropna=False)
        .agg(**aggregation)
        .reset_index()
        .sort_values(
            "Output" if "Output" in aggregation else list(aggregation)[0],
            ascending=False,
        )
    )


def generate_operations_insights(df, columns=None):
    data, detected = prepare_operations_data(df)
    columns = columns or detected
    insights = []

    kpis = calculate_operations_kpis(
        data,
        columns,
    )

    if kpis["efficiency"]:
        insights.append(
            f"Average operational efficiency is "
            f"{kpis['efficiency']:.2f}%."
        )

    if kpis["utilization"]:
        insights.append(
            f"Labor utilization is "
            f"{kpis['utilization']:.2f}%."
        )

    if kpis["quality_rate"]:
        insights.append(
            f"Average quality rate is "
            f"{kpis['quality_rate']:.2f}%."
        )

    if kpis["downtime"]:
        insights.append(
            f"Total recorded downtime is "
            f"{kpis['downtime']:,.2f} hours."
        )

    if kpis["productivity"]:
        insights.append(
            f"Output productivity is "
            f"{kpis['productivity']:,.2f} units per labor hour."
        )

    performance = operation_performance(
        data,
        columns,
    )

    operation = columns.get("operation")

    if (
        not performance.empty
        and operation
        and "Output" in performance.columns
    ):
        top = performance.iloc[0]
        insights.append(
            f"Highest-output operation: "
            f"{top[operation]} "
            f"({top['Output']:,.0f} units)."
        )

    downtime = downtime_by_operation(
        data,
        columns,
    )

    if (
        not downtime.empty
        and operation
    ):
        worst = downtime.iloc[0]
        insights.append(
            f"Highest recorded downtime: "
            f"{worst[operation]} "
            f"({worst['Downtime']:,.2f} hours)."
        )

    if not insights:
        insights.append(
            "Add operation, output, efficiency, "
            "downtime, cost, or quality fields "
            "for deeper operations analysis."
        )

    return insights
