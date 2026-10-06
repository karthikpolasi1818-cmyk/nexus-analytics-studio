import pandas as pd
import numpy as np


ALIASES = {
    "machine_id": ["machine_id", "machineid", "equipment_id", "asset_id"],
    "machine": ["machine", "machine_name", "equipment", "equipment_name"],
    "production_date": ["production_date", "date", "manufacturing_date", "timestamp"],
    "product": ["product", "product_name", "product_id", "item"],
    "line": ["line", "production_line", "assembly_line", "line_id"],
    "shift": ["shift", "work_shift"],
    "units_produced": ["units_produced", "production", "output", "units", "quantity_produced"],
    "target_units": ["target_units", "production_target", "target_output", "target"],
    "defects": ["defects", "defect_count", "defective_units", "scrap"],
    "quality_rate": ["quality_rate", "quality", "yield", "first_pass_yield"],
    "downtime": ["downtime", "downtime_hours", "machine_downtime", "idle_hours"],
    "maintenance_cost": ["maintenance_cost", "maintenance_expense", "repair_cost"],
    "production_cost": ["production_cost", "manufacturing_cost", "cost"],
    "energy": ["energy", "energy_consumption", "power_consumption", "kwh"],
    "labor_hours": ["labor_hours", "work_hours", "man_hours"],
    "cycle_time": ["cycle_time", "cycle_time_minutes", "production_cycle_time"],
    "status": ["status", "machine_status", "equipment_status"],
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
            if len(na) >= 4 and (ncol.startswith(na) or ncol.endswith(na)):
                return col

    return None


def detect_manufacturing_fields(df):
    return {
        key: _find_column(df, aliases)
        for key, aliases in ALIASES.items()
    }


def prepare_manufacturing_data(df):
    work = df.copy()
    fields = detect_manufacturing_fields(work)

    for logical, source in fields.items():
        if source and logical not in work.columns:
            work[logical] = work[source]

    numeric_columns = [
        "units_produced",
        "target_units",
        "defects",
        "quality_rate",
        "downtime",
        "maintenance_cost",
        "production_cost",
        "energy",
        "labor_hours",
        "cycle_time",
    ]

    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "production_date" in work.columns:
        work["production_date"] = pd.to_datetime(
            work["production_date"],
            errors="coerce",
        )

    if "quality_rate" not in work.columns:
        if "units_produced" in work.columns and "defects" in work.columns:
            total = work["units_produced"] + work["defects"]
            work["quality_rate"] = np.where(
                total.gt(0),
                work["units_produced"] / total * 100,
                np.nan,
            )

    if "target_units" in work.columns and "units_produced" in work.columns:
        work["production_variance"] = (
            work["units_produced"] - work["target_units"]
        )
        work["target_achievement"] = np.where(
            work["target_units"].gt(0),
            work["units_produced"] / work["target_units"] * 100,
            np.nan,
        )

    if "downtime" in work.columns:
        work["availability_indicator"] = 1 - (
            work["downtime"].fillna(0) /
            work["downtime"].fillna(0).add(24).replace(0, np.nan)
        )

    return work, fields


def calculate_manufacturing_kpis(df):
    result = {
        "records": len(df),
        "units_produced": None,
        "target_units": None,
        "target_achievement": None,
        "defects": None,
        "quality_rate": None,
        "downtime": None,
        "maintenance_cost": None,
        "production_cost": None,
        "energy": None,
        "labor_hours": None,
        "cycle_time": None,
    }

    if "units_produced" in df:
        result["units_produced"] = float(df["units_produced"].sum())

    if "target_units" in df:
        result["target_units"] = float(df["target_units"].sum())

    if result["target_units"]:
        result["target_achievement"] = (
            result["units_produced"] / result["target_units"] * 100
        )

    if "defects" in df:
        result["defects"] = float(df["defects"].sum())

    if "quality_rate" in df:
        result["quality_rate"] = float(df["quality_rate"].mean())

    if "downtime" in df:
        result["downtime"] = float(df["downtime"].sum())

    if "maintenance_cost" in df:
        result["maintenance_cost"] = float(df["maintenance_cost"].sum())

    if "production_cost" in df:
        result["production_cost"] = float(df["production_cost"].sum())

    if "energy" in df:
        result["energy"] = float(df["energy"].sum())

    if "labor_hours" in df:
        result["labor_hours"] = float(df["labor_hours"].sum())

    if "cycle_time" in df:
        result["cycle_time"] = float(df["cycle_time"].mean())

    return result


def production_by_product(df):
    if "product" not in df or "units_produced" not in df:
        return pd.DataFrame()

    return (
        df.groupby("product", dropna=False)["units_produced"]
        .sum()
        .reset_index(name="Units Produced")
        .sort_values("Units Produced", ascending=False)
    )


def production_by_line(df):
    if "line" not in df or "units_produced" not in df:
        return pd.DataFrame()

    return (
        df.groupby("line", dropna=False)["units_produced"]
        .sum()
        .reset_index(name="Units Produced")
        .sort_values("Units Produced", ascending=False)
    )


def machine_performance(df):
    if "machine" not in df:
        return pd.DataFrame()

    aggregations = {}
    if "units_produced" in df:
        aggregations["Units Produced"] = ("units_produced", "sum")
    if "defects" in df:
        aggregations["Defects"] = ("defects", "sum")
    if "downtime" in df:
        aggregations["Downtime"] = ("downtime", "sum")
    if "quality_rate" in df:
        aggregations["Quality Rate"] = ("quality_rate", "mean")

    if not aggregations:
        return pd.DataFrame()

    return (
        df.groupby("machine", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(
            "Units Produced" if "Units Produced" in aggregations else list(aggregations)[0],
            ascending=False,
        )
    )


def downtime_by_machine(df):
    if "machine" not in df or "downtime" not in df:
        return pd.DataFrame()

    return (
        df.groupby("machine", dropna=False)["downtime"]
        .sum()
        .reset_index(name="Downtime")
        .sort_values("Downtime", ascending=False)
    )


def defects_by_product(df):
    if "product" not in df or "defects" not in df:
        return pd.DataFrame()

    return (
        df.groupby("product", dropna=False)["defects"]
        .sum()
        .reset_index(name="Defects")
        .sort_values("Defects", ascending=False)
    )


def quality_by_line(df):
    if "line" not in df or "quality_rate" not in df:
        return pd.DataFrame()

    return (
        df.groupby("line", dropna=False)["quality_rate"]
        .mean()
        .reset_index(name="Quality Rate")
        .sort_values("Quality Rate", ascending=False)
    )


def production_trend(df):
    if "production_date" not in df or "units_produced" not in df:
        return pd.DataFrame()

    work = df.dropna(subset=["production_date"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["Date"] = work["production_date"].dt.date

    result = (
        work.groupby("Date")
        .agg(
            Units_Produced=("units_produced", "sum"),
            Defects=("defects", "sum") if "defects" in work else ("units_produced", "size"),
        )
        .reset_index()
    )

    return result


def cost_by_line(df):
    if "line" not in df:
        return pd.DataFrame()

    available = {}
    if "production_cost" in df:
        available["Production Cost"] = ("production_cost", "sum")
    if "maintenance_cost" in df:
        available["Maintenance Cost"] = ("maintenance_cost", "sum")

    if not available:
        return pd.DataFrame()

    return (
        df.groupby("line", dropna=False)
        .agg(**available)
        .reset_index()
    )


def generate_manufacturing_insights(df):
    kpis = calculate_manufacturing_kpis(df)
    insights = [
        f"Analyzed {kpis['records']:,} manufacturing records."
    ]

    if kpis["units_produced"] is not None:
        insights.append(
            f"Recorded production volume is {kpis['units_produced']:,.0f} units."
        )

    if kpis["target_achievement"] is not None:
        insights.append(
            f"Overall target achievement is {kpis['target_achievement']:.2f}%."
        )

    if kpis["quality_rate"] is not None:
        insights.append(
            f"Average quality rate is {kpis['quality_rate']:.2f}%."
        )

    if kpis["downtime"] is not None:
        insights.append(
            f"Recorded downtime totals {kpis['downtime']:,.2f} hours."
        )

    machines = downtime_by_machine(df)
    if not machines.empty:
        row = machines.iloc[0]
        insights.append(
            f"{row['machine']} has the highest recorded downtime at "
            f"{row['Downtime']:,.2f} hours."
        )

    return insights


def manufacturing_summary(df):
    return {
        "kpis": calculate_manufacturing_kpis(df),
        "production_by_product": production_by_product(df),
        "production_by_line": production_by_line(df),
        "machine_performance": machine_performance(df),
        "downtime_by_machine": downtime_by_machine(df),
        "defects_by_product": defects_by_product(df),
        "quality_by_line": quality_by_line(df),
        "production_trend": production_trend(df),
        "cost_by_line": cost_by_line(df),
        "insights": generate_manufacturing_insights(df),
    }
