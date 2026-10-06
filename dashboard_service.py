from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


DOMAIN_RULES = {
    "Sales Analytics": [
        "sales", "revenue", "profit", "quantity", "order", "region",
        "product", "customer", "discount"
    ],
    "Customer Analytics": [
        "customer", "customer_id", "customer segment", "segment",
        "retention", "churn", "lifetime value", "clv"
    ],
    "Marketing Analytics": [
        "campaign", "impressions", "clicks", "conversions", "spend",
        "ctr", "conversion_rate", "roas", "cpc", "marketing"
    ],
    "Financial Analytics": [
        "revenue", "expense", "cost", "profit", "margin", "ebitda",
        "cash flow", "cashflow", "asset", "liability"
    ],
    "Supply Chain Analytics": [
        "inventory", "stock", "supplier", "lead time", "warehouse",
        "shipment", "delivery", "fill rate", "stockout", "shipping"
    ],
    "Product Analytics": [
        "product", "sku", "rating", "review", "return", "units sold",
        "feature", "product category"
    ],
    "Operations Analytics": [
        "operation", "cycle time", "throughput", "efficiency",
        "downtime", "utilization", "process"
    ],
    "HR Analytics": [
        "employee", "employee id", "department", "salary", "attrition",
        "hire", "hiring", "absence", "tenure", "performance"
    ],
    "Fraud Analytics": [
        "fraud", "transaction", "suspicious", "risk", "chargeback",
        "anomaly", "fraud flag"
    ],
    "Healthcare Analytics": [
        "patient", "diagnosis", "admission", "discharge", "treatment",
        "doctor", "hospital", "medical", "disease"
    ],
    "Manufacturing Analytics": [
        "production", "units produced", "machine", "defect", "downtime",
        "utilization", "plant", "factory", "production cost", "quality"
    ],
    "E-commerce Analytics": [
        "order", "product", "customer", "cart", "conversion",
        "return", "shipping", "payment", "gmv", "aov"
    ],
}


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_norm(c): str(c) for c in df.columns}

    for candidate in candidates:
        key = _norm(candidate)
        if key in normalized:
            return normalized[key]

    for column in df.columns:
        c = _norm(column)
        for candidate in candidates:
            k = _norm(candidate)
            if k in c or c in k:
                return str(column)

    return None


def _number(value: Any) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for record in frame.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[str(key)] = None
            elif isinstance(value, np.integer):
                cleaned[str(key)] = int(value)
            elif isinstance(value, np.floating):
                cleaned[str(key)] = float(value)
            elif isinstance(value, pd.Timestamp):
                cleaned[str(key)] = value.isoformat()
            else:
                cleaned[str(key)] = value
        result.append(cleaned)
    return result


def _detect_domain(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [_norm(c) for c in df.columns]
    rankings = []

    for domain, keywords in DOMAIN_RULES.items():
        matched = []
        for keyword in keywords:
            key = _norm(keyword)
            if any(key == c or key in c or c in key for c in columns):
                matched.append(keyword)

        if matched:
            rankings.append({
                "module": domain,
                "score": len(set(matched)),
                "matched_fields": matched,
            })

    rankings.sort(key=lambda x: x["score"], reverse=True)

    if not rankings:
        return [{"module": "General Analytics", "score": 0, "matched_fields": []}]

    return rankings


def _quality(df: pd.DataFrame) -> dict[str, Any]:
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "completeness": max(0.0, 100.0 - (missing / total_cells * 100.0)),
        "missing_cells": missing,
        "duplicate_rows": duplicates,
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_columns": int(
            df.select_dtypes(include=["object", "category", "string"]).shape[1]
        ),
        "date_columns": int(
            sum(
                pd.api.types.is_datetime64_any_dtype(df[c])
                for c in df.columns
            )
        ),
    }


def _group_performance(
    frame: pd.DataFrame,
    group_col: str | None,
    value_col: str | None,
    secondary_col: str | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    if not group_col or not value_col:
        return []

    work = frame.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)

    aggregations = {"value": (value_col, "sum")}

    if secondary_col:
        work[secondary_col] = pd.to_numeric(
            work[secondary_col], errors="coerce"
        ).fillna(0)
        aggregations["secondary"] = (secondary_col, "sum")

    grouped = (
        work.groupby(group_col, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("value", ascending=False)
        .head(limit)
    )

    records = []
    for _, row in grouped.iterrows():
        item = {
            "name": str(row[group_col]),
            "value": _number(row["value"]),
        }
        if secondary_col:
            item["secondary"] = _number(row["secondary"])
        records.append(item)

    return records


def _trend(
    frame: pd.DataFrame,
    date_col: str | None,
    value_columns: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not date_col or not value_columns:
        return []

    work = frame.copy()
    work["_nexus_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["_nexus_date"])

    if work.empty:
        return []

    work["_period"] = work["_nexus_date"].dt.to_period("M").astype(str)

    aggregation = {}
    for column, output in value_columns:
        if column:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
            aggregation[output] = (column, "sum")

    if not aggregation:
        return []

    grouped = (
        work.groupby("_period", as_index=False)
        .agg(**aggregation)
        .sort_values("_period")
    )

    return _safe_records(grouped)


def _first_numeric(df: pd.DataFrame, candidates: list[str]) -> str | None:
    col = _find_column(df, candidates)
    if col is not None:
        return col

    numeric = df.select_dtypes(include=np.number).columns
    return str(numeric[0]) if len(numeric) else None


def _domain_dashboard(
    frame: pd.DataFrame,
    domain: str,
    generic: dict[str, Any],
) -> dict[str, Any]:
    date_col = _find_column(
        frame,
        ["Date", "Order Date", "Transaction Date", "Campaign Date",
         "Production Date", "Hire Date", "Admission Date"]
    )

    charts: list[dict[str, Any]] = []
    insights: list[str] = []

    if domain == "Marketing Analytics":
        spend = _first_numeric(frame, ["Spend", "Ad Spend", "Marketing Spend", "Cost"])
        revenue = _first_numeric(frame, ["Revenue", "Sales"])
        impressions = _first_numeric(frame, ["Impressions"])
        clicks = _first_numeric(frame, ["Clicks"])
        conversions = _first_numeric(frame, ["Conversions"])
        ctr = _first_numeric(frame, ["CTR", "Click Through Rate"])
        conversion_rate = _first_numeric(frame, ["Conversion Rate", "Conversion_Rate"])
        roas = _first_numeric(frame, ["ROAS"])
        channel = _find_column(frame, ["Channel", "Platform", "Source"])
        campaign = _find_column(frame, ["Campaign", "Campaign Name"])

        total_spend = _number(frame[spend].sum()) if spend else 0
        total_revenue = _number(frame[revenue].sum()) if revenue else 0
        total_clicks = _number(frame[clicks].sum()) if clicks else 0
        total_conversions = _number(frame[conversions].sum()) if conversions else 0

        kpis = {
            "spend": total_spend,
            "revenue": total_revenue,
            "clicks": total_clicks,
            "impressions": _number(frame[impressions].sum()) if impressions else 0,
            "conversions": total_conversions,
            "ctr": (
                total_clicks / _number(frame[impressions].sum()) * 100
                if impressions and _number(frame[impressions].sum()) else 0
            ),
            "conversion_rate": (
                total_conversions / total_clicks * 100
                if total_clicks else 0
            ),
            "roas": total_revenue / total_spend if total_spend else 0,
        }

        channel_perf = _group_performance(frame, channel, revenue, spend)
        campaign_perf = _group_performance(frame, campaign, revenue, spend)
        trend = _trend(frame, date_col, [(revenue, "revenue"), (spend, "spend")])

        if channel_perf:
            charts.append({
                "type": "bar",
                "title": "Revenue by Channel",
                "description": "Marketing revenue ranked by channel.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Revenue"}],
                "data": channel_perf,
            })

        if campaign_perf:
            charts.append({
                "type": "bar",
                "title": "Campaign Performance",
                "description": "Revenue contribution by campaign.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Revenue"}],
                "data": campaign_perf,
            })

        if trend:
            charts.append({
                "type": "line",
                "title": "Marketing Trend",
                "description": "Monthly marketing revenue and spend.",
                "x_key": "period",
                "series": [
                    {"data_key": "revenue", "label": "Revenue"},
                    {"data_key": "spend", "label": "Spend"},
                ],
                "data": trend,
            })

        if roas:
            insights.append(f"Overall ROAS is {kpis['roas']:.2f}.")
        if total_spend:
            insights.append(f"Total marketing spend is {_number(total_spend):,.2f}.")
        if total_conversions:
            insights.append(f"Total conversions are {int(total_conversions):,}.")

        return {
            **generic,
            "domain": domain,
            "kpis": kpis,
            "trend": trend,
            "channel_performance": channel_perf,
            "campaign_performance": campaign_perf,
            "charts": charts,
            "insights": insights,
        }

    if domain == "Manufacturing Analytics":
        units = _first_numeric(frame, ["Units Produced", "Units_Produced", "Production", "Output"])
        defects = _first_numeric(frame, ["Defects", "Defect Count", "Defect_Count"])
        downtime = _first_numeric(frame, ["Downtime", "Downtime Hours", "Downtime_Hours"])
        utilization = _first_numeric(frame, ["Utilization", "Machine Utilization", "Utilization Rate"])
        cost = _first_numeric(frame, ["Production Cost", "Production_Cost", "Cost"])
        product = _find_column(frame, ["Product", "Product Name", "Item"])
        machine = _find_column(frame, ["Machine", "Machine ID", "Machine_ID"])
        plant = _find_column(frame, ["Plant", "Factory", "Location"])

        total_units = _number(frame[units].sum()) if units else 0
        total_defects = _number(frame[defects].sum()) if defects else 0
        total_downtime = _number(frame[downtime].sum()) if downtime else 0
        avg_utilization = _number(frame[utilization].mean()) if utilization else 0
        total_cost = _number(frame[cost].sum()) if cost else 0

        kpis = {
            "units_produced": total_units,
            "defects": total_defects,
            "defect_rate": total_defects / total_units * 100 if total_units else 0,
            "downtime": total_downtime,
            "machine_utilization": avg_utilization,
            "production_cost": total_cost,
        }

        product_perf = _group_performance(frame, product, units, defects)
        machine_perf = _group_performance(frame, machine, units, defects)
        plant_perf = _group_performance(frame, plant, units, defects)
        trend = _trend(frame, date_col, [(units, "units_produced"), (defects, "defects")])

        if trend:
            charts.append({
                "type": "line",
                "title": "Production Trend",
                "description": "Monthly production output and defects.",
                "x_key": "period",
                "series": [
                    {"data_key": "units_produced", "label": "Units Produced"},
                    {"data_key": "defects", "label": "Defects"},
                ],
                "data": trend,
            })

        if product_perf:
            charts.append({
                "type": "bar",
                "title": "Units Produced by Product",
                "description": "Top products ranked by production output.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Units Produced"}],
                "data": product_perf,
            })

        if machine_perf:
            charts.append({
                "type": "bar",
                "title": "Production by Machine",
                "description": "Production output by machine.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Units Produced"}],
                "data": machine_perf,
            })

        if plant_perf:
            charts.append({
                "type": "bar",
                "title": "Production by Plant",
                "description": "Production output by plant or factory.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Units Produced"}],
                "data": plant_perf,
            })

        if total_units:
            insights.append(f"Total production output is {int(total_units):,} units.")
        if total_units and defects:
            insights.append(f"Defect rate is {kpis['defect_rate']:.2f}%.")
        if avg_utilization:
            insights.append(f"Average machine utilization is {avg_utilization:.2f}%.")

        return {
            **generic,
            "domain": domain,
            "kpis": kpis,
            "trend": trend,
            "product_performance": product_perf,
            "machine_performance": machine_perf,
            "plant_performance": plant_perf,
            "charts": charts,
            "insights": insights,
        }

    if domain == "HR Analytics":
        employee = _find_column(frame, ["Employee ID", "Employee", "Employee_ID"])
        salary = _first_numeric(frame, ["Salary", "Annual Salary", "Compensation"])
        attrition = _find_column(frame, ["Attrition", "Attrition Flag"])
        department = _find_column(frame, ["Department", "Team"])
        hire_date = _find_column(frame, ["Hire Date", "Joining Date", "Hire_Date"])
        absence = _first_numeric(frame, ["Absence", "Absence Days", "Absence_Days"])

        employees = (
            int(frame[employee].dropna().astype(str).nunique())
            if employee else int(len(frame))
        )
        attrition_count = 0
        if attrition:
            attrition_count = int(
                frame[attrition].astype(str).str.lower().isin(
                    ["yes", "true", "1", "left"]
                ).sum()
            )

        kpis = {
            "employees": employees,
            "attrition": attrition_count,
            "attrition_rate": attrition_count / employees * 100 if employees else 0,
            "average_salary": _number(frame[salary].mean()) if salary else 0,
            "absence_days": _number(frame[absence].sum()) if absence else 0,
        }

        dept_perf = _group_performance(frame, department, salary)
        if dept_perf:
            charts.append({
                "type": "bar",
                "title": "Average Department Salary",
                "description": "Salary distribution across departments.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Salary"}],
                "data": dept_perf,
            })

        insights.append(f"Workforce size is {employees:,}.")
        if employees:
            insights.append(f"Attrition rate is {kpis['attrition_rate']:.2f}%.")

        return {
            **generic,
            "domain": domain,
            "kpis": kpis,
            "department_performance": dept_perf,
            "charts": charts,
            "insights": insights,
        }

    if domain == "Customer Analytics":
        customer = _find_column(frame, ["Customer ID", "Customer", "Customer_ID"])
        segment = _find_column(frame, ["Customer Segment", "Segment"])
        revenue = _first_numeric(frame, ["Revenue", "Sales", "Customer Value", "CLV"])
        churn = _find_column(frame, ["Churn", "Churn Flag"])
        retention = _first_numeric(frame, ["Retention", "Retention Rate"])

        customers = (
            int(frame[customer].dropna().astype(str).nunique())
            if customer else int(len(frame))
        )
        churned = 0
        if churn:
            churned = int(
                frame[churn].astype(str).str.lower().isin(
                    ["yes", "true", "1", "churned"]
                ).sum()
            )

        kpis = {
            "customers": customers,
            "churned_customers": churned,
            "churn_rate": churned / customers * 100 if customers else 0,
            "average_customer_value": _number(frame[revenue].mean()) if revenue else 0,
            "retention_rate": _number(frame[retention].mean()) if retention else 0,
        }

        segment_perf = _group_performance(frame, segment, revenue)
        if segment_perf:
            charts.append({
                "type": "bar",
                "title": "Customer Value by Segment",
                "description": "Customer value ranked by segment.",
                "x_key": "name",
                "series": [{"data_key": "value", "label": "Customer Value"}],
                "data": segment_perf,
            })

        insights.append(f"NEXUS identified {customers:,} customers.")
        if customers:
            insights.append(f"Customer churn rate is {kpis['churn_rate']:.2f}%.")

        return {
            **generic,
            "domain": domain,
            "kpis": kpis,
            "segment_performance": segment_perf,
            "charts": charts,
            "insights": insights,
        }

    if domain == "Financial Analytics":
        revenue = _first_numeric(frame, ["Revenue", "Sales", "Income"])
        expense = _first_numeric(frame, ["Expense", "Expenses", "Operating Expense"])
        profit = _first_numeric(frame, ["Profit", "Net Profit", "Gross Profit"])
        cash = _first_numeric(frame, ["Cash Flow", "Cashflow", "Net Cash Flow"])
        date_trend = _trend(
            frame, date_col,
            [(revenue, "revenue"), (expense, "expense"), (profit, "profit")]
        )

        total_revenue = _number(frame[revenue].sum()) if revenue else 0
        total_expense = _number(frame[expense].sum()) if expense else 0
        total_profit = _number(frame[profit].sum()) if profit else total_revenue - total_expense

        kpis = {
            "revenue": total_revenue,
            "expenses": total_expense,
            "profit": total_profit,
            "profit_margin": total_profit / total_revenue * 100 if total_revenue else 0,
            "cash_flow": _number(frame[cash].sum()) if cash else 0,
        }

        if date_trend:
            charts.append({
                "type": "line",
                "title": "Financial Trend",
                "description": "Monthly revenue, expenses and profit.",
                "x_key": "period",
                "series": [
                    {"data_key": "revenue", "label": "Revenue"},
                    {"data_key": "expense", "label": "Expenses"},
                    {"data_key": "profit", "label": "Profit"},
                ],
                "data": date_trend,
            })

        insights.append(f"Total revenue is {total_revenue:,.2f}.")
        insights.append(f"Total expenses are {total_expense:,.2f}.")
        insights.append(f"Profit margin is {kpis['profit_margin']:.2f}%.")

        return {
            **generic,
            "domain": domain,
            "kpis": kpis,
            "trend": date_trend,
            "charts": charts,
            "insights": insights,
        }

    # Generic fallback for the remaining domains, while preserving
    # the universal sales-style fields already consumed by the UI.
    return {
        **generic,
        "domain": domain,
        "charts": charts,
        "insights": insights or [
            f"{domain} detected from the uploaded dataset.",
            f"NEXUS found {frame.shape[0]:,} rows and {frame.shape[1]:,} columns.",
        ],
    }


def build_dashboard(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()

    frame.columns = [str(c).strip() for c in frame.columns]

    revenue_col = _find_column(
        frame, ["Sales", "Revenue", "Amount", "Net Sales", "Total Sales"]
    )
    profit_col = _find_column(
        frame, ["Profit", "Net Profit", "Gross Profit", "Operating Profit"]
    )
    quantity_col = _find_column(
        frame, ["Quantity", "Qty", "Units", "Units Sold"]
    )
    customer_col = _find_column(
        frame, ["Customer", "Customer ID", "Customer_ID", "Client"]
    )
    order_col = _find_column(
        frame, ["Order ID", "Order_ID", "Transaction ID", "Invoice ID"]
    )
    date_col = _find_column(
        frame,
        ["Date", "Order Date", "Transaction Date", "Invoice Date",
         "Production Date", "Campaign Date"]
    )
    category_col = _find_column(
        frame, ["Category", "Product Category", "Product_Category"]
    )
    region_col = _find_column(
        frame, ["Region", "State", "Zone", "Territory"]
    )
    product_col = _find_column(
        frame, ["Product", "Product Name", "Product_Name", "Item"]
    )
    channel_col = _find_column(
        frame, ["Channel", "Sales Channel", "Sales_Channel", "Platform"]
    )

    for col in [revenue_col, profit_col, quantity_col]:
        if col:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    revenue = _number(frame[revenue_col].sum()) if revenue_col else 0
    profit = _number(frame[profit_col].sum()) if profit_col else 0
    quantity = _number(frame[quantity_col].sum()) if quantity_col else 0
    orders = (
        int(frame[order_col].nunique())
        if order_col else int(len(frame))
    )
    customers = (
        int(frame[customer_col].dropna().astype(str).nunique())
        if customer_col else 0
    )

    trend = _trend(
        frame,
        date_col,
        [(revenue_col, "revenue"), (profit_col, "profit")]
    )

    category_raw = _group_performance(frame, category_col, revenue_col, profit_col)
    region_raw = _group_performance(frame, region_col, revenue_col, profit_col)
    product_raw = _group_performance(frame, product_col, revenue_col, quantity_col)
    channel_raw = _group_performance(frame, channel_col, revenue_col, profit_col)

    category_performance = [
        {"category": x["name"], "revenue": x["value"], "profit": x.get("secondary", 0)}
        for x in category_raw
    ]
    region_performance = [
        {"region": x["name"], "revenue": x["value"], "profit": x.get("secondary", 0)}
        for x in region_raw
    ]
    product_performance = [
        {
            "product": x["name"],
            "revenue": x["value"],
            "profit": 0,
            "quantity": x.get("secondary", 0),
        }
        for x in product_raw
    ]
    channel_performance = [
        {"channel": x["name"], "revenue": x["value"], "profit": x.get("secondary", 0)}
        for x in channel_raw
    ]

    generic = {
        "columns_detected": {
            "revenue": revenue_col,
            "profit": profit_col,
            "quantity": quantity_col,
            "customer": customer_col,
            "order": order_col,
            "date": date_col,
            "category": category_col,
            "region": region_col,
            "product": product_col,
            "channel": channel_col,
        },
        "kpis": {
            "revenue": revenue,
            "profit": profit,
            "orders": orders,
            "customers": customers,
            "quantity": quantity,
            "profit_margin": profit / revenue * 100 if revenue else 0,
            "average_order_value": revenue / orders if orders else 0,
        },
        "trend": trend,
        "category_performance": category_performance,
        "region_performance": region_performance,
        "product_performance": product_performance,
        "channel_performance": channel_performance,
        "quality": _quality(frame),
        "charts": [],
        "insights": [],
    }

    rankings = _detect_domain(frame)
    primary_domain = rankings[0]["module"]

    result = _domain_dashboard(frame, primary_domain, generic)
    result["domain"] = primary_domain
    result["domain_rankings"] = rankings

    return result
