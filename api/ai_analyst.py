from __future__ import annotations

import io
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make the existing NEXUS app importable when this module is imported
# from api/main.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.loader import load_file


MODULE_ALIASES = {
    "sales": ["sales", "revenue", "orders", "units", "profit"],
    "customer": ["customer", "customers", "segment", "churn", "lifetime"],
    "marketing": ["marketing", "campaign", "channel", "click", "conversion", "roas"],
    "financial": ["finance", "financial", "revenue", "expense", "profit", "margin"],
    "supply_chain": ["supply", "supplier", "inventory", "delivery", "lead", "warehouse"],
    "product": ["product", "sku", "category", "rating"],
    "operations": ["operations", "output", "target", "downtime", "efficiency"],
    "hr": ["employee", "employees", "department", "salary", "attrition", "performance"],
    "fraud": ["fraud", "transaction", "risk", "suspicious"],
    "healthcare": ["patient", "patients", "diagnosis", "hospital", "stay", "medical"],
    "manufacturing": ["machine", "production", "defect", "maintenance", "quality"],
    "ecommerce": ["ecommerce", "e-commerce", "order", "channel", "payment", "rating"],
}

COLUMN_ALIASES = {
    "revenue": ["revenue", "sales", "sales_amount", "total_sales", "amount"],
    "profit": ["profit", "net_profit", "gross_profit", "profit_amount"],
    "cost": ["cost", "cost_amount", "expense", "expenses", "spend"],
    "quantity": ["quantity", "qty", "units", "units_sold", "units_produced"],
    "order": ["order", "orders", "order_id", "transaction", "transaction_id"],
    "customer": ["customer", "customer_id", "client", "client_id"],
    "product": ["product", "product_name", "sku", "item"],
    "category": ["category", "product_category", "segment"],
    "date": ["date", "order_date", "sales_date", "transaction_date", "created_at", "timestamp"],
    "region": ["region", "area", "territory", "zone"],
    "department": ["department", "dept", "team"],
    "rating": ["rating", "review_score", "score"],
    "fraud": ["fraud", "is_fraud", "fraud_flag", "fraudulent"],
    "risk": ["risk", "risk_score"],
    "status": ["status", "order_status", "transaction_status"],
}


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _best_dataframe(dataset: dict[str, Any]):
    frames = dataset.get("dataframes") or {}
    candidates = [
        (str(name), frame)
        for name, frame in frames.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda x: x[1].shape[0] * x[1].shape[1])


def load_dataframe(filename: str, content: bytes):
    class Adapter:
        def __init__(self, name, data):
            self.name = name
            self.filename = name
            self.type = ""
            self.content_type = ""
            self._data = data
        def getvalue(self):
            return self._data
        def read(self):
            return self._data
        def getbuffer(self):
            return memoryview(self._data)
        @property
        def size(self):
            return len(self._data)

    dataset = load_file(Adapter(filename, content))
    return _best_dataframe(dataset)


def find_column(df: pd.DataFrame, concept: str, question: str = ""):
    normalized = {normalize(c): c for c in df.columns}
    aliases = [normalize(x) for x in COLUMN_ALIASES.get(concept, [concept])]
    q = normalize(question)

    # Exact alias first.
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    # Alias contained in a real column name.
    for alias in aliases:
        for col_norm, original in normalized.items():
            if alias and (alias in col_norm or col_norm in alias):
                return original

    # Question terms can help with domain-specific names.
    q_tokens = set(q.split("_"))
    for col_norm, original in normalized.items():
        if len(q_tokens.intersection(set(col_norm.split("_")))) >= 1:
            if any(word in q for word in aliases):
                return original
    return None


def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df):
    return [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) <= min(100, max(10, len(df) // 2))
    ]


def date_column(df, question=""):
    for c in df.columns:
        if any(x in normalize(c) for x in ["date", "time", "timestamp", "month"]):
            converted = pd.to_datetime(df[c], errors="coerce")
            if converted.notna().mean() >= 0.6:
                return c
    best = None
    best_ratio = 0
    for c in df.columns:
        converted = pd.to_datetime(df[c], errors="coerce")
        ratio = converted.notna().mean()
        if ratio > best_ratio and ratio >= 0.8:
            best, best_ratio = c, ratio
    return best


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return "N/A"
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.2f}"
    return str(value)


def classify(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["missing", "null", "blank", "empty"]):
        return "missing"
    if any(x in q for x in ["duplicate", "duplicates"]):
        return "duplicates"
    if any(x in q for x in ["correlation", "correlate", "relationship between"]):
        return "correlation"
    if any(x in q for x in ["trend", "over time", "monthly", "weekly", "daily", "growth"]):
        return "trend"
    if any(x in q for x in ["top", "highest", "best", "largest", "maximum", "max"]):
        return "top"
    if any(x in q for x in ["bottom", "lowest", "worst", "smallest", "minimum", "min"]):
        return "bottom"
    if any(x in q for x in ["average", "mean", "avg"]):
        return "average"
    if any(x in q for x in ["count", "how many", "number of"]):
        return "count"
    if any(x in q for x in ["why", "decrease", "decreased", "increase", "increased", "decline", "drop", "grew"]):
        return "diagnostic"
    return "summary"


def table_payload(frame: pd.DataFrame, limit=10):
    frame = frame.head(limit).copy()
    return {
        "columns": [str(c) for c in frame.columns],
        "rows": [
            [json_safe(v) for v in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def analyze_question(df: pd.DataFrame, question: str, module: str = "") -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "success": False,
            "answer": "The dataset is empty, so I cannot perform the requested analysis.",
            "intent": "summary",
            "evidence": [],
            "table": {"columns": [], "rows": []},
        }

    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    intent = classify(question)
    q = question.lower()
    evidence = []
    result_table = pd.DataFrame()
    answer = ""

    metric = None
    for concept in ["revenue", "profit", "cost", "quantity", "rating", "risk"]:
        if any(word in q for word in COLUMN_ALIASES[concept]) or concept in q:
            metric = find_column(df, concept, question)
            if metric:
                break
    if metric is None:
        nums = numeric_columns(df)
        metric = nums[0] if nums else None

    group = None
    for concept in ["category", "region", "department", "product", "customer", "status"]:
        col = find_column(df, concept, question)
        if col and any(word in q for word in [concept, *COLUMN_ALIASES.get(concept, [])]):
            group = col
            break
    if group is None:
        cats = categorical_columns(df)
        group = cats[0] if cats else None

    if intent == "missing":
        counts = df.isna().sum().sort_values(ascending=False)
        counts = counts[counts > 0]
        if counts.empty:
            answer = "No missing values were detected in the dataset."
        else:
            result_table = counts.rename("missing_values").reset_index()
            result_table.columns = ["column", "missing_values"]
            answer = f"I found {int(counts.sum()):,} missing values across {len(counts)} columns. The largest concentration is in {counts.index[0]} ({int(counts.iloc[0]):,})."
            evidence.append({"metric": "total_missing_values", "value": int(counts.sum())})

    elif intent == "duplicates":
        duplicate_count = int(df.duplicated().sum())
        answer = f"I found {duplicate_count:,} duplicate rows in a dataset containing {len(df):,} rows."
        evidence.append({"metric": "duplicate_rows", "value": duplicate_count})

    elif intent == "correlation":
        nums = numeric_columns(df)
        if len(nums) < 2:
            answer = "There are not enough numeric columns to calculate correlations."
        else:
            corr = df[nums].corr(numeric_only=True).abs()
            np.fill_diagonal(corr.values, np.nan)
            stacked = corr.stack().sort_values(ascending=False)
            if stacked.empty:
                answer = "No usable numeric correlation was found."
            else:
                a, b = stacked.index[0]
                raw = df[[a, b]].corr(numeric_only=True).iloc[0, 1]
                answer = f"The strongest numeric relationship I found is between {a} and {b}, with a correlation of {raw:.3f}."
                result_table = pd.DataFrame({"column_a": [a], "column_b": [b], "correlation": [raw]})
                evidence.append({"metric": "correlation", "value": float(raw), "columns": [str(a), str(b)]})

    elif intent in {"top", "bottom"}:
        if metric is None:
            answer = "I could not identify a numeric metric to rank."
        elif group is not None:
            grouped = df.groupby(group, dropna=False)[metric].sum(numeric_only=True).sort_values(
                ascending=(intent == "bottom")
            ).head(10)
            result_table = grouped.rename(metric).reset_index()
            direction = "lowest" if intent == "bottom" else "highest"
            answer = f"By {group}, the {direction} {metric} is {grouped.index[0]} at {fmt(grouped.iloc[0])}."
            evidence.append({"metric": metric, "group": str(grouped.index[0]), "value": json_safe(grouped.iloc[0])})
        else:
            value = df[metric].max() if intent == "top" else df[metric].min()
            row = df.loc[df[metric].idxmax() if intent == "top" else df[metric].idxmin()]
            result_table = pd.DataFrame([row])
            direction = "highest" if intent == "top" else "lowest"
            answer = f"The {direction} {metric} value is {fmt(value)}."
            evidence.append({"metric": metric, "value": json_safe(value)})

    elif intent == "average":
        if metric is None:
            answer = "I could not identify a numeric metric for the average."
        else:
            value = float(df[metric].mean())
            answer = f"The average {metric} is {fmt(value)} across {df[metric].notna().sum():,} records."
            evidence.append({"metric": metric, "average": value})

    elif intent == "count":
        answer = f"The dataset contains {len(df):,} rows."
        evidence.append({"metric": "row_count", "value": len(df)})
        if group is not None:
            counts = df[group].value_counts(dropna=False).head(10)
            result_table = counts.rename("count").reset_index()

    elif intent == "trend":
        dcol = date_column(df, question)
        if dcol is None or metric is None:
            answer = "I need both a date column and a numeric metric to calculate a trend."
        else:
            temp = df[[dcol, metric]].copy()
            temp[dcol] = pd.to_datetime(temp[dcol], errors="coerce")
            temp = temp.dropna(subset=[dcol, metric])
            if temp.empty:
                answer = "There are no usable date/metric records for a trend."
            else:
                temp["period"] = temp[dcol].dt.to_period("M").astype(str)
                trend = temp.groupby("period")[metric].sum().reset_index()
                result_table = trend.tail(12)
                first = float(trend[metric].iloc[0])
                last = float(trend[metric].iloc[-1])
                change = ((last - first) / abs(first) * 100) if first else None
                if change is None:
                    answer = f"{metric} changed from {fmt(first)} to {fmt(last)}."
                else:
                    direction = "increased" if change >= 0 else "decreased"
                    answer = f"{metric} {direction} by {abs(change):.1f}% from the first to the latest available month."
                    evidence.append({"metric": metric, "first": first, "latest": last, "change_percent": change})

    elif intent == "diagnostic":
        dcol = date_column(df, question)
        if metric is not None and dcol is not None:
            temp = df[[dcol, metric]].copy()
            temp[dcol] = pd.to_datetime(temp[dcol], errors="coerce")
            temp = temp.dropna(subset=[dcol, metric])
            if not temp.empty:
                temp["period"] = temp[dcol].dt.to_period("M").astype(str)
                trend = temp.groupby("period")[metric].sum().reset_index()
                if len(trend) >= 2:
                    trend["change"] = trend[metric].diff()
                    worst = trend.loc[trend["change"].idxmin()]
                    best = trend.loc[trend["change"].idxmax()]
                    result_table = trend.tail(12)
                    answer = (
                        f"I found the largest month-over-month decline in {metric} during "
                        f"{worst['period']} ({fmt(worst['change'])}). "
                        f"The largest increase was in {best['period']} ({fmt(best['change'])})."
                    )
                    evidence.append({
                        "largest_decline_period": str(worst["period"]),
                        "largest_decline": json_safe(worst["change"]),
                    })
                else:
                    answer = "There are not enough time periods to diagnose a change."
            else:
                answer = "There are no usable date/metric records for diagnostic analysis."
        elif metric is not None and group is not None:
            grouped = df.groupby(group)[metric].sum().sort_values()
            if len(grouped) >= 2:
                low_name, low_value = grouped.index[0], grouped.iloc[0]
                high_name, high_value = grouped.index[-1], grouped.iloc[-1]
                result_table = grouped.reset_index(name=metric)
                answer = (
                    f"{metric} varies most across {group}. "
                    f"The highest group is {high_name} ({fmt(high_value)}) and "
                    f"the lowest is {low_name} ({fmt(low_value)})."
                )
            else:
                answer = "There are not enough groups to diagnose the change."
        else:
            answer = "I could not identify a suitable metric and dimension for a diagnostic analysis."

    else:
        numeric = numeric_columns(df)
        answer = f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns."
        if metric:
            answer += f" Total {metric}: {fmt(df[metric].sum())}."
            evidence.append({"metric": metric, "total": json_safe(df[metric].sum())})
        if group and metric:
            grouped = df.groupby(group)[metric].sum().sort_values(ascending=False).head(5)
            result_table = grouped.rename(metric).reset_index()
            if not grouped.empty:
                answer += f" The leading {group} by {metric} is {grouped.index[0]}."

    return {
        "success": True,
        "intent": intent,
        "module": module,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "table": table_payload(result_table, 12) if not result_table.empty else {"columns": [], "rows": []},
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": [str(c) for c in df.columns],
        },
    }
