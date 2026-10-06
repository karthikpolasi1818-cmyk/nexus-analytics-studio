"""
NEXUS Analytics Studio
AI Context Service v1.1

Builds compact, JSON-safe analytical context from the FULL pandas
DataFrame. The AI Analyst can therefore reason over the complete
dataset without sending every raw row to the browser.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


MAX_CATEGORY_VALUES = 20
MAX_CORRELATIONS = 20
MAX_ANOMALIES = 20
MAX_TREND_POINTS = 36


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]

    return str(value)


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {
        _normalize(column): str(column)
        for column in df.columns
    }


def _find_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:

    lookup = _column_lookup(df)

    # Exact normalized match.
    for alias in aliases:
        normalized = _normalize(alias)

        if normalized in lookup:
            return lookup[normalized]

    # Contained match.
    for alias in aliases:
        normalized = _normalize(alias)

        for candidate, original in lookup.items():
            if (
                normalized
                and (
                    normalized in candidate
                    or candidate in normalized
                )
            ):
                return original

    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    result = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        unique = int(df[column].nunique(dropna=True))

        if 1 <= unique <= min(100, max(20, len(df) // 2)):
            result.append(str(column))

    return result


def _date_columns(df: pd.DataFrame) -> list[str]:
    result = []

    for column in df.columns:
        name = _normalize(column)

        if not any(
            token in name
            for token in [
                "date",
                "time",
                "timestamp",
                "month",
                "year",
            ]
        ):
            continue

        converted = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        if converted.notna().mean() >= 0.6:
            result.append(str(column))

    return result


def _quality(df: pd.DataFrame) -> dict[str, Any]:
    total_cells = int(df.shape[0] * df.shape[1])
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    completeness = (
        ((total_cells - missing) / total_cells) * 100
        if total_cells
        else 0.0
    )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": total_cells,
        "missing_cells": missing,
        "missing_percent": (
            (missing / total_cells) * 100
            if total_cells
            else 0.0
        ),
        "completeness_percent": completeness,
        "duplicate_rows": duplicates,
    }


def _numeric_statistics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for column in _numeric_columns(df):
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        result[column] = {
            "count": int(series.count()),
            "sum": _safe(series.sum()),
            "mean": _safe(series.mean()),
            "median": _safe(series.median()),
            "std": _safe(series.std()),
            "min": _safe(series.min()),
            "max": _safe(series.max()),
            "q1": _safe(series.quantile(0.25)),
            "q3": _safe(series.quantile(0.75)),
        }

    return result


def _categorical_statistics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for column in _categorical_columns(df):
        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(dropna=False)
            .head(MAX_CATEGORY_VALUES)
        )

        result[column] = {
            "unique": int(df[column].nunique(dropna=True)),
            "top_values": [
                {
                    "value": str(index),
                    "count": int(value),
                    "percent": (
                        float(value) / len(df) * 100
                        if len(df)
                        else 0.0
                    ),
                }
                for index, value in counts.items()
            ],
        }

    return result


def _group_performance(
    df: pd.DataFrame,
) -> dict[str, Any]:

    numeric = _numeric_columns(df)
    categorical = _categorical_columns(df)

    result: dict[str, Any] = {}

    # Prevent the context from exploding on very wide datasets.
    numeric = numeric[:15]
    categorical = categorical[:10]

    for group in categorical:
        group_result: dict[str, Any] = {}

        for metric in numeric:
            temp = df[[group, metric]].copy()

            temp[metric] = pd.to_numeric(
                temp[metric],
                errors="coerce",
            )

            temp = temp.dropna(subset=[metric])

            if temp.empty:
                continue

            grouped = (
                temp.groupby(
                    group,
                    dropna=False,
                )[metric]
                .sum()
                .sort_values(ascending=False)
                .head(MAX_CATEGORY_VALUES)
            )

            group_result[metric] = [
                {
                    "value": str(index),
                    "metric_value": _safe(value),
                }
                for index, value in grouped.items()
            ]

        if group_result:
            result[group] = group_result

    return result


def _derived_metrics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    metrics: dict[str, Any] = {}

    revenue = _find_column(
        df,
        [
            "revenue",
            "sales",
            "sales_amount",
            "total_sales",
        ],
    )

    spend = _find_column(
        df,
        [
            "spend",
            "cost",
            "marketing_spend",
            "ad_spend",
            "advertising_spend",
        ],
    )

    impressions = _find_column(
        df,
        ["impressions", "impression"],
    )

    clicks = _find_column(
        df,
        ["clicks", "click"],
    )

    conversions = _find_column(
        df,
        [
            "conversions",
            "conversion",
            "orders",
        ],
    )

    if revenue and spend:
        total_revenue = pd.to_numeric(
            df[revenue],
            errors="coerce",
        ).sum()

        total_spend = pd.to_numeric(
            df[spend],
            errors="coerce",
        ).sum()

        if total_spend:
            metrics["roas"] = {
                "value": _safe(
                    total_revenue / total_spend
                ),
                "formula": "total revenue / total spend",
                "revenue_column": revenue,
                "spend_column": spend,
            }

    if clicks and impressions:
        total_clicks = pd.to_numeric(
            df[clicks],
            errors="coerce",
        ).sum()

        total_impressions = pd.to_numeric(
            df[impressions],
            errors="coerce",
        ).sum()

        if total_impressions:
            metrics["ctr"] = {
                "value": _safe(
                    total_clicks
                    / total_impressions
                    * 100
                ),
                "unit": "percent",
                "formula": "total clicks / total impressions * 100",
            }

    if conversions and clicks:
        total_conversions = pd.to_numeric(
            df[conversions],
            errors="coerce",
        ).sum()

        total_clicks = pd.to_numeric(
            df[clicks],
            errors="coerce",
        ).sum()

        if total_clicks:
            metrics["conversion_rate"] = {
                "value": _safe(
                    total_conversions
                    / total_clicks
                    * 100
                ),
                "unit": "percent",
                "formula": "total conversions / total clicks * 100",
            }

    if spend and conversions:
        total_spend = pd.to_numeric(
            df[spend],
            errors="coerce",
        ).sum()

        total_conversions = pd.to_numeric(
            df[conversions],
            errors="coerce",
        ).sum()

        if total_conversions:
            metrics["cpa"] = {
                "value": _safe(
                    total_spend / total_conversions
                ),
                "formula": "total spend / total conversions",
            }

    if spend and clicks:
        total_spend = pd.to_numeric(
            df[spend],
            errors="coerce",
        ).sum()

        total_clicks = pd.to_numeric(
            df[clicks],
            errors="coerce",
        ).sum()

        if total_clicks:
            metrics["cpc"] = {
                "value": _safe(
                    total_spend / total_clicks
                ),
                "formula": "total spend / total clicks",
            }

    return metrics


def _correlations(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    numeric = _numeric_columns(df)

    if len(numeric) < 2:
        return []

    numeric = numeric[:25]

    frame = df[numeric].apply(
        pd.to_numeric,
        errors="coerce",
    )

    corr = frame.corr()

    relationships = []

    for i, column_a in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            column_b = corr.columns[j]

            value = corr.iloc[i, j]

            if pd.isna(value):
                continue

            relationships.append({
                "column_a": str(column_a),
                "column_b": str(column_b),
                "correlation": float(value),
                "absolute_correlation": abs(float(value)),
            })

    relationships.sort(
        key=lambda item: item["absolute_correlation"],
        reverse=True,
    )

    return relationships[:MAX_CORRELATIONS]


def _anomalies(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    results = []

    for column in _numeric_columns(df):
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = series.dropna()

        if len(valid) < 4:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        if not math.isfinite(float(iqr)) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)

        anomaly_rows = df.loc[mask, [column]].head(5)

        if anomaly_rows.empty:
            continue

        results.append({
            "column": column,
            "count": int(mask.sum()),
            "lower_bound": _safe(lower),
            "upper_bound": _safe(upper),
            "examples": [
                {
                    "row": int(index)
                    if isinstance(index, (int, np.integer))
                    else str(index),
                    "value": _safe(value),
                }
                for index, value in anomaly_rows[column].items()
            ],
        })

    results.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return results[:MAX_ANOMALIES]


def _trends(
    df: pd.DataFrame,
) -> dict[str, Any]:

    dates = _date_columns(df)
    numeric = _numeric_columns(df)

    if not dates or not numeric:
        return {}

    date_col = dates[0]

    temp = df.copy()

    temp[date_col] = pd.to_datetime(
        temp[date_col],
        errors="coerce",
    )

    temp = temp.dropna(subset=[date_col])

    if temp.empty:
        return {}

    temp["_nexus_period"] = (
        temp[date_col]
        .dt.to_period("M")
        .astype(str)
    )

    result: dict[str, Any] = {
        "date_column": date_col,
        "metrics": {},
    }

    for metric in numeric[:15]:
        values = pd.to_numeric(
            temp[metric],
            errors="coerce",
        )

        working = pd.DataFrame({
            "period": temp["_nexus_period"],
            "value": values,
        }).dropna()

        if working.empty:
            continue

        trend = (
            working.groupby("period")["value"]
            .sum()
            .reset_index()
            .tail(MAX_TREND_POINTS)
        )

        points = [
            {
                "period": str(row.period),
                "value": _safe(row.value),
            }
            for row in trend.itertuples(index=False)
        ]

        if not points:
            continue

        first = points[0]["value"]
        latest = points[-1]["value"]

        change_percent = None

        if (
            isinstance(first, (int, float))
            and first != 0
            and isinstance(latest, (int, float))
        ):
            change_percent = (
                (latest - first)
                / abs(first)
                * 100
            )

        result["metrics"][metric] = {
            "points": points,
            "first": first,
            "latest": latest,
            "change_percent": _safe(change_percent),
        }

    return result


def build_ai_context(
    df: pd.DataFrame,
    module: str = "",
) -> dict[str, Any]:

    if df is None or df.empty:
        return {
            "version": "1.1",
            "module": module,
            "available": False,
        }

    working = df.copy()

    working.columns = [
        str(column)
        for column in working.columns
    ]

    context = {
        "version": "1.1",
        "available": True,
        "module": module,
        "shape": {
            "rows": int(working.shape[0]),
            "columns": int(working.shape[1]),
        },
        "columns": [
            str(column)
            for column in working.columns
        ],
        "quality": _quality(working),
        "numeric_statistics": _numeric_statistics(
            working
        ),
        "categorical_statistics": _categorical_statistics(
            working
        ),
        "group_performance": _group_performance(
            working
        ),
        "derived_metrics": _derived_metrics(
            working
        ),
        "correlations": _correlations(
            working
        ),
        "anomalies": _anomalies(
            working
        ),
        "trends": _trends(
            working
        ),
    }

    return _safe(context)