from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _outliers(df: pd.DataFrame) -> list[dict[str, Any]]:
    result = []

    for column in df.select_dtypes(include=np.number).columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if len(series) < 5:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            count = 0
            lower = None
            upper = None
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())

        result.append({
            "column": str(column),
            "outlier_count": count,
            "outlier_rate": round(count / len(series) * 100, 3),
            "lower_bound": lower,
            "upper_bound": upper,
        })

    return sorted(result, key=lambda x: x["outlier_count"], reverse=True)


def _invalid_dates(df: pd.DataFrame) -> list[dict[str, Any]]:
    result = []

    for column in df.columns:
        name = _norm(column)

        if not any(x in name for x in ("date", "time", "timestamp", "dob")):
            continue

        series = df[column]
        non_empty = int(series.notna().sum())

        if non_empty == 0:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            invalid = 0
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            invalid = int(non_empty - parsed.notna().sum())

        if invalid:
            result.append({
                "column": str(column),
                "invalid_count": invalid,
                "invalid_rate": round(invalid / non_empty * 100, 3),
            })

    return result


def _possible_pii(df: pd.DataFrame) -> list[dict[str, Any]]:
    patterns = {
        "email": ("email", "e mail"),
        "phone": ("phone", "mobile", "telephone"),
        "name": ("full name", "customer name", "employee name"),
        "address": ("address", "street"),
        "government_id": ("aadhaar", "passport", "pan number", "ssn"),
    }

    result = []

    for column in df.columns:
        name = _norm(column)

        for pii_type, words in patterns.items():
            if any(word in name for word in words):
                result.append({
                    "column": str(column),
                    "type": pii_type,
                    "reason": "Column name may contain sensitive personal information.",
                })
                break

    return result


def _high_cardinality(df: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    rows = max(len(df), 1)

    for column in df.columns:
        unique = int(df[column].nunique(dropna=True))
        rate = unique / rows

        if unique > 10 and rate >= 0.90:
            result.append({
                "column": str(column),
                "unique_values": unique,
                "cardinality_rate": round(rate * 100, 3),
            })

    return sorted(result, key=lambda x: x["cardinality_rate"], reverse=True)


def _duplicate_keys(df: pd.DataFrame) -> list[dict[str, Any]]:
    result = []

    for column in df.columns:
        name = _norm(column)

        if not (
            name == "id"
            or name == "sku"
            or name.endswith(" id")
            or name.endswith("_id")
        ):
            continue

        values = df[column].dropna()

        if values.empty:
            continue

        duplicate_rows = int(values.duplicated(keep=False).sum())

        if duplicate_rows:
            result.append({
                "column": str(column),
                "duplicate_value_rows": duplicate_rows,
                "unique_values": int(values.nunique()),
            })

    return result


def analyze_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()

    rows = int(frame.shape[0])
    columns = int(frame.shape[1])
    total_cells = max(rows * columns, 1)

    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())

    missing_columns = []

    for column in frame.columns:
        count = int(frame[column].isna().sum())

        if count:
            missing_columns.append({
                "column": str(column),
                "missing_count": count,
                "missing_rate": round(count / max(rows, 1) * 100, 3),
            })

    missing_columns.sort(
        key=lambda x: x["missing_count"],
        reverse=True,
    )

    outliers = _outliers(frame)
    invalid_dates = _invalid_dates(frame)

    constant_columns = [
        str(column)
        for column in frame.columns
        if frame[column].nunique(dropna=False) <= 1
    ]

    high_cardinality = _high_cardinality(frame)
    possible_pii = _possible_pii(frame)
    duplicate_keys = _duplicate_keys(frame)

    missing_rate = missing_cells / total_cells * 100
    duplicate_rate = duplicate_rows / max(rows, 1) * 100
    outlier_rows = sum(x["outlier_count"] for x in outliers)
    invalid_date_values = sum(x["invalid_count"] for x in invalid_dates)

    score = 100.0
    score -= min(30.0, missing_rate * 0.8)
    score -= min(20.0, duplicate_rate * 0.7)

    if rows:
        score -= min(20.0, outlier_rows / rows * 100 * 0.35)
        score -= min(15.0, invalid_date_values / rows * 100 * 0.5)

    score -= min(5.0, len(constant_columns) * 0.5)
    score = max(0.0, min(100.0, score))

    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 60:
        grade = "Needs Attention"
    else:
        grade = "Poor"

    issues = []

    if missing_cells:
        issues.append({
            "severity": "warning",
            "type": "missing_values",
            "message": f"{missing_cells:,} missing cells detected.",
            "recommendation": "Review missing fields before analysis.",
        })

    if duplicate_rows:
        issues.append({
            "severity": "warning",
            "type": "duplicate_rows",
            "message": f"{duplicate_rows:,} duplicate rows detected.",
            "recommendation": "Check whether duplicate records are valid.",
        })

    if outlier_rows:
        issues.append({
            "severity": "warning",
            "type": "outliers",
            "message": f"{outlier_rows:,} numeric outlier observations detected.",
            "recommendation": "Inspect extreme values before statistical analysis.",
        })

    if invalid_date_values:
        issues.append({
            "severity": "error",
            "type": "invalid_dates",
            "message": f"{invalid_date_values:,} invalid date values detected.",
            "recommendation": "Standardize and repair invalid date values.",
        })

    if constant_columns:
        issues.append({
            "severity": "info",
            "type": "constant_columns",
            "message": f"{len(constant_columns)} constant column(s) detected.",
            "recommendation": "Consider removing columns with no analytical variation.",
        })

    if high_cardinality:
        issues.append({
            "severity": "info",
            "type": "high_cardinality",
            "message": f"{len(high_cardinality)} high-cardinality column(s) detected.",
            "recommendation": "Review ID-like fields before categorical charts.",
        })

    if possible_pii:
        issues.append({
            "severity": "security",
            "type": "possible_pii",
            "message": f"{len(possible_pii)} potentially sensitive column(s) detected.",
            "recommendation": "Restrict access or mask sensitive fields.",
        })

    return {
        "score": round(score, 2),
        "grade": grade,
        "rows": rows,
        "columns": columns,
        "missing_cells": missing_cells,
        "missing_rate": round(missing_rate, 3),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rate, 3),
        "outlier_rows": int(outlier_rows),
        "invalid_date_values": int(invalid_date_values),
        "missing_columns": missing_columns[:25],
        "outliers": outliers[:25],
        "invalid_dates": invalid_dates[:25],
        "constant_columns": constant_columns[:25],
        "high_cardinality": high_cardinality[:25],
        "possible_pii": possible_pii[:25],
        "duplicate_keys": duplicate_keys[:25],
        "issues": issues,
    }
