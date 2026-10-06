from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    ).strip("_")

    aliases = {
        "customerid": "customer_id",
        "customer": "customer_id",
        "customer_number": "customer_id",
        "cust_id": "customer_id",

        "orderid": "order_id",
        "order_number": "order_id",

        "productid": "product_id",
        "product_number": "product_id",

        "employeeid": "employee_id",
        "employee_number": "employee_id",
        "emp_id": "employee_id",

        "supplierid": "supplier_id",
        "supplier_number": "supplier_id",

        "vendorid": "vendor_id",
        "vendor_number": "vendor_id",

        "patientid": "patient_id",
        "patient_number": "patient_id",

        "transactionid": "transaction_id",
        "transaction_number": "transaction_id",

        "accountid": "account_id",
        "account_number": "account_id",
    }

    return aliases.get(text, text)


# ============================================================
# COLUMN INFORMATION
# ============================================================

def _column_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    result = []

    for column in df.columns:

        series = df[column]

        normalized = _normalize_column_name(column)

        non_null = series.dropna()

        unique_count = int(
            non_null.nunique()
        )

        row_count = max(
            int(len(non_null)),
            1,
        )

        uniqueness_ratio = (
            unique_count / row_count
        )

        result.append(
            {
                "name": str(column),
                "normalized": normalized,
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "non_null": int(series.notna().sum()),
                "unique": unique_count,
                "uniqueness_ratio": round(
                    uniqueness_ratio,
                    4,
                ),
            }
        )

    return result


# ============================================================
# KEY LIKENESS
# ============================================================

def _is_key_like(column: dict[str, Any]) -> bool:

    name = column["normalized"]

    key_terms = (
        "id",
        "code",
        "number",
        "no",
        "key",
    )

    if (
        name.endswith("_id")
        or name.endswith("_code")
        or name.endswith("_number")
        or name.endswith("_no")
        or name == "id"
    ):
        return True

    return any(
        term in name.split("_")
        for term in key_terms
    )


# ============================================================
# VALUE OVERLAP
# ============================================================

def _value_overlap(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float, int]:

    left_values = set(
        left.dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    right_values = set(
        right.dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not left_values or not right_values:
        return 0.0, 0

    intersection = (
        left_values & right_values
    )

    overlap_count = len(intersection)

    smaller = min(
        len(left_values),
        len(right_values),
    )

    if smaller == 0:
        return 0.0, 0

    overlap_ratio = (
        overlap_count / smaller
    )

    return (
        float(overlap_ratio),
        overlap_count,
    )


# ============================================================
# RELATIONSHIP CONFIDENCE
# ============================================================

def _confidence(
    normalized_match: bool,
    key_like: bool,
    overlap_ratio: float,
) -> str:

    if (
        normalized_match
        and key_like
        and overlap_ratio >= 0.5
    ):
        return "high"

    if (
        normalized_match
        and overlap_ratio >= 0.2
    ):
        return "medium"

    if overlap_ratio >= 0.1:
        return "possible"

    return "low"


# ============================================================
# RELATIONSHIP ANALYSIS
# ============================================================

def _find_relationships(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> list[dict[str, Any]]:

    relationships = []

    left_columns = _column_info(left_df)
    right_columns = _column_info(right_df)

    for left_column in left_columns:

        for right_column in right_columns:

            left_normalized = (
                left_column["normalized"]
            )

            right_normalized = (
                right_column["normalized"]
            )

            normalized_match = (
                left_normalized
                == right_normalized
            )

            name_similarity = (
                left_normalized
                in right_normalized
                or right_normalized
                in left_normalized
            )

            if not normalized_match and not name_similarity:
                continue

            left_series = left_df[
                left_column["name"]
            ]

            right_series = right_df[
                right_column["name"]
            ]

            overlap_ratio, overlap_count = (
                _value_overlap(
                    left_series,
                    right_series,
                )
            )

            key_like = (
                _is_key_like(left_column)
                or _is_key_like(right_column)
            )

            confidence = _confidence(
                normalized_match,
                key_like,
                overlap_ratio,
            )

            if confidence == "low":
                continue

            relationships.append(
                {
                    "left_dataset": left_name,
                    "left_column": left_column["name"],
                    "right_dataset": right_name,
                    "right_column": right_column["name"],
                    "normalized_key": left_normalized,
                    "confidence": confidence,
                    "overlap_ratio": round(
                        overlap_ratio * 100,
                        2,
                    ),
                    "overlap_values": overlap_count,
                    "left_unique_values": left_column["unique"],
                    "right_unique_values": right_column["unique"],
                    "join_type": "left",
                    "safe_to_join": (
                        confidence
                        in {"high", "medium"}
                    ),
                }
            )

    return relationships


# ============================================================
# JOIN ANALYSIS
# ============================================================

def _safe_join_analysis(
    relationship: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
) -> dict[str, Any]:

    left_name = relationship["left_dataset"]
    right_name = relationship["right_dataset"]

    left_column = relationship["left_column"]
    right_column = relationship["right_column"]

    left_df = datasets[left_name]
    right_df = datasets[right_name]

    try:

        left_keys = (
            left_df[left_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        right_keys = (
            right_df[right_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        right_key_set = set(
            right_keys.tolist()
        )

        matched_rows = int(
            left_keys.isin(
                right_key_set
            ).sum()
        )

        total_rows = int(
            len(left_keys)
        )

        match_rate = (
            matched_rows / total_rows * 100
            if total_rows
            else 0.0
        )

        joined = pd.DataFrame()

        left_copy = left_df.copy()
        right_copy = right_df.copy()

        left_copy[
            "__nexus_join_key__"
        ] = (
            left_copy[left_column]
            .astype(str)
            .str.strip()
        )

        right_copy[
            "__nexus_join_key__"
        ] = (
            right_copy[right_column]
            .astype(str)
            .str.strip()
        )

        joined = left_copy.merge(
            right_copy,
            on="__nexus_join_key__",
            how="inner",
            suffixes=(
                "_left",
                "_right",
            ),
        )

        return {
            "left_dataset": left_name,
            "right_dataset": right_name,
            "left_column": left_column,
            "right_column": right_column,
            "matched_left_rows": matched_rows,
            "left_rows": total_rows,
            "match_rate": round(
                match_rate,
                2,
            ),
            "joined_rows": int(
                len(joined)
            ),
            "safe": bool(
                relationship["safe_to_join"]
            ),
        }

    except Exception as exc:

        return {
            "left_dataset": left_name,
            "right_dataset": right_name,
            "left_column": left_column,
            "right_column": right_column,
            "matched_left_rows": 0,
            "left_rows": int(
                len(left_df)
            ),
            "match_rate": 0.0,
            "joined_rows": 0,
            "safe": False,
            "error": str(exc),
        }


# ============================================================
# PUBLIC API
# ============================================================

def analyze_cross_file_relationships(
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:

    clean_datasets = []

    for item in datasets:

        if not isinstance(
            item,
            dict,
        ):
            continue

        df = item.get(
            "dataframe"
        )

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            continue

        if df.empty:
            continue

        clean_datasets.append(
            {
                "filename": str(
                    item.get(
                        "filename",
                        "dataset",
                    )
                ),
                "sheet": item.get(
                    "sheet"
                ),
                "dataframe": df,
            }
        )

    dataset_summary = []

    dataframe_map = {}

    for item in clean_datasets:

        name = item["filename"]

        if name in dataframe_map:
            name = (
                f'{name}::{item.get("sheet")}'
            )

        dataframe_map[name] = (
            item["dataframe"]
        )

        dataset_summary.append(
            {
                "filename": name,
                "sheet": item.get("sheet"),
                "rows": int(
                    len(
                        item["dataframe"]
                    )
                ),
                "columns": int(
                    len(
                        item["dataframe"].columns
                    )
                ),
                "column_names": [
                    str(column)
                    for column
                    in item["dataframe"].columns
                ],
            }
        )

    relationships = []

    dataset_names = list(
        dataframe_map.keys()
    )

    for index, left_name in enumerate(
        dataset_names
    ):

        for right_name in dataset_names[
            index + 1:
        ]:

            found = _find_relationships(
                left_name,
                dataframe_map[left_name],
                right_name,
                dataframe_map[right_name],
            )

            relationships.extend(
                found
            )

    confidence_order = {
        "high": 0,
        "medium": 1,
        "possible": 2,
    }

    relationships.sort(
        key=lambda item: (
            confidence_order.get(
                item["confidence"],
                99,
            ),
            -item["overlap_ratio"],
        )
    )

    joined_analyses = []

    for relationship in relationships:

        if relationship[
            "confidence"
        ] not in {
            "high",
            "medium",
        }:
            continue

        joined_analyses.append(
            _safe_join_analysis(
                relationship,
                dataframe_map,
            )
        )

        if len(joined_analyses) >= 20:
            break

    opportunities = []

    for relationship in relationships:

        if relationship[
            "confidence"
        ] == "high":

            opportunities.append(
                {
                    "type": "cross_file_join",
                    "title": (
                        f"Join "
                        f"{relationship['left_dataset']} "
                        f"with "
                        f"{relationship['right_dataset']}"
                    ),
                    "description": (
                        f"Both datasets share "
                        f"the key "
                        f"'{relationship['left_column']}' "
                        f"with "
                        f"{relationship['overlap_ratio']}% "
                        f"value overlap."
                    ),
                    "confidence": (
                        relationship[
                            "confidence"
                        ]
                    ),
                }
            )

    return {
        "datasets": dataset_summary,
        "relationships": relationships,
        "joined_analyses": joined_analyses,
        "analysis_opportunities": opportunities,
        "stats": {
            "datasets": len(
                dataset_summary
            ),
            "relationships": len(
                relationships
            ),
            "high_confidence": sum(
                1
                for relationship
                in relationships
                if relationship[
                    "confidence"
                ] == "high"
            ),
            "medium_confidence": sum(
                1
                for relationship
                in relationships
                if relationship[
                    "confidence"
                ] == "medium"
            ),
            "possible": sum(
                1
                for relationship
                in relationships
                if relationship[
                    "confidence"
                ] == "possible"
            ),
        },
    }


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def analyze_cross_file(
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:

    return analyze_cross_file_relationships(
        datasets
    )
