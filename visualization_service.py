from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _norm(value: Any) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").split()
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_norm(c): str(c) for c in df.columns}

    for candidate in candidates:
        key = _norm(candidate)
        if key in normalized:
            return normalized[key]

    for column in df.columns:
        col = _norm(column)
        for candidate in candidates:
            key = _norm(candidate)
            if key in col or col in key:
                return str(column)

    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.select_dtypes(include=np.number).columns
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    ]


def _date_columns(df: pd.DataFrame) -> list[str]:
    detected = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            detected.append(str(column))
            continue

        name = _norm(column)

        if any(
            token in name
            for token in [
                "date",
                "time",
                "timestamp",
                "month",
                "year",
            ]
        ):
            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            if parsed.notna().mean() >= 0.60:
                detected.append(str(column))

    return detected


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(k): _json_value(v)
            for k, v in record.items()
        }
        for record in frame.to_dict(orient="records")
    ]


def _numeric_target_columns(df: pd.DataFrame) -> list[str]:
    numeric = _numeric_columns(df)

    preferred_keywords = [
        "revenue",
        "sales",
        "profit",
        "amount",
        "value",
        "cost",
        "spend",
        "quantity",
        "units",
        "production",
        "output",
        "conversion",
        "click",
        "impression",
        "defect",
        "downtime",
        "salary",
        "expense",
        "margin",
        "rating",
    ]

    preferred = [
        column
        for column in numeric
        if any(
            keyword in _norm(column)
            for keyword in preferred_keywords
        )
    ]

    remaining = [
        column
        for column in numeric
        if column not in preferred
    ]

    return preferred + remaining


def _aggregate_category(
    df: pd.DataFrame,
    category: str,
    value: str,
    limit: int = 12,
) -> pd.DataFrame:
    frame = df[[category, value]].copy()

    frame[value] = pd.to_numeric(
        frame[value],
        errors="coerce",
    ).fillna(0)

    grouped = (
        frame.groupby(
            category,
            dropna=False,
        )[value]
        .sum()
        .reset_index()
        .sort_values(value, ascending=False)
        .head(limit)
    )

    return grouped


def _time_series(
    df: pd.DataFrame,
    date_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    frame = df.copy()

    frame["_nexus_date"] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["_nexus_date"]
    )

    if frame.empty:
        return pd.DataFrame()

    frame["_nexus_period"] = (
        frame["_nexus_date"]
        .dt.to_period("M")
        .astype(str)
    )

    for column in value_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0)

    result = (
        frame.groupby("_nexus_period", as_index=False)[
            value_columns
        ]
        .sum()
        .sort_values("_nexus_period")
    )

    return result


def _chart(
    chart_type: str,
    title: str,
    description: str,
    x_key: str,
    series: list[dict[str, Any]],
    data: list[dict[str, Any]],
    layout: str | None = None,
) -> dict[str, Any]:
    result = {
        "type": chart_type,
        "chartType": chart_type,
        "title": title,
        "description": description,
        "x_key": x_key,
        "xKey": x_key,
        "series": series,
        "data": data,
    }

    if layout:
        result["layout"] = layout

    return result


def build_visualizations(
    df: pd.DataFrame,
    dashboard: dict[str, Any] | None = None,
    max_charts: int = 8,
) -> list[dict[str, Any]]:
    """
    Automatically selects useful visualizations from dataset structure.

    Output is intentionally simple JSON so the existing Next.js AutoChart
    component can consume it without changing the analytics engines.
    """

    frame = df.copy()

    if frame.empty:
        return []

    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    numeric = _numeric_target_columns(frame)
    categorical = _categorical_columns(frame)
    dates = _date_columns(frame)

    charts: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()

    # ---------------------------------------------------------
    # 1. Time-series visualization
    # ---------------------------------------------------------

    if dates and numeric:
        date_column = dates[0]
        time_values = numeric[:3]

        time_frame = _time_series(
            frame,
            date_column,
            time_values,
        )

        if not time_frame.empty:
            series = [
                {
                    "data_key": column,
                    "dataKey": column,
                    "label": column,
                    "axisLabel": column,
                    "valueFormat": "compact",
                }
                for column in time_values
                if column in time_frame.columns
            ]

            data = _records(
                time_frame.rename(
                    columns={
                        "_nexus_period": "period"
                    }
                )
            )

            charts.append(
                _chart(
                    "line",
                    "Time Trend",
                    f"Monthly trend based on {date_column}.",
                    "period",
                    series,
                    data,
                )
            )

    # ---------------------------------------------------------
    # 2. Category / dimension rankings
    # ---------------------------------------------------------

    for category in categorical[:4]:
        if not numeric:
            break

        value = numeric[0]

        pair = (category, value)

        if pair in used_pairs:
            continue

        grouped = _aggregate_category(
            frame,
            category,
            value,
        )

        if grouped.empty:
            continue

        grouped = grouped.rename(
            columns={
                category: "category",
                value: "value",
            }
        )

        data = _records(grouped)

        charts.append(
            _chart(
                "bar",
                f"{value} by {category}",
                f"Top {category} groups ranked by {value}.",
                "category",
                [
                    {
                        "data_key": "value",
                        "dataKey": "value",
                        "label": value,
                        "axisLabel": value,
                        "valueFormat": "compact",
                    }
                ],
                data,
                "vertical",
            )
        )

        used_pairs.add(pair)

        if len(charts) >= max_charts:
            return charts

    # ---------------------------------------------------------
    # 3. Scatter / relationship visualization
    # ---------------------------------------------------------

    if len(numeric) >= 2:
        x_column = numeric[0]
        y_column = numeric[1]

        scatter = frame[
            [x_column, y_column]
        ].copy()

        scatter[x_column] = pd.to_numeric(
            scatter[x_column],
            errors="coerce",
        )

        scatter[y_column] = pd.to_numeric(
            scatter[y_column],
            errors="coerce",
        )

        scatter = scatter.dropna().head(500)

        if not scatter.empty:
            scatter = scatter.rename(
                columns={
                    x_column: "x",
                    y_column: "y",
                }
            )

            charts.append(
                _chart(
                    "scatter",
                    f"{x_column} vs {y_column}",
                    "Relationship between two numeric measures.",
                    "x",
                    [
                        {
                            "data_key": "y",
                            "dataKey": "y",
                            "label": y_column,
                            "axisLabel": y_column,
                            "valueFormat": "compact",
                        }
                    ],
                    _records(scatter),
                )
            )

    # ---------------------------------------------------------
    # 4. Additional categorical metric
    # ---------------------------------------------------------

    if len(numeric) >= 2 and categorical:
        category = categorical[0]
        value = numeric[1]

        grouped = _aggregate_category(
            frame,
            category,
            value,
        )

        if not grouped.empty:
            grouped = grouped.rename(
                columns={
                    category: "category",
                    value: "value",
                }
            )

            charts.append(
                _chart(
                    "bar",
                    f"{value} by {category}",
                    f"Comparison of {value} across {category}.",
                    "category",
                    [
                        {
                            "data_key": "value",
                            "dataKey": "value",
                            "label": value,
                            "axisLabel": value,
                            "valueFormat": "compact",
                        }
                    ],
                    _records(grouped),
                    "vertical",
                )
            )

    # ---------------------------------------------------------
    # 5. Numeric distribution summary
    # ---------------------------------------------------------

    if len(numeric) >= 3 and not categorical:
        distribution_column = numeric[0]

        values = pd.to_numeric(
            frame[distribution_column],
            errors="coerce",
        ).dropna()

        if len(values) >= 5:
            bins = min(10, max(5, int(np.sqrt(len(values)))))

            histogram, edges = np.histogram(
                values,
                bins=bins,
            )

            rows = []

            for index, count in enumerate(histogram):
                rows.append(
                    {
                        "range": (
                            f"{edges[index]:.2f}"
                            f"–"
                            f"{edges[index + 1]:.2f}"
                        ),
                        "count": int(count),
                    }
                )

            charts.append(
                _chart(
                    "bar",
                    f"{distribution_column} Distribution",
                    "Distribution of the selected numeric measure.",
                    "range",
                    [
                        {
                            "data_key": "count",
                            "dataKey": "count",
                            "label": "Count",
                            "axisLabel": "Count",
                            "valueFormat": "integer",
                        }
                    ],
                    rows,
                )
            )

    return charts[:max_charts]
