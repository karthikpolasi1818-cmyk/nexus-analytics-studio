import pandas as pd


def profile_dataframe(df: pd.DataFrame):

    rows = len(df)

    columns = len(df.columns)

    total_cells = rows * columns

    missing_cells = int(
        df.isna().sum().sum()
    )

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    numeric_columns = list(
        df.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns
    )

    datetime_columns = list(
        df.select_dtypes(
            include="datetime"
        ).columns
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "column_names": list(
            df.columns
        ),
    }