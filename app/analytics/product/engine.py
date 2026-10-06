from __future__ import annotations

import pandas as pd


COLUMN_ALIASES = {
    "product_id": [
        "product_id",
        "productid",
        "sku",
        "sku_id",
        "item_id",
    ],
    "product": [
        "product",
        "product_name",
        "item",
        "item_name",
        "product_title",
    ],
    "category": [
        "category",
        "product_category",
        "category_name",
        "department",
    ],
    "subcategory": [
        "subcategory",
        "sub_category",
        "product_subcategory",
    ],
    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "net_sales",
        "total_sales",
        "amount",
    ],
    "units": [
        "units",
        "units_sold",
        "quantity",
        "qty",
        "quantity_sold",
    ],
    "price": [
        "price",
        "unit_price",
        "selling_price",
        "sale_price",
        "avg_price",
    ],
    "cost": [
        "cost",
        "unit_cost",
        "cost_amount",
        "total_cost",
    ],
    "profit": [
        "profit",
        "gross_profit",
        "net_profit",
        "profit_amount",
    ],
    "orders": [
        "orders",
        "order_count",
        "number_of_orders",
        "order_id",
        "orders_count",
    ],
    "rating": [
        "rating",
        "avg_rating",
        "review_rating",
        "product_rating",
        "stars",
    ],
    "reviews": [
        "reviews",
        "review_count",
        "number_of_reviews",
        "ratings_count",
    ],
    "date": [
        "date",
        "order_date",
        "sale_date",
        "transaction_date",
        "created_at",
        "purchase_date",
    ],
    "active_users": [
        "active_users",
        "active_customers",
        "users",
        "customers",
    ],
    "engagement": [
        "engagement",
        "engagement_rate",
        "usage",
        "usage_rate",
        "views",
        "page_views",
    ],
}


def _normalize(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(columns, aliases):
    normalized = {_normalize(column): column for column in columns}

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


def detect_product_columns(df: pd.DataFrame) -> dict:
    return {
        field: find_column(df.columns, aliases)
        for field, aliases in COLUMN_ALIASES.items()
    }


def prepare_product_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = df.copy()
    columns = detect_product_columns(data)

    numeric_fields = [
        "revenue",
        "units",
        "price",
        "cost",
        "profit",
        "orders",
        "rating",
        "reviews",
        "active_users",
        "engagement",
    ]

    for field in numeric_fields:
        column = columns.get(field)

        if column:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    date_column = columns.get("date")

    if date_column:
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )

    # Derive revenue when price and units are available.
    if not columns.get("revenue"):
        price_column = columns.get("price")
        units_column = columns.get("units")

        if price_column and units_column:
            data["_derived_revenue"] = (
                data[price_column].fillna(0)
                * data[units_column].fillna(0)
            )
            columns["revenue"] = "_derived_revenue"

    # Derive profit when revenue and cost are available.
    if not columns.get("profit"):
        revenue_column = columns.get("revenue")
        cost_column = columns.get("cost")

        if revenue_column and cost_column:
            data["_derived_profit"] = (
                data[revenue_column].fillna(0)
                - data[cost_column].fillna(0)
            )
            columns["profit"] = "_derived_profit"

    return data, columns


def _sum(data: pd.DataFrame, column: str | None) -> float:
    if not column or column not in data.columns:
        return 0.0

    return float(
        pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(0).sum()
    )


def _mean(data: pd.DataFrame, column: str | None) -> float:
    if not column or column not in data.columns:
        return 0.0

    series = pd.to_numeric(
        data[column],
        errors="coerce",
    ).dropna()

    return float(series.mean()) if not series.empty else 0.0


def _nunique(data: pd.DataFrame, column: str | None) -> int:
    if not column or column not in data.columns:
        return 0

    return int(data[column].dropna().nunique())


def _count(data: pd.DataFrame, column: str | None) -> int:
    if not column or column not in data.columns:
        return 0

    return int(data[column].notna().sum())


def calculate_product_kpis(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> dict:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product")
    product_id_column = columns.get("product_id")
    revenue_column = columns.get("revenue")
    units_column = columns.get("units")
    price_column = columns.get("price")
    profit_column = columns.get("profit")
    orders_column = columns.get("orders")
    rating_column = columns.get("rating")
    reviews_column = columns.get("reviews")
    active_users_column = columns.get("active_users")
    engagement_column = columns.get("engagement")

    if product_id_column:
        product_count = _nunique(data, product_id_column)
    elif product_column:
        product_count = _nunique(data, product_column)
    else:
        product_count = len(data)

    units = _sum(data, units_column)
    revenue = _sum(data, revenue_column)
    profit = _sum(data, profit_column)

    if orders_column:
        orders = _sum(data, orders_column)
    else:
        orders = len(data)

    margin = (
        profit / revenue * 100
        if revenue
        else 0.0
    )

    return {
        "products": product_count,
        "units": units,
        "revenue": revenue,
        "average_price": _mean(data, price_column),
        "profit": profit,
        "profit_margin": margin,
        "orders": orders,
        "average_rating": _mean(data, rating_column),
        "reviews": _sum(data, reviews_column),
        "active_users": _sum(data, active_users_column),
        "engagement": _mean(data, engagement_column),
    }


def revenue_by_product(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product") or columns.get("product_id")
    revenue_column = columns.get("revenue")

    if not product_column or not revenue_column:
        return pd.DataFrame()

    result = (
        data.groupby(product_column, dropna=False)[revenue_column]
        .sum()
        .reset_index(name="Revenue")
        .sort_values("Revenue", ascending=False)
    )

    return result


def units_by_product(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product") or columns.get("product_id")
    units_column = columns.get("units")

    if not product_column or not units_column:
        return pd.DataFrame()

    return (
        data.groupby(product_column, dropna=False)[units_column]
        .sum()
        .reset_index(name="Units")
        .sort_values("Units", ascending=False)
    )


def product_performance(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product") or columns.get("product_id")

    if not product_column:
        return pd.DataFrame()

    aggregations = {}

    if columns.get("revenue"):
        aggregations["Revenue"] = (
            columns["revenue"],
            "sum",
        )

    if columns.get("units"):
        aggregations["Units"] = (
            columns["units"],
            "sum",
        )

    if columns.get("profit"):
        aggregations["Profit"] = (
            columns["profit"],
            "sum",
        )

    if columns.get("rating"):
        aggregations["Rating"] = (
            columns["rating"],
            "mean",
        )

    if columns.get("reviews"):
        aggregations["Reviews"] = (
            columns["reviews"],
            "sum",
        )

    if not aggregations:
        return (
            data.groupby(product_column, dropna=False)
            .size()
            .reset_index(name="Records")
            .sort_values("Records", ascending=False)
        )

    result = (
        data.groupby(product_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    if "Revenue" in result.columns:
        result["Profit Margin %"] = (
            result["Profit"] / result["Revenue"] * 100
            if "Profit" in result.columns
            else 0
        )

    return result.sort_values(
        "Revenue" if "Revenue" in result.columns else result.columns[-1],
        ascending=False,
    )


def revenue_by_category(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    category_column = columns.get("category")
    revenue_column = columns.get("revenue")

    if not category_column or not revenue_column:
        return pd.DataFrame()

    return (
        data.groupby(category_column, dropna=False)[revenue_column]
        .sum()
        .reset_index(name="Revenue")
        .sort_values("Revenue", ascending=False)
    )


def category_performance(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    category_column = columns.get("category")

    if not category_column:
        return pd.DataFrame()

    aggregations = {}

    if columns.get("revenue"):
        aggregations["Revenue"] = (
            columns["revenue"],
            "sum",
        )

    if columns.get("units"):
        aggregations["Units"] = (
            columns["units"],
            "sum",
        )

    if columns.get("profit"):
        aggregations["Profit"] = (
            columns["profit"],
            "sum",
        )

    if columns.get("rating"):
        aggregations["Rating"] = (
            columns["rating"],
            "mean",
        )

    if not aggregations:
        return pd.DataFrame()

    result = (
        data.groupby(category_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    if "Revenue" in result.columns and "Profit" in result.columns:
        result["Profit Margin %"] = (
            result["Profit"]
            / result["Revenue"]
            * 100
        )

    return result.sort_values(
        "Revenue" if "Revenue" in result.columns else result.columns[-1],
        ascending=False,
    )


def price_vs_performance(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product") or columns.get("product_id")
    price_column = columns.get("price")

    if not product_column or not price_column:
        return pd.DataFrame()

    aggregations = {
        "Average Price": (
            price_column,
            "mean",
        )
    }

    if columns.get("units"):
        aggregations["Units"] = (
            columns["units"],
            "sum",
        )

    if columns.get("revenue"):
        aggregations["Revenue"] = (
            columns["revenue"],
            "sum",
        )

    if columns.get("profit"):
        aggregations["Profit"] = (
            columns["profit"],
            "sum",
        )

    return (
        data.groupby(product_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("Average Price", ascending=False)
    )


def ratings_and_reviews(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    product_column = columns.get("product") or columns.get("product_id")

    if not product_column:
        return pd.DataFrame()

    aggregations = {}

    if columns.get("rating"):
        aggregations["Average Rating"] = (
            columns["rating"],
            "mean",
        )

    if columns.get("reviews"):
        aggregations["Reviews"] = (
            columns["reviews"],
            "sum",
        )

    if not aggregations:
        return pd.DataFrame()

    return (
        data.groupby(product_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(
            "Average Rating"
            if "Average Rating" in aggregations
            else "Reviews",
            ascending=False,
        )
    )


def product_trend(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    date_column = columns.get("date")
    revenue_column = columns.get("revenue")

    if not date_column or not revenue_column:
        return pd.DataFrame()

    data = data.dropna(subset=[date_column]).copy()
    data["Period"] = data[date_column].dt.to_period("M").astype(str)

    result = (
        data.groupby("Period")[revenue_column]
        .sum()
        .reset_index(name="Revenue")
    )

    return result.sort_values("Period")


def generate_product_insights(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> list[str]:
    data, detected = prepare_product_data(df)
    columns = columns or detected
    insights = []

    product_perf = product_performance(data, columns)

    if not product_perf.empty:
        product_column = (
            columns.get("product")
            or columns.get("product_id")
        )

        if (
            product_column
            and "Revenue" in product_perf.columns
        ):
            top = product_perf.iloc[0]
            insights.append(
                f"Top product by revenue: "
                f"{top[product_column]} "
                f"({top['Revenue']:,.2f})."
            )

        if (
            product_column
            and "Profit" in product_perf.columns
        ):
            best_profit = product_perf.sort_values(
                "Profit",
                ascending=False,
            ).iloc[0]

            insights.append(
                f"Highest-profit product: "
                f"{best_profit[product_column]} "
                f"({best_profit['Profit']:,.2f})."
            )

    category_perf = category_performance(data, columns)

    if (
        not category_perf.empty
        and "Revenue" in category_perf.columns
    ):
        category_column = columns.get("category")

        if category_column:
            top_category = category_perf.iloc[0]

            insights.append(
                f"Top category by revenue: "
                f"{top_category[category_column]} "
                f"({top_category['Revenue']:,.2f})."
            )

    if columns.get("rating"):
        avg_rating = _mean(
            data,
            columns["rating"],
        )

        insights.append(
            f"Average product rating is "
            f"{avg_rating:.2f}."
        )

    if columns.get("reviews"):
        total_reviews = _sum(
            data,
            columns["reviews"],
        )

        insights.append(
            f"Total recorded reviews: "
            f"{total_reviews:,.0f}."
        )

    if not insights:
        insights.append(
            "Add product, revenue, quantity, "
            "price, rating, or category fields "
            "for deeper product insights."
        )

    return insights


def product_summary(
    df: pd.DataFrame,
    columns: dict | None = None,
) -> pd.DataFrame:
    data, detected = prepare_product_data(df)
    columns = columns or detected

    kpis = calculate_product_kpis(data, columns)

    return pd.DataFrame(
        [
            {
                "Metric": "Products",
                "Value": kpis["products"],
            },
            {
                "Metric": "Units",
                "Value": kpis["units"],
            },
            {
                "Metric": "Revenue",
                "Value": kpis["revenue"],
            },
            {
                "Metric": "Average Price",
                "Value": kpis["average_price"],
            },
            {
                "Metric": "Profit",
                "Value": kpis["profit"],
            },
            {
                "Metric": "Profit Margin %",
                "Value": kpis["profit_margin"],
            },
            {
                "Metric": "Orders",
                "Value": kpis["orders"],
            },
            {
                "Metric": "Average Rating",
                "Value": kpis["average_rating"],
            },
        ]
    )
