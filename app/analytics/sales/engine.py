import pandas as pd


COLUMN_ALIASES = {

    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "sales amount",
        "total_sales",
        "total sales",
        "net_sales",
        "net sales",
        "amount",
        "order_value",
        "order value",
    ],

    "profit": [
        "profit",
        "net_profit",
        "net profit",
        "gross_profit",
        "gross profit",
        "profit_amount",
        "profit amount",
    ],

    "quantity": [
        "quantity",
        "qty",
        "units",
        "units_sold",
        "units sold",
    ],

    "order_id": [
        "order_id",
        "order id",
        "order",
        "transaction_id",
        "transaction id",
    ],

    "product": [
        "product",
        "product_name",
        "product name",
        "item",
        "item_name",
        "item name",
    ],

    "category": [
        "category",
        "product_category",
        "product category",
        "segment",
    ],

    "customer": [
        "customer",
        "customer_name",
        "customer name",
        "customer_id",
        "customer id",
    ],

    "region": [
        "region",
        "area",
        "territory",
        "location",
        "state",
        "city",
    ],

    "date": [
        "date",
        "order_date",
        "order date",
        "sales_date",
        "sales date",
        "transaction_date",
        "transaction date",
    ],
}


def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def find_column(df, aliases):

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # Exact matching
    for alias in aliases:

        normalized_alias = (
            normalize_column_name(alias)
        )

        if normalized_alias in normalized_columns:

            return normalized_columns[
                normalized_alias
            ]

    # Partial matching
    for column in df.columns:

        normalized_column = (
            normalize_column_name(column)
        )

        for alias in aliases:

            normalized_alias = (
                normalize_column_name(alias)
            )

            if (
                normalized_alias
                in normalized_column
            ):

                return column

    return None


def detect_sales_columns(df):

    mapping = {}

    for field, aliases in (
        COLUMN_ALIASES.items()
    ):

        mapping[field] = find_column(
            df,
            aliases
        )

    return mapping


def prepare_sales_data(df):

    df = df.copy()

    mapping = detect_sales_columns(
        df
    )

    # Date
    if mapping["date"]:

        df[
            mapping["date"]
        ] = pd.to_datetime(
            df[mapping["date"]],
            errors="coerce"
        )

    # Numeric columns
    for field in [
        "revenue",
        "profit",
        "quantity",
    ]:

        column = mapping[field]

        if column:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df, mapping


def calculate_sales_kpis(
    df,
    mapping
):

    revenue_column = mapping[
        "revenue"
    ]

    profit_column = mapping[
        "profit"
    ]

    quantity_column = mapping[
        "quantity"
    ]

    order_column = mapping[
        "order_id"
    ]

    revenue = (
        float(
            df[revenue_column].sum()
        )
        if revenue_column
        else 0
    )

    profit = (
        float(
            df[profit_column].sum()
        )
        if profit_column
        else 0
    )

    quantity = (
        float(
            df[quantity_column].sum()
        )
        if quantity_column
        else 0
    )

    orders = (
        int(
            df[order_column].nunique()
        )
        if order_column
        else len(df)
    )

    margin = (
        profit / revenue * 100
        if revenue
        else 0
    )

    average_order_value = (
        revenue / orders
        if orders
        else 0
    )

    return {
        "revenue": revenue,
        "profit": profit,
        "quantity": quantity,
        "orders": orders,
        "margin": margin,
        "average_order_value": (
            average_order_value
        ),
    }


def revenue_by_region(
    df,
    mapping
):

    revenue_column = mapping[
        "revenue"
    ]

    region_column = mapping[
        "region"
    ]

    if not revenue_column:
        return None

    if not region_column:
        return None

    return (
        df.groupby(
            region_column
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def revenue_by_product(
    df,
    mapping
):

    revenue_column = mapping[
        "revenue"
    ]

    product_column = mapping[
        "product"
    ]

    if not revenue_column:
        return None

    if not product_column:
        return None

    return (
        df.groupby(
            product_column
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
        .head(10)
    )


def revenue_by_category(
    df,
    mapping
):

    revenue_column = mapping[
        "revenue"
    ]

    category_column = mapping[
        "category"
    ]

    if not revenue_column:
        return None

    if not category_column:
        return None

    return (
        df.groupby(
            category_column
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def revenue_over_time(
    df,
    mapping
):

    date_column = mapping[
        "date"
    ]

    revenue_column = mapping[
        "revenue"
    ]

    if not date_column:
        return None

    if not revenue_column:
        return None

    temp = df[
        [
            date_column,
            revenue_column
        ]
    ].dropna()

    if temp.empty:
        return None

    temp["period"] = (
        temp[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    return (
        temp.groupby(
            "period"
        )[revenue_column]
        .sum()
        .reset_index()
    )