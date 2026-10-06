import pandas as pd


# =========================================================
# CUSTOMER COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {
    "customer_id": [
        "customer_id",
        "customer id",
        "client_id",
        "client id",
        "customer_number",
        "customer number",
        "client_number",
        "client number",
    ],

    "customer_name": [
        "customer_name",
        "customer name",
        "client_name",
        "client name",
        "customer_full_name",
        "customer full name",
    ],

    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "sales amount",
        "total_spend",
        "total spend",
        "customer_value",
        "customer value",
        "amount",
    ],

    "orders": [
        "orders",
        "order_count",
        "order count",
        "number_of_orders",
        "number of orders",
        "purchases",
        "purchase_count",
        "purchase count",
    ],

    "order_id": [
        "order_id",
        "order id",
        "transaction_id",
        "transaction id",
        "purchase_id",
        "purchase id",
    ],

    "segment": [
        "segment",
        "customer_segment",
        "customer segment",
        "customer_tier",
        "customer tier",
        "tier",
    ],

    "churn": [
        "churn",
        "churned",
        "is_churned",
        "is churned",
        "churn_status",
        "churn status",
    ],

    "retention": [
        "retention",
        "retained",
        "is_retained",
        "is retained",
        "retention_status",
        "retention status",
    ],

    "clv": [
        "clv",
        "customer_lifetime_value",
        "customer lifetime value",
        "lifetime_value",
        "lifetime value",
    ],

    "age": [
        "age",
        "customer_age",
        "customer age",
    ],

    "gender": [
        "gender",
        "sex",
    ],

    "region": [
        "region",
        "location",
        "city",
        "state",
        "territory",
    ],

    "date": [
        "date",
        "order_date",
        "order date",
        "purchase_date",
        "purchase date",
        "transaction_date",
        "transaction date",
    ],
}


# =========================================================
# NORMALIZE COLUMN
# =========================================================

def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(df, aliases):

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # Exact matching first
    for alias in aliases:

        normalized_alias = (
            normalize_column_name(alias)
        )

        if normalized_alias in normalized_columns:

            return normalized_columns[
                normalized_alias
            ]

    # Carefully allow prefix/suffix matching
    # but NEVER arbitrary substring matching.
    for column in df.columns:

        normalized_column = (
            normalize_column_name(column)
        )

        for alias in aliases:

            normalized_alias = (
                normalize_column_name(alias)
            )

            if (
                normalized_column.startswith(
                    normalized_alias + "_"
                )
                or normalized_column.endswith(
                    "_" + normalized_alias
                )
            ):

                return column

    return None


# =========================================================
# DETECT CUSTOMER COLUMNS
# =========================================================

def detect_customer_columns(df):

    mapping = {}

    for field, aliases in COLUMN_ALIASES.items():

        mapping[field] = find_column(
            df,
            aliases
        )

    return mapping


# =========================================================
# PREPARE CUSTOMER DATA
# =========================================================

def prepare_customer_data(df):

    df = df.copy()

    mapping = detect_customer_columns(
        df
    )

    numeric_fields = [
        "revenue",
        "orders",
        "clv",
        "age",
    ]

    for field in numeric_fields:

        column = mapping.get(field)

        if column:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if mapping["date"]:

        df[mapping["date"]] = pd.to_datetime(
            df[mapping["date"]],
            errors="coerce"
        )

    return df, mapping


# =========================================================
# CUSTOMER KPIs
# =========================================================

def calculate_customer_kpis(
    df,
    mapping
):

    # -----------------------------------------------------
    # CUSTOMER COUNT
    # -----------------------------------------------------

    if mapping["customer_id"]:

        total_customers = int(
            df[mapping["customer_id"]]
            .nunique()
        )

        customer_count_type = (
            "Unique customers"
        )

    else:

        total_customers = len(df)

        customer_count_type = (
            "Customer records"
        )

    # -----------------------------------------------------
    # REVENUE
    # -----------------------------------------------------

    if mapping["revenue"]:

        total_revenue = float(
            df[mapping["revenue"]]
            .sum()
        )

    else:

        total_revenue = 0.0

    # -----------------------------------------------------
    # ORDERS
    # -----------------------------------------------------

    if mapping["order_id"]:

        total_orders = int(
            df[mapping["order_id"]]
            .nunique()
        )

    elif mapping["orders"]:

        total_orders = int(
            df[mapping["orders"]]
            .sum()
        )

    else:

        total_orders = len(df)

    # -----------------------------------------------------
    # CUSTOMER VALUE
    # -----------------------------------------------------

    average_customer_value = (
        total_revenue / total_customers
        if total_customers
        else 0
    )

    # -----------------------------------------------------
    # ORDER VALUE
    # -----------------------------------------------------

    average_order_value = (
        total_revenue / total_orders
        if total_orders
        else 0
    )

    # -----------------------------------------------------
    # CHURN
    # -----------------------------------------------------

    churn_rate = None

    if mapping["churn"]:

        churn_series = (
            df[mapping["churn"]]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        churn_values = churn_series.isin(
            [
                "1",
                "true",
                "yes",
                "y",
                "churned",
            ]
        )

        churn_rate = (
            churn_values.mean() * 100
        )

    # -----------------------------------------------------
    # RETENTION
    # -----------------------------------------------------

    retention_rate = None

    if mapping["retention"]:

        retention_series = (
            df[mapping["retention"]]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        retained_values = retention_series.isin(
            [
                "1",
                "true",
                "yes",
                "y",
                "retained",
            ]
        )

        retention_rate = (
            retained_values.mean() * 100
        )

    # Derive missing metric when possible
    if (
        retention_rate is None
        and churn_rate is not None
    ):

        retention_rate = (
            100 - churn_rate
        )

    if (
        churn_rate is None
        and retention_rate is not None
    ):

        churn_rate = (
            100 - retention_rate
        )

    # -----------------------------------------------------
    # CLV
    # -----------------------------------------------------

    average_clv = None

    if mapping["clv"]:

        average_clv = float(
            df[mapping["clv"]]
            .mean()
        )

    return {
        "total_customers": total_customers,
        "customer_count_type": customer_count_type,
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "average_customer_value": average_customer_value,
        "average_order_value": average_order_value,
        "churn_rate": churn_rate,
        "retention_rate": retention_rate,
        "average_clv": average_clv,
    }


# =========================================================
# REVENUE BY SEGMENT
# =========================================================

def revenue_by_segment(
    df,
    mapping
):

    segment_column = mapping["segment"]
    revenue_column = mapping["revenue"]

    if (
        not segment_column
        or not revenue_column
    ):

        return None

    return (
        df.groupby(
            segment_column,
            dropna=False
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


# =========================================================
# CUSTOMERS BY SEGMENT
# =========================================================

def customers_by_segment(
    df,
    mapping
):

    segment_column = mapping["segment"]

    if not segment_column:

        return None

    customer_column = mapping["customer_id"]

    # Real customer count
    if customer_column:

        result = (
            df.groupby(
                segment_column,
                dropna=False
            )[customer_column]
            .nunique()
            .reset_index(
                name="Customers"
            )
        )

    # No Customer ID available:
    # count records instead of pretending
    # they are unique customers.
    else:

        result = (
            df.groupby(
                segment_column,
                dropna=False
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

    return result.sort_values(
        "Customers",
        ascending=False
    )


# =========================================================
# REVENUE BY REGION
# =========================================================

def revenue_by_region(
    df,
    mapping
):

    region_column = mapping["region"]
    revenue_column = mapping["revenue"]

    if (
        not region_column
        or not revenue_column
    ):

        return None

    return (
        df.groupby(
            region_column,
            dropna=False
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


# =========================================================
# TOP CUSTOMERS
# =========================================================

def top_customers(
    df,
    mapping,
    limit=10
):

    customer_column = mapping["customer_id"]
    revenue_column = mapping["revenue"]

    if (
        not customer_column
        or not revenue_column
    ):

        return None

    result = (
        df.groupby(
            customer_column,
            dropna=False
        )[revenue_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(limit)
        .reset_index()
    )

    result.columns = [
        customer_column,
        "Revenue",
    ]

    return result


# =========================================================
# CUSTOMER SUMMARY
# =========================================================

def customer_summary(
    df,
    mapping
):

    customer_column = mapping["customer_id"]

    if not customer_column:

        return None

    aggregation = {}

    if mapping["revenue"]:

        aggregation["Revenue"] = (
            mapping["revenue"],
            "sum"
        )

    if mapping["order_id"]:

        aggregation["Orders"] = (
            mapping["order_id"],
            "nunique"
        )

    elif mapping["orders"]:

        aggregation["Orders"] = (
            mapping["orders"],
            "sum"
        )

    if mapping["clv"]:

        aggregation["CLV"] = (
            mapping["clv"],
            "mean"
        )

    if not aggregation:

        return None

    return (
        df.groupby(
            customer_column,
            dropna=False
        )
        .agg(**aggregation)
        .reset_index()
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

def generate_customer_insights(
    df,
    mapping
):

    insights = []

    kpis = calculate_customer_kpis(
        df,
        mapping
    )

    # Customer count
    if mapping["customer_id"]:

        insights.append(
            f"The dataset contains "
            f"{kpis['total_customers']:,} "
            f"unique customers."
        )

    else:

        insights.append(
            f"The dataset contains "
            f"{kpis['total_customers']:,} "
            f"customer records, but no dedicated "
            f"Customer ID field was detected."
        )

    # Revenue
    if kpis["total_revenue"] > 0:

        insights.append(
            f"Total customer revenue is "
            f"₹{kpis['total_revenue']:,.0f}."
        )

    # Average customer value
    if kpis["average_customer_value"] > 0:

        insights.append(
            f"Average revenue per customer record "
            f"is approximately "
            f"₹{kpis['average_customer_value']:,.2f}."
        )

    # Average order value
    if kpis["average_order_value"] > 0:

        insights.append(
            f"Average order value is "
            f"₹{kpis['average_order_value']:,.2f}."
        )

    # Retention
    if kpis["retention_rate"] is not None:

        insights.append(
            f"Observed retention rate is "
            f"{kpis['retention_rate']:.2f}%."
        )

    # Churn
    if kpis["churn_rate"] is not None:

        insights.append(
            f"Observed churn rate is "
            f"{kpis['churn_rate']:.2f}%."
        )

    # CLV
    if kpis["average_clv"] is not None:

        insights.append(
            f"Average customer lifetime value "
            f"is ₹{kpis['average_clv']:,.2f}."
        )

    # Best segment
    segment_data = revenue_by_segment(
        df,
        mapping
    )

    if (
        segment_data is not None
        and not segment_data.empty
    ):

        segment_column = mapping["segment"]
        revenue_column = mapping["revenue"]

        best_segment = segment_data.iloc[0]

        insights.append(
            f"The highest-revenue customer segment "
            f"is {best_segment[segment_column]} "
            f"with ₹{best_segment[revenue_column]:,.0f}."
        )

    # Top customer
    top_data = top_customers(
        df,
        mapping
    )

    if (
        top_data is not None
        and not top_data.empty
    ):

        top_customer = top_data.iloc[0]

        insights.append(
            f"The highest-value customer generated "
            f"₹{top_customer['Revenue']:,.0f}."
        )

    return insights