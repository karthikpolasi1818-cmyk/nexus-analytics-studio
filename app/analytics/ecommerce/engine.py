import re
import numpy as np
import pandas as pd


ALIASES = {
    "order_id": [
        "order_id", "orderid", "order_number", "order_no", "transaction_id",
        "invoice_id", "id"
    ],
    "customer_id": [
        "customer_id", "customerid", "customer_number", "customer_no", "client_id"
    ],
    "order_date": [
        "order_date", "orderdate", "date", "transaction_date", "purchase_date",
        "created_date", "created_at"
    ],
    "product": [
        "product", "product_name", "item", "item_name", "sku_name"
    ],
    "category": [
        "category", "product_category", "product_type", "department"
    ],
    "quantity": [
        "quantity", "qty", "units", "units_sold", "items"
    ],
    "unit_price": [
        "unit_price", "price", "selling_price", "item_price"
    ],
    "revenue": [
        "revenue", "sales", "sales_amount", "total_sales", "order_value",
        "amount", "net_sales"
    ],
    "discount": [
        "discount", "discount_amount", "discount_value", "discount_pct",
        "discount_percent"
    ],
    "cost": [
        "cost", "cost_amount", "product_cost", "cogs"
    ],
    "profit": [
        "profit", "profit_amount", "gross_profit", "net_profit"
    ],
    "channel": [
        "channel", "sales_channel", "order_channel", "platform", "source"
    ],
    "payment_method": [
        "payment_method", "payment", "payment_type", "pay_method"
    ],
    "order_status": [
        "order_status", "status", "order_state", "fulfillment_status"
    ],
    "customer_segment": [
        "customer_segment", "segment", "customer_type", "customer_group"
    ],
    "city": ["city", "customer_city", "shipping_city"],
    "state": ["state", "customer_state", "shipping_state", "region"],
    "country": ["country", "customer_country", "shipping_country"],
    "rating": ["rating", "customer_rating", "review_rating", "product_rating"],
}


def _norm(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _resolve_column(df, aliases):
    normalized = {_norm(c): c for c in df.columns}

    # Exact aliases first.
    for alias in aliases:
        key = _norm(alias)
        if key in normalized:
            return normalized[key]

    # Conservative prefix/suffix matching.
    for alias in aliases:
        key = _norm(alias)
        for n, original in normalized.items():
            if n.startswith(key + "_") or n.endswith("_" + key):
                return original

    return None


def _find_columns(df):
    found = {}
    for canonical, aliases in ALIASES.items():
        col = _resolve_column(df, aliases)
        if col is not None:
            found[canonical] = col
    return found


def _num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def prepare_ecommerce_data(df):
    data = df.copy()
    mapping = _find_columns(data)

    for canonical, source in mapping.items():
        if canonical not in data.columns:
            data[canonical] = data[source]

    for col in [
        "quantity", "unit_price", "revenue", "discount",
        "cost", "profit", "rating"
    ]:
        if col in data.columns:
            data[col] = _num(data[col])

    if "order_date" in data.columns:
        data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")

    if "revenue" not in data.columns and {"quantity", "unit_price"}.issubset(data.columns):
        data["revenue"] = data["quantity"].fillna(0) * data["unit_price"].fillna(0)

    if "profit" not in data.columns and {"revenue", "cost"}.issubset(data.columns):
        data["profit"] = data["revenue"].fillna(0) - data["cost"].fillna(0)

    if "discount" in data.columns and data["discount"].dropna().size:
        if data["discount"].dropna().abs().max() <= 1.0:
            data["discount_pct"] = data["discount"] * 100
        else:
            data["discount_pct"] = data["discount"]

    data.attrs["ecommerce_mapping"] = mapping
    return data


def calculate_ecommerce_kpis(df):
    data = prepare_ecommerce_data(df)
    kpis = {}

    if "order_id" in data.columns:
        kpis["Total Orders"] = int(data["order_id"].nunique())
    else:
        kpis["Total Orders"] = int(len(data))

    if "customer_id" in data.columns:
        kpis["Unique Customers"] = int(data["customer_id"].nunique())

    if "revenue" in data.columns:
        revenue = data["revenue"].sum()
        kpis["Revenue"] = float(revenue)
        kpis["Average Order Value"] = (
            float(revenue / kpis["Total Orders"])
            if kpis["Total Orders"] else 0.0
        )

    if "profit" in data.columns:
        profit = data["profit"].sum()
        kpis["Profit"] = float(profit)
        if "revenue" in data.columns and data["revenue"].sum():
            kpis["Profit Margin"] = float(profit / data["revenue"].sum() * 100)

    if "quantity" in data.columns:
        kpis["Units Sold"] = float(data["quantity"].sum())

    if "discount_pct" in data.columns:
        kpis["Average Discount"] = float(data["discount_pct"].mean())

    if "rating" in data.columns:
        kpis["Average Rating"] = float(data["rating"].mean())

    if "order_status" in data.columns:
        delivered = data["order_status"].astype(str).str.lower().eq("delivered").sum()
        kpis["Delivered Orders"] = int(delivered)
        kpis["Delivery Rate"] = (
            float(delivered / len(data) * 100) if len(data) else 0.0
        )

    return kpis


def _group_sum(data, group_col, value_col="revenue"):
    if group_col not in data.columns or value_col not in data.columns:
        return pd.DataFrame()
    out = (
        data.groupby(group_col, dropna=False)[value_col]
        .sum()
        .reset_index()
        .sort_values(value_col, ascending=False)
    )
    return out


def sales_by_category(df):
    data = prepare_ecommerce_data(df)
    return _group_sum(data, "category")


def sales_by_product(df):
    data = prepare_ecommerce_data(df)
    return _group_sum(data, "product")


def sales_by_channel(df):
    data = prepare_ecommerce_data(df)
    return _group_sum(data, "channel")


def orders_by_channel(df):
    data = prepare_ecommerce_data(df)
    if "channel" not in data.columns:
        return pd.DataFrame()
    if "order_id" in data.columns:
        out = data.groupby("channel")["order_id"].nunique().reset_index(name="Orders")
    else:
        out = data.groupby("channel").size().reset_index(name="Orders")
    return out.sort_values("Orders", ascending=False)


def customer_segments(df):
    data = prepare_ecommerce_data(df)
    if "customer_segment" not in data.columns:
        return pd.DataFrame()

    agg = {"Revenue": ("revenue", "sum")} if "revenue" in data.columns else {}
    if "order_id" in data.columns:
        agg["Orders"] = ("order_id", "nunique")
    if "customer_id" in data.columns:
        agg["Customers"] = ("customer_id", "nunique")

    if not agg:
        return data["customer_segment"].value_counts().reset_index(
            name="Customers"
        )

    out = data.groupby("customer_segment", dropna=False).agg(**agg).reset_index()
    return out.sort_values(
        "Revenue" if "Revenue" in out.columns else out.columns[-1],
        ascending=False,
    )


def daily_sales_trend(df):
    data = prepare_ecommerce_data(df)
    if "order_date" not in data.columns or "revenue" not in data.columns:
        return pd.DataFrame()
    valid = data.dropna(subset=["order_date"])
    out = (
        valid.groupby(valid["order_date"].dt.date)["revenue"]
        .sum()
        .reset_index(name="Revenue")
        .rename(columns={"order_date": "Date"})
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date")


def payment_method_analysis(df):
    data = prepare_ecommerce_data(df)
    if "payment_method" not in data.columns:
        return pd.DataFrame()
    if "revenue" in data.columns:
        out = data.groupby("payment_method")["revenue"].sum().reset_index(name="Revenue")
    else:
        out = data["payment_method"].value_counts().reset_index()
        out.columns = ["payment_method", "Orders"]
    return out.sort_values(out.columns[-1], ascending=False)


def order_status_analysis(df):
    data = prepare_ecommerce_data(df)
    if "order_status" not in data.columns:
        return pd.DataFrame()
    if "order_id" in data.columns:
        out = data.groupby("order_status")["order_id"].nunique().reset_index(name="Orders")
    else:
        out = data["order_status"].value_counts().reset_index()
        out.columns = ["order_status", "Orders"]
    out["Share (%)"] = (
        out["Orders"] / out["Orders"].sum() * 100 if out["Orders"].sum() else 0
    )
    return out.sort_values("Orders", ascending=False)


def geographic_sales(df):
    data = prepare_ecommerce_data(df)
    group_col = next(
        (c for c in ["country", "state", "city"] if c in data.columns),
        None,
    )
    if group_col is None:
        return pd.DataFrame()
    return _group_sum(data, group_col)


def generate_ecommerce_insights(df):
    data = prepare_ecommerce_data(df)
    insights = []
    kpis = calculate_ecommerce_kpis(data)

    if "Revenue" in kpis:
        insights.append(f"Total revenue is {kpis['Revenue']:,.2f}.")
    if "Average Order Value" in kpis:
        insights.append(f"Average order value is {kpis['Average Order Value']:,.2f}.")

    if "category" in data.columns and "revenue" in data.columns and len(data):
        category = data.groupby("category")["revenue"].sum().sort_values(ascending=False)
        if len(category):
            insights.append(
                f"Top revenue category is {category.index[0]} "
                f"with {category.iloc[0]:,.2f} in revenue."
            )

    if "channel" in data.columns and "revenue" in data.columns and len(data):
        channel = data.groupby("channel")["revenue"].sum().sort_values(ascending=False)
        if len(channel):
            insights.append(
                f"Highest-revenue channel is {channel.index[0]} "
                f"with {channel.iloc[0]:,.2f}."
            )

    if "order_status" in data.columns and len(data):
        cancelled = data["order_status"].astype(str).str.lower().eq("cancelled").mean() * 100
        if cancelled > 0:
            insights.append(f"Cancelled orders represent {cancelled:.1f}% of rows.")

    if "rating" in data.columns:
        rating = data["rating"].mean()
        insights.append(f"Average available customer/product rating is {rating:.2f}.")

    return insights


def ecommerce_summary(df):
    return {
        "kpis": calculate_ecommerce_kpis(df),
        "insights": generate_ecommerce_insights(df),
        "detected_fields": _find_columns(df),
    }
