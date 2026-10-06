import pandas as pd
import numpy as np


COLUMN_ALIASES = {
    "supplier": [
        "supplier",
        "supplier_name",
        "vendor",
        "vendor_name",
    ],
    "supplier_id": [
        "supplier_id",
        "vendor_id",
    ],
    "inventory": [
        "inventory",
        "inventory_level",
        "stock",
        "stock_level",
        "on_hand",
        "quantity_in_stock",
    ],
    "quantity": [
        "quantity",
        "qty",
        "units",
        "units_ordered",
        "order_quantity",
    ],
    "shipment": [
        "shipment",
        "shipment_id",
        "shipments",
        "shipping",
    ],
    "delivery": [
        "delivery",
        "delivery_time",
        "delivery_days",
        "delivery_date",
        "delivered",
    ],
    "lead_time": [
        "lead_time",
        "leadtime",
        "supplier_lead_time",
        "procurement_lead_time",
    ],
    "order_id": [
        "order_id",
        "orderid",
        "purchase_order_id",
        "po_id",
    ],
    "product": [
        "product",
        "product_name",
        "sku",
        "item",
        "item_name",
    ],
    "warehouse": [
        "warehouse",
        "warehouse_name",
        "distribution_center",
        "dc",
        "location",
    ],
    "region": [
        "region",
        "zone",
        "area",
        "territory",
    ],
    "cost": [
        "cost",
        "unit_cost",
        "total_cost",
        "purchase_cost",
    ],
    "date": [
        "date",
        "order_date",
        "shipment_date",
        "delivery_date",
        "transaction_date",
        "month",
        "period",
    ],
    "on_time": [
        "on_time",
        "on_time_delivery",
        "ontime",
        "delivered_on_time",
        "delivery_status",
    ],
}


def _normalise(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, aliases):
    normalized_columns = {
        _normalise(column): column
        for column in df.columns
    }

    normalized_aliases = [
        _normalise(alias)
        for alias in aliases
    ]

    # Exact match first
    for alias in normalized_aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    # Safe prefix/suffix matching
    for normalized_column, original_column in normalized_columns.items():
        for alias in normalized_aliases:
            if (
                normalized_column.startswith(alias + "_")
                or normalized_column.endswith("_" + alias)
            ):
                return original_column

    return None


def detect_supply_chain_columns(df):
    return {
        field: find_column(df, aliases)
        for field, aliases in COLUMN_ALIASES.items()
    }


def _numeric(result, column):
    if column:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )


def prepare_supply_chain_data(df):
    result = df.copy()

    mapping = detect_supply_chain_columns(
        result
    )

    for field in [
        "inventory",
        "quantity",
        "delivery",
        "lead_time",
        "cost",
    ]:
        _numeric(
            result,
            mapping.get(field),
        )

    date_column = mapping.get("date")

    if date_column:
        result[date_column] = pd.to_datetime(
            result[date_column],
            errors="coerce",
        )

    on_time_column = mapping.get("on_time")

    if on_time_column:

        if result[on_time_column].dtype == object:

            normalized = (
                result[on_time_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            result["_On_Time_Flag"] = normalized.map(
                {
                    "yes": 1,
                    "y": 1,
                    "true": 1,
                    "on time": 1,
                    "on-time": 1,
                    "delivered": 1,
                    "no": 0,
                    "n": 0,
                    "false": 0,
                    "late": 0,
                    "delayed": 0,
                }
            )

        else:

            result["_On_Time_Flag"] = pd.to_numeric(
                result[on_time_column],
                errors="coerce",
            )

    return result, mapping


def calculate_supply_chain_kpis(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    def total(field):

        column = mapping.get(field)

        if not column:
            return 0.0

        return float(
            data[column]
            .fillna(0)
            .sum()
        )

    def average(field):

        column = mapping.get(field)

        if not column:
            return 0.0

        valid = data[column].dropna()

        if valid.empty:
            return 0.0

        return float(
            valid.mean()
        )

    inventory = average("inventory")
    quantity = total("quantity")
    delivery = average("delivery")
    lead_time = average("lead_time")
    cost = total("cost")

    order_column = mapping.get(
        "order_id"
    )

    supplier_column = mapping.get(
        "supplier"
    )

    shipment_column = mapping.get(
        "shipment"
    )

    orders = (
        int(
            data[order_column]
            .nunique()
        )
        if order_column
        else int(len(data))
    )

    suppliers = (
        int(
            data[supplier_column]
            .nunique()
        )
        if supplier_column
        else 0
    )

    shipments = (
        int(
            data[shipment_column]
            .nunique()
        )
        if shipment_column
        else 0
    )

    on_time = 0.0

    if "_On_Time_Flag" in data.columns:

        valid = data[
            "_On_Time_Flag"
        ].dropna()

        if not valid.empty:
            on_time = float(
                valid.mean() * 100
            )

    return {
        "Orders": orders,
        "Suppliers": suppliers,
        "Shipments": shipments,
        "Inventory": inventory,
        "Quantity": quantity,
        "Average Delivery Time": delivery,
        "Average Lead Time": lead_time,
        "Supply Chain Cost": cost,
        "On-Time Delivery": on_time,
    }


def inventory_by_product(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    product = mapping.get("product")
    inventory = mapping.get("inventory")

    if not product or not inventory:
        return pd.DataFrame()

    result = (
        data.groupby(
            product,
            dropna=False,
        )[inventory]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Product",
        "Inventory",
    ]

    return result.sort_values(
        "Inventory",
        ascending=False,
    )


def inventory_by_warehouse(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    warehouse = mapping.get(
        "warehouse"
    )

    inventory = mapping.get(
        "inventory"
    )

    if not warehouse or not inventory:
        return pd.DataFrame()

    result = (
        data.groupby(
            warehouse,
            dropna=False,
        )[inventory]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Warehouse",
        "Inventory",
    ]

    return result.sort_values(
        "Inventory",
        ascending=False,
    )


def quantity_by_supplier(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    supplier = mapping.get(
        "supplier"
    )

    quantity = mapping.get(
        "quantity"
    )

    if not supplier or not quantity:
        return pd.DataFrame()

    result = (
        data.groupby(
            supplier,
            dropna=False,
        )[quantity]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Supplier",
        "Quantity",
    ]

    return result.sort_values(
        "Quantity",
        ascending=False,
    )


def cost_by_supplier(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    supplier = mapping.get(
        "supplier"
    )

    cost = mapping.get(
        "cost"
    )

    if not supplier or not cost:
        return pd.DataFrame()

    result = (
        data.groupby(
            supplier,
            dropna=False,
        )[cost]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Supplier",
        "Cost",
    ]

    return result.sort_values(
        "Cost",
        ascending=False,
    )


def supplier_performance(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    supplier = mapping.get(
        "supplier"
    )

    if not supplier:
        return pd.DataFrame()

    result = (
        data.groupby(
            supplier,
            dropna=False,
        )
        .size()
        .reset_index(
            name="Records"
        )
    )

    quantity = mapping.get(
        "quantity"
    )

    cost = mapping.get(
        "cost"
    )

    lead_time = mapping.get(
        "lead_time"
    )

    if quantity:

        quantity_data = (
            data.groupby(
                supplier,
                dropna=False,
            )[quantity]
            .sum()
            .reset_index(
                name="Quantity"
            )
        )

        result = result.merge(
            quantity_data,
            on=supplier,
            how="left",
        )

    if cost:

        cost_data = (
            data.groupby(
                supplier,
                dropna=False,
            )[cost]
            .sum()
            .reset_index(
                name="Cost"
            )
        )

        result = result.merge(
            cost_data,
            on=supplier,
            how="left",
        )

    if lead_time:

        lead_data = (
            data.groupby(
                supplier,
                dropna=False,
            )[lead_time]
            .mean()
            .reset_index(
                name="Avg Lead Time"
            )
        )

        result = result.merge(
            lead_data,
            on=supplier,
            how="left",
        )

    if "_On_Time_Flag" in data.columns:

        on_time_data = (
            data.groupby(
                supplier,
                dropna=False,
            )["_On_Time_Flag"]
            .mean()
            .mul(100)
            .reset_index(
                name="On-Time Delivery"
            )
        )

        result = result.merge(
            on_time_data,
            on=supplier,
            how="left",
        )

    result = result.rename(
        columns={
            supplier: "Supplier"
        }
    )

    return result.sort_values(
        "Records",
        ascending=False,
    )


def delivery_performance(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    date_column = mapping.get(
        "date"
    )

    if not date_column:
        return pd.DataFrame()

    working = data.dropna(
        subset=[date_column]
    ).copy()

    if working.empty:
        return pd.DataFrame()

    working["Period"] = (
        working[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    aggregations = {}

    delivery = mapping.get(
        "delivery"
    )

    lead_time = mapping.get(
        "lead_time"
    )

    if delivery:
        aggregations[
            delivery
        ] = "mean"

    if lead_time:
        aggregations[
            lead_time
        ] = "mean"

    if "_On_Time_Flag" in working.columns:
        aggregations[
            "_On_Time_Flag"
        ] = "mean"

    if not aggregations:
        return pd.DataFrame()

    result = (
        working.groupby(
            "Period"
        )
        .agg(aggregations)
        .reset_index()
    )

    rename_map = {}

    if delivery:
        rename_map[
            delivery
        ] = "Average Delivery Time"

    if lead_time:
        rename_map[
            lead_time
        ] = "Average Lead Time"

    result = result.rename(
        columns=rename_map
    )

    if "_On_Time_Flag" in result.columns:

        result["On-Time Delivery"] = (
            result["_On_Time_Flag"] * 100
        )

        result = result.drop(
            columns=[
                "_On_Time_Flag"
            ]
        )

    return result


def inventory_risk(df):

    data, mapping = prepare_supply_chain_data(
        df
    )

    inventory = mapping.get(
        "inventory"
    )

    product = mapping.get(
        "product"
    )

    warehouse = mapping.get(
        "warehouse"
    )

    if not inventory:
        return pd.DataFrame()

    group_column = (
        product or warehouse
    )

    if not group_column:
        return pd.DataFrame()

    result = (
        data.groupby(
            group_column,
            dropna=False,
        )[inventory]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Item",
        "Inventory",
    ]

    result["Risk"] = np.select(
        [
            result["Inventory"] <= 10,
            result["Inventory"] <= 50,
        ],
        [
            "Critical",
            "Low",
        ],
        default="Healthy",
    )

    result["Risk_Score"] = np.select(
        [
            result["Inventory"] <= 10,
            result["Inventory"] <= 50,
        ],
        [
            3,
            2,
        ],
        default=1,
    )

    return result.sort_values(
        "Risk_Score",
        ascending=False,
    )


def supply_chain_summary(df):

    kpis = calculate_supply_chain_kpis(
        df
    )

    return pd.DataFrame(
        {
            "Metric": list(
                kpis.keys()
            ),
            "Value": list(
                kpis.values()
            ),
        }
    )


def generate_supply_chain_insights(df):

    insights = []

    kpis = calculate_supply_chain_kpis(
        df
    )

    if kpis["Suppliers"]:

        insights.append(
            f"The dataset contains "
            f"{kpis['Suppliers']:,} suppliers."
        )

    if kpis["Shipments"]:

        insights.append(
            f"The dataset contains "
            f"{kpis['Shipments']:,} shipments."
        )

    if kpis["On-Time Delivery"]:

        insights.append(
            f"Overall on-time delivery is "
            f"{kpis['On-Time Delivery']:.2f}%."
        )

    if kpis["Average Lead Time"]:

        insights.append(
            f"Average supplier lead time is "
            f"{kpis['Average Lead Time']:.2f} days."
        )

    if kpis["Average Delivery Time"]:

        insights.append(
            f"Average delivery time is "
            f"{kpis['Average Delivery Time']:.2f} days."
        )

    supplier_data = supplier_performance(
        df
    )

    if not supplier_data.empty:

        if "On-Time Delivery" in supplier_data.columns:

            valid = supplier_data.dropna(
                subset=[
                    "On-Time Delivery"
                ]
            )

            if not valid.empty:

                best = valid.sort_values(
                    "On-Time Delivery",
                    ascending=False,
                ).iloc[0]

                insights.append(
                    f"Highest supplier on-time "
                    f"performance: {best['Supplier']} "
                    f"({best['On-Time Delivery']:.2f}%)."
                )

    risk_data = inventory_risk(
        df
    )

    if not risk_data.empty:

        critical = int(
            (
                risk_data["Risk"]
                == "Critical"
            ).sum()
        )

        low = int(
            (
                risk_data["Risk"]
                == "Low"
            ).sum()
        )

        if critical:

            insights.append(
                f"{critical} inventory item(s) "
                f"are in the critical-risk range."
            )

        if low:

            insights.append(
                f"{low} inventory item(s) "
                f"are in the low-stock range."
            )

    return insights