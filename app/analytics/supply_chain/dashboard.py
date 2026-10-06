import streamlit as st
import plotly.express as px

from .engine import (
    prepare_supply_chain_data,
    calculate_supply_chain_kpis,
    inventory_by_product,
    inventory_by_warehouse,
    quantity_by_supplier,
    cost_by_supplier,
    supplier_performance,
    delivery_performance,
    inventory_risk,
    supply_chain_summary,
    generate_supply_chain_insights,
)


def number(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _chart_key(prefix, df):
    return f"supply_chain_{prefix}_{id(df)}"


def _show_chart(fig, key):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


def render_supply_chain_dashboard(df):

    st.markdown(
        "## 🚚 Supply Chain Analytics"
    )

    st.caption(
        "Analyze suppliers, inventory, shipments, "
        "delivery performance, lead times, "
        "logistics costs and inventory risk."
    )

    prepared_df, mapping = (
        prepare_supply_chain_data(df)
    )

    # ==================================================
    # DETECTED FIELDS
    # ==================================================

    st.markdown(
        "### 🔍 Detected Supply Chain Fields"
    )

    detected = {
        key.replace("_", " ").title(): value
        for key, value in mapping.items()
        if value is not None
    }

    if not detected:

        st.warning(
            "No recognizable supply chain "
            "fields were detected."
        )

        return

    st.json(detected)

    # ==================================================
    # KPIs
    # ==================================================

    kpis = calculate_supply_chain_kpis(
        prepared_df
    )

    st.markdown(
        "### 📊 Supply Chain KPIs"
    )

    row1 = st.columns(4)

    row1[0].metric(
        "Orders",
        f"{kpis['Orders']:,}",
    )

    row1[1].metric(
        "Suppliers",
        f"{kpis['Suppliers']:,}",
    )

    row1[2].metric(
        "Shipments",
        f"{kpis['Shipments']:,}",
    )

    row1[3].metric(
        "Inventory",
        number(
            kpis["Inventory"]
        ),
    )

    row2 = st.columns(4)

    row2[0].metric(
        "Quantity",
        number(
            kpis["Quantity"]
        ),
    )

    row2[1].metric(
        "Avg Delivery Time",
        f"{kpis['Average Delivery Time']:.2f} days",
    )

    row2[2].metric(
        "Avg Lead Time",
        f"{kpis['Average Lead Time']:.2f} days",
    )

    row2[3].metric(
        "On-Time Delivery",
        f"{kpis['On-Time Delivery']:.2f}%",
    )

    # ==================================================
    # COST
    # ==================================================

    if kpis["Supply Chain Cost"] != 0:

        st.markdown(
            "### 💰 Supply Chain Cost"
        )

        st.metric(
            "Total Supply Chain Cost",
            f"₹{kpis['Supply Chain Cost']:,.2f}",
        )

    # ==================================================
    # DELIVERY PERFORMANCE
    # ==================================================

    delivery_data = delivery_performance(
        prepared_df
    )

    if not delivery_data.empty:

        st.markdown(
            "### 📦 Delivery Performance"
        )

        metrics = [
            column
            for column in [
                "Average Delivery Time",
                "Average Lead Time",
            ]
            if column in delivery_data.columns
        ]

        if metrics:

            fig = px.line(
                delivery_data,
                x="Period",
                y=metrics,
                markers=True,
                title=(
                    "Delivery and Lead Time Trend"
                ),
            )

            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Days",
            )

            _show_chart(
                fig,
                _chart_key(
                    "delivery_trend",
                    prepared_df,
                ),
            )

        if (
            "On-Time Delivery"
            in delivery_data.columns
        ):

            fig = px.line(
                delivery_data,
                x="Period",
                y="On-Time Delivery",
                markers=True,
                title=(
                    "On-Time Delivery Trend"
                ),
            )

            fig.update_layout(
                xaxis_title="Period",
                yaxis_title=(
                    "On-Time Delivery (%)"
                ),
            )

            _show_chart(
                fig,
                _chart_key(
                    "on_time_trend",
                    prepared_df,
                ),
            )

    # ==================================================
    # INVENTORY
    # ==================================================

    product_inventory = (
        inventory_by_product(
            prepared_df
        )
    )

    if not product_inventory.empty:

        st.markdown(
            "### 📦 Inventory by Product"
        )

        fig = px.bar(
            product_inventory.head(20),
            x="Product",
            y="Inventory",
            title=(
                "Top Products by Inventory"
            ),
        )

        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Inventory",
        )

        _show_chart(
            fig,
            _chart_key(
                "inventory_product",
                prepared_df,
            ),
        )

    warehouse_inventory = (
        inventory_by_warehouse(
            prepared_df
        )
    )

    if not warehouse_inventory.empty:

        st.markdown(
            "### 🏭 Inventory by Warehouse"
        )

        fig = px.bar(
            warehouse_inventory,
            x="Warehouse",
            y="Inventory",
            title=(
                "Inventory by Warehouse"
            ),
        )

        fig.update_layout(
            xaxis_title="Warehouse",
            yaxis_title="Inventory",
        )

        _show_chart(
            fig,
            _chart_key(
                "inventory_warehouse",
                prepared_df,
            ),
        )

    # ==================================================
    # SUPPLIER ANALYSIS
    # ==================================================

    supplier_quantity = (
        quantity_by_supplier(
            prepared_df
        )
    )

    if not supplier_quantity.empty:

        st.markdown(
            "### 🤝 Quantity by Supplier"
        )

        fig = px.bar(
            supplier_quantity.head(20),
            x="Supplier",
            y="Quantity",
            title="Supplier Quantity",
        )

        _show_chart(
            fig,
            _chart_key(
                "supplier_quantity",
                prepared_df,
            ),
        )

    supplier_cost = (
        cost_by_supplier(
            prepared_df
        )
    )

    if not supplier_cost.empty:

        st.markdown(
            "### 💰 Cost by Supplier"
        )

        fig = px.bar(
            supplier_cost.head(20),
            x="Supplier",
            y="Cost",
            title="Supplier Cost",
        )

        _show_chart(
            fig,
            _chart_key(
                "supplier_cost",
                prepared_df,
            ),
        )

    supplier_data = (
        supplier_performance(
            prepared_df
        )
    )

    if not supplier_data.empty:

        st.markdown(
            "### 🏆 Supplier Performance"
        )

        st.dataframe(
            supplier_data,
            use_container_width=True,
        )

    # ==================================================
    # INVENTORY RISK
    # ==================================================

    risk_data = inventory_risk(
        prepared_df
    )

    if not risk_data.empty:

        st.markdown(
            "### ⚠️ Inventory Risk"
        )

        st.dataframe(
            risk_data,
            use_container_width=True,
        )

        fig = px.bar(
            risk_data.head(30),
            x="Item",
            y="Inventory",
            color="Risk",
            title=(
                "Inventory Risk Assessment"
            ),
        )

        _show_chart(
            fig,
            _chart_key(
                "inventory_risk",
                prepared_df,
            ),
        )

    # ==================================================
    # SUMMARY
    # ==================================================

    st.markdown(
        "### 📋 Supply Chain Summary"
    )

    summary = supply_chain_summary(
        prepared_df
    )

    st.dataframe(
        summary,
        use_container_width=True,
    )

    # ==================================================
    # BUSINESS INSIGHTS
    # ==================================================

    st.markdown(
        "### 🧠 Supply Chain Business Insights"
    )

    insights = (
        generate_supply_chain_insights(
            prepared_df
        )
    )

    if insights:

        for insight in insights:
            st.info(insight)

    else:

        st.info(
            "Not enough supply chain information "
            "was available to generate insights."
        )

    # ==================================================
    # RAW DATA
    # ==================================================

    with st.expander(
        "📄 View Supply Chain Dataset"
    ):

        st.dataframe(
            prepared_df,
            use_container_width=True,
        )