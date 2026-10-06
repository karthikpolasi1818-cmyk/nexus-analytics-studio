import pandas as pd
import numpy as np


COLUMN_ALIASES = {
    "revenue": [
        "revenue",
        "sales_revenue",
        "total_revenue",
        "income",
        "sales",
    ],
    "expense": [
        "expense",
        "expenses",
        "total_expense",
        "operating_expense",
        "opex",
    ],
    "cost": [
        "cost",
        "total_cost",
        "cost_of_goods_sold",
        "cogs",
        "cost_of_goods",
    ],
    "profit": [
        "profit",
        "net_profit",
        "gross_profit",
        "operating_profit",
    ],
    "budget": [
        "budget",
        "planned_budget",
        "allocated_budget",
        "budget_amount",
    ],
    "cash_flow": [
        "cash_flow",
        "cashflow",
        "net_cash_flow",
        "cash",
    ],
    "date": [
        "date",
        "transaction_date",
        "invoice_date",
        "financial_date",
        "month",
        "period",
        "year",
    ],
    "category": [
        "category",
        "expense_category",
        "cost_category",
        "financial_category",
        "account_category",
    ],
    "department": [
        "department",
        "business_unit",
        "division",
        "team",
    ],
    "account": [
        "account",
        "account_name",
        "account_type",
        "ledger_account",
    ],
    "transaction_id": [
        "transaction_id",
        "transactionid",
        "transaction",
        "invoice_id",
        "invoice_number",
    ],
}


def _normalise(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, aliases):
    """
    Safely find a matching dataframe column.

    Priority:
    1. Exact normalized match
    2. Prefix/suffix match

    Avoids arbitrary substring matching that can cause
    incorrect financial field detection.
    """
    normalized_columns = {
        _normalise(column): column
        for column in df.columns
    }

    normalized_aliases = [_normalise(alias) for alias in aliases]

    # Exact match
    for alias in normalized_aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    # Prefix / suffix match
    for normalized_column, original_column in normalized_columns.items():
        for alias in normalized_aliases:
            if (
                normalized_column.startswith(alias + "_")
                or normalized_column.endswith("_" + alias)
            ):
                return original_column

    return None


def detect_financial_columns(df):
    mapping = {}

    for field, aliases in COLUMN_ALIASES.items():
        mapping[field] = find_column(df, aliases)

    return mapping


def prepare_financial_data(df):
    result = df.copy()
    mapping = detect_financial_columns(result)

    numeric_fields = [
        "revenue",
        "expense",
        "cost",
        "profit",
        "budget",
        "cash_flow",
    ]

    for field in numeric_fields:
        column = mapping.get(field)

        if column is not None:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    date_column = mapping.get("date")

    if date_column is not None:
        result[date_column] = pd.to_datetime(
            result[date_column],
            errors="coerce",
        )

    # Calculate profit if it does not exist
    if mapping.get("profit") is None:
        revenue_column = mapping.get("revenue")
        expense_column = mapping.get("expense")
        cost_column = mapping.get("cost")

        if revenue_column is not None:
            if expense_column is not None:
                result["Calculated_Profit"] = (
                    result[revenue_column]
                    - result[expense_column].fillna(0)
                )

                mapping["profit"] = "Calculated_Profit"

            elif cost_column is not None:
                result["Calculated_Profit"] = (
                    result[revenue_column]
                    - result[cost_column].fillna(0)
                )

                mapping["profit"] = "Calculated_Profit"

    return result, mapping


def calculate_financial_kpis(df):
    df, mapping = prepare_financial_data(df)

    revenue = 0.0
    expense = 0.0
    cost = 0.0
    profit = 0.0
    budget = 0.0
    cash_flow = 0.0

    if mapping.get("revenue"):
        revenue = float(
            df[mapping["revenue"]].fillna(0).sum()
        )

    if mapping.get("expense"):
        expense = float(
            df[mapping["expense"]].fillna(0).sum()
        )

    if mapping.get("cost"):
        cost = float(
            df[mapping["cost"]].fillna(0).sum()
        )

    if mapping.get("profit"):
        profit = float(
            df[mapping["profit"]].fillna(0).sum()
        )
    elif revenue:
        profit = revenue - expense - cost

    if mapping.get("budget"):
        budget = float(
            df[mapping["budget"]].fillna(0).sum()
        )

    if mapping.get("cash_flow"):
        cash_flow = float(
            df[mapping["cash_flow"]].fillna(0).sum()
        )
    else:
        cash_flow = profit

    profit_margin = (
        profit / revenue * 100
        if revenue != 0
        else 0
    )

    expense_ratio = (
        expense / revenue * 100
        if revenue != 0
        else 0
    )

    budget_variance = (
        revenue - budget
        if budget != 0
        else 0
    )

    budget_utilization = (
        revenue / budget * 100
        if budget != 0
        else 0
    )

    return {
        "Revenue": revenue,
        "Expenses": expense,
        "Cost": cost,
        "Profit": profit,
        "Profit Margin": profit_margin,
        "Budget": budget,
        "Budget Variance": budget_variance,
        "Budget Utilization": budget_utilization,
        "Cash Flow": cash_flow,
        "Expense Ratio": expense_ratio,
    }


def revenue_by_category(df):
    df, mapping = prepare_financial_data(df)

    category = mapping.get("category")
    revenue = mapping.get("revenue")

    if not category or not revenue:
        return pd.DataFrame()

    result = (
        df.groupby(category, dropna=False)[revenue]
        .sum()
        .reset_index()
    )

    result.columns = ["Category", "Revenue"]

    return result.sort_values(
        "Revenue",
        ascending=False,
    )


def expenses_by_category(df):
    df, mapping = prepare_financial_data(df)

    category = mapping.get("category")
    expense = mapping.get("expense")

    if not category or not expense:
        return pd.DataFrame()

    result = (
        df.groupby(category, dropna=False)[expense]
        .sum()
        .reset_index()
    )

    result.columns = ["Category", "Expenses"]

    return result.sort_values(
        "Expenses",
        ascending=False,
    )


def profit_by_category(df):
    df, mapping = prepare_financial_data(df)

    category = mapping.get("category")
    profit = mapping.get("profit")

    if not category or not profit:
        return pd.DataFrame()

    result = (
        df.groupby(category, dropna=False)[profit]
        .sum()
        .reset_index()
    )

    result.columns = ["Category", "Profit"]

    return result.sort_values(
        "Profit",
        ascending=False,
    )


def revenue_by_department(df):
    df, mapping = prepare_financial_data(df)

    department = mapping.get("department")
    revenue = mapping.get("revenue")

    if not department or not revenue:
        return pd.DataFrame()

    result = (
        df.groupby(department, dropna=False)[revenue]
        .sum()
        .reset_index()
    )

    result.columns = ["Department", "Revenue"]

    return result.sort_values(
        "Revenue",
        ascending=False,
    )


def expenses_by_department(df):
    df, mapping = prepare_financial_data(df)

    department = mapping.get("department")
    expense = mapping.get("expense")

    if not department or not expense:
        return pd.DataFrame()

    result = (
        df.groupby(department, dropna=False)[expense]
        .sum()
        .reset_index()
    )

    result.columns = ["Department", "Expenses"]

    return result.sort_values(
        "Expenses",
        ascending=False,
    )


def financial_trend(df):
    df, mapping = prepare_financial_data(df)

    date_column = mapping.get("date")
    revenue_column = mapping.get("revenue")
    expense_column = mapping.get("expense")
    profit_column = mapping.get("profit")

    if not date_column:
        return pd.DataFrame()

    if not any([
        revenue_column,
        expense_column,
        profit_column,
    ]):
        return pd.DataFrame()

    working = df.copy()

    working["_Financial_Date"] = pd.to_datetime(
        working[date_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=["_Financial_Date"]
    )

    working["_Period"] = (
        working["_Financial_Date"]
        .dt.to_period("M")
        .astype(str)
    )

    aggregations = {}

    if revenue_column:
        aggregations[revenue_column] = "sum"

    if expense_column:
        aggregations[expense_column] = "sum"

    if profit_column:
        aggregations[profit_column] = "sum"

    result = (
        working.groupby("_Period")
        .agg(aggregations)
        .reset_index()
    )

    rename_map = {
        "_Period": "Period",
    }

    if revenue_column:
        rename_map[revenue_column] = "Revenue"

    if expense_column:
        rename_map[expense_column] = "Expenses"

    if profit_column:
        rename_map[profit_column] = "Profit"

    result = result.rename(
        columns=rename_map
    )

    return result


def budget_analysis(df):
    df, mapping = prepare_financial_data(df)

    budget = mapping.get("budget")
    revenue = mapping.get("revenue")

    if not budget:
        return pd.DataFrame()

    working = df.copy()

    if mapping.get("date"):
        working["_Date"] = pd.to_datetime(
            working[mapping["date"]],
            errors="coerce",
        )
        working = working.dropna(
            subset=["_Date"]
        )
        working["_Period"] = (
            working["_Date"]
            .dt.to_period("M")
            .astype(str)
        )

        group_column = "_Period"
    elif mapping.get("category"):
        group_column = mapping["category"]
    elif mapping.get("department"):
        group_column = mapping["department"]
    else:
        return pd.DataFrame()

    aggregation = {
        budget: "sum",
    }

    if revenue:
        aggregation[revenue] = "sum"

    result = (
        working.groupby(group_column)
        .agg(aggregation)
        .reset_index()
    )

    if group_column == "_Period":
        result = result.rename(
            columns={"_Period": "Period"}
        )
    elif group_column == mapping.get("category"):
        result = result.rename(
            columns={group_column: "Category"}
        )
    elif group_column == mapping.get("department"):
        result = result.rename(
            columns={group_column: "Department"}
        )

    result = result.rename(
        columns={budget: "Budget"}
    )

    if revenue:
        result = result.rename(
            columns={revenue: "Actual"}
        )

        result["Variance"] = (
            result["Actual"] - result["Budget"]
        )

        result["Utilization"] = np.where(
            result["Budget"] != 0,
            result["Actual"]
            / result["Budget"]
            * 100,
            0,
        )

    return result


def financial_summary(df):
    kpis = calculate_financial_kpis(df)

    return pd.DataFrame(
        {
            "Metric": list(kpis.keys()),
            "Value": list(kpis.values()),
        }
    )


def generate_financial_insights(df):
    insights = []

    kpis = calculate_financial_kpis(df)

    revenue = kpis["Revenue"]
    expenses = kpis["Expenses"]
    profit = kpis["Profit"]
    margin = kpis["Profit Margin"]
    cash_flow = kpis["Cash Flow"]

    if revenue > 0:
        insights.append(
            f"Total revenue is {revenue:,.2f}."
        )

    if profit > 0:
        insights.append(
            f"Profit is {profit:,.2f} with a "
            f"profit margin of {margin:.2f}%."
        )
    elif profit < 0:
        insights.append(
            f"The dataset shows a loss of "
            f"{abs(profit):,.2f}."
        )

    if expenses > 0 and revenue > 0:
        expense_ratio = (
            expenses / revenue * 100
        )

        insights.append(
            f"Expenses represent "
            f"{expense_ratio:.2f}% of revenue."
        )

    if cash_flow != 0:
        if cash_flow > 0:
            insights.append(
                f"Net cash flow is positive at "
                f"{cash_flow:,.2f}."
            )
        else:
            insights.append(
                f"Net cash flow is negative at "
                f"{abs(cash_flow):,.2f}."
            )

    category_data = revenue_by_category(df)

    if not category_data.empty:
        best = category_data.iloc[0]

        insights.append(
            f"The highest-revenue category is "
            f"{best['Category']} with "
            f"{best['Revenue']:,.2f}."
        )

    expense_data = expenses_by_category(df)

    if not expense_data.empty:
        highest_expense = expense_data.iloc[0]

        insights.append(
            f"The largest expense category is "
            f"{highest_expense['Category']} with "
            f"{highest_expense['Expenses']:,.2f}."
        )

    return insights