import re


MODULE_KEYWORDS = {

    "Sales Analytics": [
        "sales",
        "revenue",
        "orders",
        "order",
        "quantity",
        "qty",
        "salesperson",
        "region",
        "product",
        "profit",
    ],

    "Customer Analytics": [
        "customer",
        "customer_id",
        "customer_name",
        "retention",
        "churn",
        "lifetime",
        "purchase",
    ],

    "Marketing Analytics": [
        "campaign",
        "click",
        "impression",
        "conversion",
        "ctr",
        "roas",
        "marketing",
    ],

    "Financial Analytics": [
        "revenue",
        "expense",
        "profit",
        "cost",
        "margin",
        "budget",
        "cash",
    ],

    "Supply Chain Analytics": [
        "supplier",
        "inventory",
        "stock",
        "shipment",
        "delivery",
        "lead_time",
        "warehouse",
    ],

    "Product Analytics": [
        "product",
        "feature",
        "session",
        "active_user",
        "engagement",
        "usage",
    ],

    "Operations Analytics": [
        "operation",
        "processing_time",
        "cycle_time",
        "efficiency",
        "throughput",
        "sla",
    ],

    "HR Analytics": [
        "employee",
        "employee_id",
        "salary",
        "department",
        "attrition",
        "hire",
        "tenure",
    ],

    "Fraud Analytics": [
        "transaction",
        "transaction_id",
        "fraud",
        "risk",
        "suspicious",
        "merchant",
    ],

    "Healthcare Analytics": [
        "patient",
        "patient_id",
        "diagnosis",
        "treatment",
        "hospital",
        "admission",
        "discharge",
    ],

    "Manufacturing Analytics": [
        "production",
        "machine",
        "defect",
        "downtime",
        "quality",
        "maintenance",
        "oee",
    ],

    "E-commerce Analytics": [
        "cart",
        "checkout",
        "order",
        "product",
        "customer",
        "conversion",
        "payment",
    ],
}


def normalize_column(column):

    return re.sub(
        r"[^a-z0-9_]",
        "_",
        str(column).lower()
    )


def detect_modules(columns):

    normalized_columns = [
        normalize_column(column)
        for column in columns
    ]

    scores = {}

    for module, keywords in (
        MODULE_KEYWORDS.items()
    ):

        score = 0

        for column in normalized_columns:

            for keyword in keywords:

                normalized_keyword = (
                    normalize_column(keyword)
                )

                if (
                    normalized_keyword
                    in column
                ):
                    score += 1

        scores[module] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked