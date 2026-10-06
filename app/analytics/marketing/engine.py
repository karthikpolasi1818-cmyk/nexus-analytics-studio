import pandas as pd


# =========================================================
# MARKETING COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {

    "campaign": [
        "campaign",
        "campaign_name",
        "campaign name",
        "campaign_id",
        "campaign id",
    ],

    "channel": [
        "channel",
        "marketing_channel",
        "marketing channel",
        "source",
        "medium",
    ],

    "impressions": [
        "impressions",
        "impression",
        "views",
        "ad_impressions",
        "ad impressions",
    ],

    "clicks": [
        "clicks",
        "click",
        "ad_clicks",
        "ad clicks",
    ],

    "conversions": [
        "conversions",
        "conversion",
        "leads",
        "purchases",
        "purchase_count",
        "purchase count",
    ],

    "spend": [
        "spend",
        "marketing_spend",
        "marketing spend",
        "ad_spend",
        "ad spend",
        "campaign_spend",
        "campaign spend",
        "cost",
    ],

    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "sales amount",
        "campaign_revenue",
        "campaign revenue",
    ],

    "ctr": [
        "ctr",
        "click_through_rate",
        "click through rate",
    ],

    "conversion_rate": [
        "conversion_rate",
        "conversion rate",
        "cvr",
    ],

    "roas": [
        "roas",
        "return_on_ad_spend",
        "return on ad spend",
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

    # Exact match
    for alias in aliases:

        normalized_alias = (
            normalize_column_name(alias)
        )

        if normalized_alias in normalized_columns:

            return normalized_columns[
                normalized_alias
            ]

    # Safe prefix/suffix match
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
# DETECT MARKETING COLUMNS
# =========================================================

def detect_marketing_columns(df):

    mapping = {}

    for field, aliases in COLUMN_ALIASES.items():

        mapping[field] = find_column(
            df,
            aliases
        )

    return mapping


# =========================================================
# PREPARE MARKETING DATA
# =========================================================

def prepare_marketing_data(df):

    df = df.copy()

    mapping = detect_marketing_columns(
        df
    )

    numeric_fields = [
        "impressions",
        "clicks",
        "conversions",
        "spend",
        "revenue",
        "ctr",
        "conversion_rate",
        "roas",
    ]

    for field in numeric_fields:

        column = mapping.get(field)

        if column:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # -----------------------------------------------------
    # CALCULATED CTR
    # -----------------------------------------------------

    if (
        not mapping["ctr"]
        and mapping["clicks"]
        and mapping["impressions"]
    ):

        calculated_column = (
            "Calculated CTR"
        )

        df[calculated_column] = (
            df[mapping["clicks"]]
            / df[mapping["impressions"]]
            .replace(0, pd.NA)
            * 100
        )

        mapping["ctr"] = calculated_column

    # -----------------------------------------------------
    # CALCULATED CONVERSION RATE
    # -----------------------------------------------------

    if (
        not mapping["conversion_rate"]
        and mapping["conversions"]
        and mapping["clicks"]
    ):

        calculated_column = (
            "Calculated Conversion Rate"
        )

        df[calculated_column] = (
            df[mapping["conversions"]]
            / df[mapping["clicks"]]
            .replace(0, pd.NA)
            * 100
        )

        mapping["conversion_rate"] = (
            calculated_column
        )

    # -----------------------------------------------------
    # CALCULATED ROAS
    # -----------------------------------------------------

    if (
        not mapping["roas"]
        and mapping["revenue"]
        and mapping["spend"]
    ):

        calculated_column = (
            "Calculated ROAS"
        )

        df[calculated_column] = (
            df[mapping["revenue"]]
            / df[mapping["spend"]]
            .replace(0, pd.NA)
        )

        mapping["roas"] = calculated_column

    return df, mapping


# =========================================================
# MARKETING KPIs
# =========================================================

def calculate_marketing_kpis(
    df,
    mapping
):

    impressions = (
        float(
            df[mapping["impressions"]]
            .sum()
        )
        if mapping["impressions"]
        else 0
    )

    clicks = (
        float(
            df[mapping["clicks"]]
            .sum()
        )
        if mapping["clicks"]
        else 0
    )

    conversions = (
        float(
            df[mapping["conversions"]]
            .sum()
        )
        if mapping["conversions"]
        else 0
    )

    spend = (
        float(
            df[mapping["spend"]]
            .sum()
        )
        if mapping["spend"]
        else 0
    )

    revenue = (
        float(
            df[mapping["revenue"]]
            .sum()
        )
        if mapping["revenue"]
        else 0
    )

    # -----------------------------------------------------
    # CTR
    # -----------------------------------------------------

    ctr = (
        clicks / impressions * 100
        if impressions
        else 0
    )

    # -----------------------------------------------------
    # CONVERSION RATE
    # -----------------------------------------------------

    conversion_rate = (
        conversions / clicks * 100
        if clicks
        else 0
    )

    # -----------------------------------------------------
    # ROAS
    # -----------------------------------------------------

    roas = (
        revenue / spend
        if spend
        else 0
    )

    # -----------------------------------------------------
    # CAC
    # -----------------------------------------------------

    cac = (
        spend / conversions
        if conversions
        else 0
    )

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    roi = (
        (revenue - spend)
        / spend
        * 100
        if spend
        else 0
    )

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": spend,
        "revenue": revenue,
        "ctr": ctr,
        "conversion_rate": conversion_rate,
        "roas": roas,
        "cac": cac,
        "roi": roi,
    }


# =========================================================
# REVENUE BY CHANNEL
# =========================================================

def revenue_by_channel(
    df,
    mapping
):

    if (
        not mapping["channel"]
        or not mapping["revenue"]
    ):

        return None

    return (
        df.groupby(
            mapping["channel"],
            dropna=False
        )[mapping["revenue"]]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


# =========================================================
# SPEND BY CHANNEL
# =========================================================

def spend_by_channel(
    df,
    mapping
):

    if (
        not mapping["channel"]
        or not mapping["spend"]
    ):

        return None

    return (
        df.groupby(
            mapping["channel"],
            dropna=False
        )[mapping["spend"]]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


# =========================================================
# ROAS BY CHANNEL
# =========================================================

def roas_by_channel(
    df,
    mapping
):

    if (
        not mapping["channel"]
        or not mapping["revenue"]
        or not mapping["spend"]
    ):

        return None

    result = (
        df.groupby(
            mapping["channel"],
            dropna=False
        )
        .agg(
            Revenue=(
                mapping["revenue"],
                "sum"
            ),
            Spend=(
                mapping["spend"],
                "sum"
            ),
        )
        .reset_index()
    )

    result["ROAS"] = (
        result["Revenue"]
        / result["Spend"].replace(
            0,
            pd.NA
        )
    )

    result["ROI"] = (
        (
            result["Revenue"]
            - result["Spend"]
        )
        / result["Spend"].replace(
            0,
            pd.NA
        )
        * 100
    )

    return result.sort_values(
        "ROAS",
        ascending=False
    )


# =========================================================
# CONVERSIONS BY CHANNEL
# =========================================================

def conversions_by_channel(
    df,
    mapping
):

    if (
        not mapping["channel"]
        or not mapping["conversions"]
    ):

        return None

    return (
        df.groupby(
            mapping["channel"],
            dropna=False
        )[mapping["conversions"]]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


# =========================================================
# CAMPAIGN PERFORMANCE
# =========================================================

def campaign_performance(
    df,
    mapping
):

    campaign_column = mapping["campaign"]

    if not campaign_column:

        return None

    aggregation = {}

    if mapping["impressions"]:

        aggregation["Impressions"] = (
            mapping["impressions"],
            "sum"
        )

    if mapping["clicks"]:

        aggregation["Clicks"] = (
            mapping["clicks"],
            "sum"
        )

    if mapping["conversions"]:

        aggregation["Conversions"] = (
            mapping["conversions"],
            "sum"
        )

    if mapping["spend"]:

        aggregation["Spend"] = (
            mapping["spend"],
            "sum"
        )

    if mapping["revenue"]:

        aggregation["Revenue"] = (
            mapping["revenue"],
            "sum"
        )

    if not aggregation:

        return None

    result = (
        df.groupby(
            campaign_column,
            dropna=False
        )
        .agg(**aggregation)
        .reset_index()
    )

    # -----------------------------------------------------
    # CTR
    # -----------------------------------------------------

    if (
        "Clicks" in result.columns
        and "Impressions"
        in result.columns
    ):

        result["CTR"] = (
            result["Clicks"]
            / result["Impressions"]
            .replace(0, pd.NA)
            * 100
        )

    # -----------------------------------------------------
    # CONVERSION RATE
    # -----------------------------------------------------

    if (
        "Conversions"
        in result.columns
        and "Clicks"
        in result.columns
    ):

        result["Conversion Rate"] = (
            result["Conversions"]
            / result["Clicks"]
            .replace(0, pd.NA)
            * 100
        )

    # -----------------------------------------------------
    # ROAS
    # -----------------------------------------------------

    if (
        "Revenue" in result.columns
        and "Spend" in result.columns
    ):

        result["ROAS"] = (
            result["Revenue"]
            / result["Spend"]
            .replace(0, pd.NA)
        )

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    if (
        "Revenue" in result.columns
        and "Spend" in result.columns
    ):

        result["ROI"] = (
            (
                result["Revenue"]
                - result["Spend"]
            )
            / result["Spend"]
            .replace(0, pd.NA)
            * 100
        )

    # -----------------------------------------------------
    # CAC
    # -----------------------------------------------------

    if (
        "Spend" in result.columns
        and "Conversions"
        in result.columns
    ):

        result["CAC"] = (
            result["Spend"]
            / result["Conversions"]
            .replace(0, pd.NA)
        )

    return result


# =========================================================
# MARKETING FUNNEL
# =========================================================

def marketing_funnel(
    df,
    mapping
):

    stages = []
    values = []

    if mapping["impressions"]:

        stages.append("Impressions")

        values.append(
            float(
                df[
                    mapping["impressions"]
                ].sum()
            )
        )

    if mapping["clicks"]:

        stages.append("Clicks")

        values.append(
            float(
                df[
                    mapping["clicks"]
                ].sum()
            )
        )

    if mapping["conversions"]:

        stages.append("Conversions")

        values.append(
            float(
                df[
                    mapping["conversions"]
                ].sum()
            )
        )

    if not stages:

        return None

    return pd.DataFrame(
        {
            "Stage": stages,
            "Count": values,
        }
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

def generate_marketing_insights(
    df,
    mapping
):

    insights = []

    kpis = calculate_marketing_kpis(
        df,
        mapping
    )

    # -----------------------------------------------------
    # REVENUE
    # -----------------------------------------------------

    if kpis["revenue"] > 0:

        insights.append(
            f"Marketing generated "
            f"₹{kpis['revenue']:,.0f} "
            f"in attributed revenue from "
            f"₹{kpis['spend']:,.0f} spend."
        )

    # -----------------------------------------------------
    # ROAS
    # -----------------------------------------------------

    if kpis["roas"] > 0:

        insights.append(
            f"Overall ROAS is "
            f"{kpis['roas']:.2f}x."
        )

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    if kpis["roi"] != 0:

        insights.append(
            f"Overall marketing ROI is "
            f"{kpis['roi']:.2f}%."
        )

    # -----------------------------------------------------
    # CTR
    # -----------------------------------------------------

    if kpis["ctr"] > 0:

        insights.append(
            f"Overall click-through rate "
            f"is {kpis['ctr']:.2f}%."
        )

    # -----------------------------------------------------
    # CONVERSION RATE
    # -----------------------------------------------------

    if kpis["conversion_rate"] > 0:

        insights.append(
            f"Overall conversion rate "
            f"is {kpis['conversion_rate']:.2f}%."
        )

    # -----------------------------------------------------
    # CAC
    # -----------------------------------------------------

    if kpis["cac"] > 0:

        insights.append(
            f"Average acquisition cost "
            f"is approximately "
            f"₹{kpis['cac']:,.2f} per conversion."
        )

    # -----------------------------------------------------
    # BEST CHANNEL
    # -----------------------------------------------------

    channel_data = roas_by_channel(
        df,
        mapping
    )

    if (
        channel_data is not None
        and not channel_data.empty
    ):

        best = channel_data.iloc[0]

        channel_column = (
            mapping["channel"]
        )

        insights.append(
            f"The highest-ROAS channel "
            f"in the dataset is "
            f"{best[channel_column]} "
            f"at {best['ROAS']:.2f}x."
        )

    # -----------------------------------------------------
    # BEST CAMPAIGN
    # -----------------------------------------------------

    campaign_data = campaign_performance(
        df,
        mapping
    )

    if (
        campaign_data is not None
        and not campaign_data.empty
        and "ROAS" in campaign_data.columns
    ):

        valid_campaigns = (
            campaign_data[
                campaign_data["ROAS"].notna()
            ]
        )

        if not valid_campaigns.empty:

            best_campaign = (
                valid_campaigns
                .sort_values(
                    "ROAS",
                    ascending=False
                )
                .iloc[0]
            )

            insights.append(
                f"The highest-ROAS campaign "
                f"is "
                f"{best_campaign[mapping['campaign']]} "
                f"at "
                f"{best_campaign['ROAS']:.2f}x."
            )

    return insights