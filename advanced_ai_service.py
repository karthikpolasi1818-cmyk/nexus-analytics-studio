"""
NEXUS Phase 6 — Advanced AI Analyst v2
Evidence-first deterministic analyst layer.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional

def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def _money(x: Any) -> str:
    n = _num(x)
    if n is None:
        return "N/A"
    if abs(n) >= 10_000_000:
        return f"₹{n/10_000_000:.2f} Cr"
    if abs(n) >= 100_000:
        return f"₹{n/100_000:.2f} L"
    return f"₹{n:,.2f}"

def _pct(x: Any) -> str:
    n = _num(x)
    return "N/A" if n is None else f"{n:.2f}%"

def _fmt(x: Any) -> str:
    n = _num(x)
    if n is None:
        return str(x)
    return f"{int(n):,}" if n.is_integer() else f"{n:,.2f}"

def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _question_type(q: str) -> str:
    q = _norm(q)
    if any(x in q for x in ["missing", "null", "quality", "duplicate", "duplicates"]):
        return "quality"
    if any(x in q for x in ["trend", "over time", "monthly", "weekly", "daily", "growth"]):
        return "trend"
    if any(x in q for x in ["highest", "top", "largest", "best", "maximum", "max"]):
        return "top"
    if any(x in q for x in ["lowest", "bottom", "smallest", "minimum", "min"]):
        return "bottom"
    if any(x in q for x in ["compare", "comparison", "versus", "difference"]):
        return "compare"
    if any(x in q for x in ["correlation", "correlated"]):
        return "correlation"
    if any(x in q for x in ["anomal", "outlier", "unusual"]):
        return "anomaly"
    return "kpi"

def _find_kpi(kpis: Dict[str, Any], q: str):
    qn = _norm(q)
    aliases = {
        "revenue": ["revenue", "sales", "turnover"],
        "profit": ["profit", "earnings"],
        "orders": ["orders", "order count", "transactions"],
        "customers": ["customers", "customer count", "clients"],
        "quantity": ["quantity", "units", "units sold"],
        "profit_margin": ["profit margin", "margin"],
        "average_order_value": ["average order value", "aov", "average order"],
        "employees": ["employees", "headcount", "workforce"],
        "attrition_rate": ["attrition", "attrition rate", "turnover rate"],
        "average_salary": ["average salary", "salary"],
        "defects": ["defects"],
        "defect_rate": ["defect rate"],
        "quality_rate": ["quality rate"],
        "downtime": ["downtime"],
        "maintenance_cost": ["maintenance cost"],
        "production_cost": ["production cost"],
        "total_charges": ["total charges", "charges"],
        "readmission_rate": ["readmission", "readmission rate"],
        "mortality_rate": ["mortality", "mortality rate"],
        "spend": ["spend", "marketing spend", "ad spend"],
        "roas": ["roas", "return on ad spend"],
        "ctr": ["ctr", "click through rate"],
        "conversion_rate": ["conversion rate", "conversion"],
    }
    for key, words in aliases.items():
        if any(w in qn for w in words):
            if key in kpis:
                return key, kpis[key]
            for actual, value in kpis.items():
                if _norm(actual) == _norm(key):
                    return actual, value
    for actual, value in kpis.items():
        if _norm(actual) in qn:
            return actual, value
    return None, None

def _series_change(rows: List[Dict[str, Any]]):
    if len(rows) < 2:
        return None
    value_key = None
    for candidate in ["revenue", "sales", "profit", "value", "amount"]:
        if candidate in rows[-1]:
            value_key = candidate
            break
    if not value_key:
        for k, v in rows[-1].items():
            if _num(v) is not None:
                value_key = k
                break
    if not value_key:
        return None
    a, b = _num(rows[-2].get(value_key)), _num(rows[-1].get(value_key))
    if a is None or b is None:
        return None
    change = b - a
    return {"key": value_key, "from": a, "to": b, "change": change,
            "pct": (change / a * 100) if a else None}

def _ranking(dashboard: Dict[str, Any], q: str):
    qn = _norm(q)
    for key, label in [
        ("category_performance", "category"),
        ("region_performance", "region"),
        ("product_performance", "product"),
        ("channel_performance", "channel"),
    ]:
        rows = dashboard.get(key) or []
        if not rows:
            continue
        metric = "profit" if "profit" in qn and "profit" in rows[0] else None
        if not metric:
            metric = "sales" if "sales" in rows[0] else ("revenue" if "revenue" in rows[0] else None)
        if metric:
            usable = []
            for r in rows:
                name = r.get(label, r.get(label.title(), r.get("name")))
                val = _num(r.get(metric))
                if name is not None and val is not None:
                    usable.append((name, val))
            if usable:
                reverse = not any(x in qn for x in ["lowest", "bottom", "smallest", "minimum", "min"])
                item = sorted(usable, key=lambda x: x[1], reverse=reverse)[0]
                direction = "highest" if reverse else "lowest"
                return f"The {direction} {label} by {metric} is **{item[0]}** with **{_fmt(item[1])}**."
    return None

def analyze_question(question: str, dataset_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = dataset_result or {}
    dashboard = result.get("dashboard") or {}
    kpis = dashboard.get("kpis") or {}
    quality = dashboard.get("quality") or {}
    qtype = _question_type(question)
    answer = ""
    evidence: List[Dict[str, Any]] = []
    chart = None

    if qtype == "quality":
        parts = []
        for key in ["quality_score", "score", "grade", "duplicate_rows", "missing_values",
                    "duplicate_ids", "numeric_outliers", "invalid_dates"]:
            if key in quality:
                parts.append(f"{key.replace('_', ' ').title()}: {quality[key]}")
                evidence.append({"source": "dashboard.quality", "field": key, "value": quality[key]})
        answer = ("Here is the current data-quality assessment:\n\n" +
                  "\n".join(f"- {p}" for p in parts)) if parts else \
                 "No data-quality metrics are available in the current analysis result."
        chart = "quality"

    elif qtype in {"top", "bottom"}:
        answer = _ranking(dashboard, question) or \
                 "I could not find a ranked category, region, product, or channel series in the current dataset."
        chart = "ranking"

    elif qtype == "trend":
        rows = dashboard.get("trend") or []
        change = _series_change(rows)
        if change:
            direction = "increased" if change["change"] > 0 else "decreased" if change["change"] < 0 else "was unchanged"
            pct = f" ({abs(change['pct']):.2f}%)" if change["pct"] is not None else ""
            answer = f"The latest period **{direction}** from {_fmt(change['from'])} to {_fmt(change['to'])}{pct} for **{change['key']}**."
            evidence.append({"source": "dashboard.trend", "periods": rows[-2:], "metric": change["key"]})
        elif rows:
            answer = "A trend series is available, but it does not contain enough numeric values to calculate the latest change."
            evidence.append({"source": "dashboard.trend", "periods": rows[-8:]})
        else:
            answer = "No time-series trend is available in the current analysis result."
        chart = "trend"

    elif qtype == "correlation":
        correlations = dashboard.get("correlations") or dashboard.get("correlation") or []
        answer = "Correlation evidence is available in the current analysis result." if correlations else \
                 "No correlation matrix/result is available in the current analysis result."
        if correlations:
            evidence.append({"source": "dashboard.correlations", "value": correlations})
        chart = "correlation"

    elif qtype == "anomaly":
        anomalies = dashboard.get("anomalies") or dashboard.get("outliers") or quality.get("numeric_outliers")
        answer = "The current analysis contains anomaly/outlier evidence." if anomalies else \
                 "No anomaly/outlier results are available in the current analysis result."
        if anomalies:
            evidence.append({"source": "dashboard.anomalies_or_quality", "value": anomalies})
        chart = "anomaly"

    else:
        key, value = _find_kpi(kpis, question)
        if key is not None:
            label = key.replace("_", " ").title()
            nk = _norm(key)
            formatted = _money(value) if any(x in nk for x in ["revenue", "profit", "salary", "cost", "spend", "charges"]) else \
                        (_pct(value) if any(x in nk for x in ["rate", "margin", "ctr"]) else _fmt(value))
            answer = f"**{label}: {formatted}**"
            evidence.append({"source": "dashboard.kpis", "field": key, "value": value})
        else:
            preview = result.get("preview") or {}
            cols, rows = preview.get("columns") or [], preview.get("rows") or []
            if "column" in _norm(question) and cols:
                answer = "The available columns are:\n\n" + ", ".join(f"`{c}`" for c in cols)
                evidence.append({"source": "preview.columns", "value": cols})
            elif rows:
                answer = f"The current dataset preview contains **{len(rows)} rows** and **{len(cols)} columns**."
                evidence.append({"source": "preview", "rows": len(rows), "columns": len(cols)})
            else:
                answer = "I could not find a matching KPI or analysis series in the current result."

    return {
        "answer": answer,
        "question": question,
        "intent": qtype,
        "evidence": evidence,
        "suggested_chart": chart,
        "dataset": {
            "filename": result.get("filename"),
            "module": result.get("module"),
            "shape": result.get("shape"),
            "columns": result.get("columns"),
        },
    }
