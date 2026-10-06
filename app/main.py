from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ============================================================
# NEXUS CORE IMPORTS
# ============================================================

from core.profiler import profile_dataframe
from core.detector import detect_modules


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NEXUS Analytics Studio API",
    version="1.0.0",
    description=(
        "API layer connecting the NEXUS Analytics Studio "
        "Next.js website to the existing Python analytics engines."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ANALYTICS MODULES
# ============================================================

ANALYTICS_MODULES = {
    "sales": "Sales Analytics",
    "customer": "Customer Analytics",
    "marketing": "Marketing Analytics",
    "financial": "Financial Analytics",
    "supply_chain": "Supply Chain Analytics",
    "product": "Product Analytics",
    "operations": "Operations Analytics",
    "hr": "HR Analytics",
    "fraud": "Fraud Analytics",
    "healthcare": "Healthcare Analytics",
    "manufacturing": "Manufacturing Analytics",
    "ecommerce": "E-commerce Analytics",
}


# ============================================================
# HELPERS
# ============================================================

def clean_value(value: Any):
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def dataframe_preview(df: pd.DataFrame, rows: int = 20):
    preview = df.head(rows).copy()

    for column in preview.columns:
        preview[column] = preview[column].map(clean_value)

    return preview.to_dict(orient="records")


def numeric_summary(df: pd.DataFrame):
    results = {}

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    id_keywords = {
        "id",
        "_id",
        "order_id",
        "customer_id",
        "transaction_id",
        "employee_id",
        "product_id",
        "campaign_id",
        "patient_id",
        "supplier_id",
        "machine_id",
    }

    for column in numeric_columns:
        normalized = str(column).strip().lower()

        is_id = (
            normalized in id_keywords
            or normalized.endswith("_id")
        )

        if is_id:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        results[str(column)] = {
            "sum": float(series.sum()),
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "count": int(series.count()),
        }

    return results


def build_chart_data(df: pd.DataFrame):
    """
    Creates simple chart-ready data for the Next.js frontend.
    """

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        return {
            "column": None,
            "labels": [],
            "values": [],
        }

    # Avoid ID columns where possible
    usable_columns = [
        column
        for column in numeric_columns
        if not str(column).strip().lower().endswith("_id")
        and str(column).strip().lower() != "id"
    ]

    selected_column = (
        usable_columns[0]
        if usable_columns
        else numeric_columns[0]
    )

    values = pd.to_numeric(
        df[selected_column],
        errors="coerce",
    ).dropna()

    values = values.head(12)

    return {
        "column": str(selected_column),
        "labels": [
            str(index + 1)
            for index in range(len(values))
        ],
        "values": [
            float(value)
            for value in values.tolist()
        ],
    }


def load_uploaded_dataframe(
    filename: str,
    content: bytes,
) -> pd.DataFrame:

    suffix = Path(filename).suffix.lower()

    try:

        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(content))

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(io.BytesIO(content))

        if suffix == ".json":
            return pd.read_json(io.BytesIO(content))

        if suffix == ".parquet":
            return pd.read_parquet(io.BytesIO(content))

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {filename}: {exc}",
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file format '{suffix}'. "
            "For dashboard analytics upload CSV, Excel, "
            "JSON, or Parquet."
        ),
    )


def get_detected_module(df: pd.DataFrame):
    try:
        rankings = detect_modules(df.columns)

        if rankings:
            module_name = rankings[0][0]
            score = rankings[0][1]

            return {
                "module": str(module_name),
                "score": int(score),
            }

    except Exception:
        pass

    return {
        "module": None,
        "score": 0,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "NEXUS Analytics Studio API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "nexus-analytics-api",
    }


# ============================================================
# MODULES
# ============================================================

@app.get("/api/modules")
def modules():

    return {
        "modules": [
            {
                "name": name,
                "slug": slug,
            }
            for slug, name in ANALYTICS_MODULES.items()
        ]
    }


# ============================================================
# MAIN ANALYTICS ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    module: str = Form(...),
):

    # --------------------------------------------------------
    # Normalize module
    # --------------------------------------------------------

    module = module.strip().lower()

    if module not in ANALYTICS_MODULES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown analytics module: {module}. "
                f"Valid modules: "
                f"{', '.join(ANALYTICS_MODULES.keys())}"
            ),
        )

    # --------------------------------------------------------
    # Read upload
    # --------------------------------------------------------

    filename = file.filename or "dataset.csv"

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Convert to dataframe
    # --------------------------------------------------------

    df = load_uploaded_dataframe(
        filename,
        content,
    )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded dataset contains no rows.",
        )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    try:
        profile = profile_dataframe(df)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset profiling failed: {exc}",
        )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    detected = get_detected_module(df)

    # --------------------------------------------------------
    # Numeric analytics
    # --------------------------------------------------------

    metrics = numeric_summary(df)

    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    chart = build_chart_data(df)

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    columns = [
        {
            "name": str(column),
            "dtype": str(df[column].dtype),
        }
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = {
        "success": True,

        "file": {
            "name": filename,
            "size": len(content),
        },

        "module": {
            "slug": module,
            "name": ANALYTICS_MODULES[module],
        },

        "detected_module": detected,

        "profile": {
            "rows": int(
                profile.get(
                    "rows",
                    len(df),
                )
            ),

            "columns": int(
                profile.get(
                    "columns",
                    len(df.columns),
                )
            ),

            "missing_percentage": float(
                profile.get(
                    "missing_percentage",
                    0,
                )
            ),

            "duplicate_rows": int(
                profile.get(
                    "duplicate_rows",
                    0,
                )
            ),

            "numeric_columns": [
                str(column)
                for column in profile.get(
                    "numeric_columns",
                    [],
                )
            ],
        },

        "columns": columns,

        "metrics": metrics,

        "chart": chart,

        "preview": dataframe_preview(df),

        "message": (
            f"{ANALYTICS_MODULES[module]} completed "
            f"successfully for {filename}."
        ),
    }

    return JSONResponse(
        content=response
    )


# ============================================================
# CROSS FILE ANALYSIS
# ============================================================

@app.post("/api/cross-file/analyze")
async def cross_file_analyze(
    files: list[UploadFile] = File(...),
    module: str = Form(...),
):

    module = module.strip().lower()

    if module not in ANALYTICS_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analytics module: {module}",
        )

    datasets = []

    for uploaded_file in files:

        filename = (
            uploaded_file.filename
            or "dataset.csv"
        )

        content = await uploaded_file.read()

        if not content:
            continue

        try:
            df = load_uploaded_dataframe(
                filename,
                content,
            )

        except HTTPException:
            continue

        datasets.append({
            "filename": filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [
                str(column)
                for column in df.columns
            ],
        })

    if not datasets:
        raise HTTPException(
            status_code=400,
            detail="No valid datasets were uploaded.",
        )

    return {
        "success": True,
        "module": {
            "slug": module,
            "name": ANALYTICS_MODULES[module],
        },
        "file_count": len(datasets),
        "datasets": datasets,
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("")
    print("=" * 65)
    print("NEXUS Analytics Studio API")
    print("=" * 65)

    print(
        "API:     http://127.0.0.1:8000"
    )

    print(
        "Docs:    http://127.0.0.1:8000/docs"
    )

    print(
        "Health:  http://127.0.0.1:8000/api/health"
    )

    print(
        "Modules: http://127.0.0.1:8000/api/modules"
    )

    print("=" * 65)
    print("")