from __future__ import annotations



from executive_intelligence_service import build_executive_intelligence

from advanced_ai_service import analyze_question
from ai_context_service import build_ai_context





import inspect

import json

import math

import sys

from pathlib import Path

from typing import Any, Optional



import numpy as np

import pandas as pd



from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse





# =========================================================

# PROJECT PATH CONFIGURATION

# =========================================================



# api/main.py



#

# The existing NEXUS Python application lives inside "app".

#

# This allows:

#   from ingestion.loader import load_file

#   from core.profiler import profile_dataframe

#   from analytics.sales.engine import ...

#

# to work correctly.



PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_DIR = PROJECT_ROOT / "app"



if not APP_DIR.exists():

    raise RuntimeError(

        f"NEXUS app directory was not found: {APP_DIR}"

    )



if str(APP_DIR) not in sys.path:

    sys.path.insert(0, str(APP_DIR))



if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))





# =========================================================

# NEXUS APPLICATION IMPORTS

# =========================================================



from ingestion.loader import load_file

from core.profiler import profile_dataframe

from core.detector import detect_modules

from dashboard_service import build_dashboard as build_universal_dashboard

from visualization_service import build_visualizations

from cross_file_service import analyze_cross_file_relationships

from quality_service import analyze_data_quality

from enterprise_dashboard_service import build_enterprise_dashboard





# =========================================================

# FASTAPI APPLICATION

# =========================================================



app = FastAPI(

    title="NEXUS Analytics Studio API",

    version="1.0.0",

    description=(

        "API layer connecting the NEXUS Analytics Studio "

        "Next.js website to the existing Python analytics "

        "engines."

    ),

)





# =========================================================

# CORS

# =========================================================



ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nexus-analytics-studio.vercel.app",
    "https://nexus-analytics-studio-7tsml49cd.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://nexus-analytics-studio(?:-[a-zA-Z0-9-]+)?\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# =========================================================

# ANALYTICS MODULES

# =========================================================



MODULE_SLUGS = {

    "Sales Analytics": "sales",

    "Customer Analytics": "customer",

    "Marketing Analytics": "marketing",

    "Financial Analytics": "financial",

    "Supply Chain Analytics": "supply_chain",

    "Product Analytics": "product",

    "Operations Analytics": "operations",

    "HR Analytics": "hr",

    "Fraud Analytics": "fraud",

    "Healthcare Analytics": "healthcare",

    "Manufacturing Analytics": "manufacturing",

    "E-commerce Analytics": "ecommerce",

}





# =========================================================

# UPLOAD FILE ADAPTER

# =========================================================



class UploadedFileAdapter:

    """

    Adapter that allows the existing Streamlit-oriented

    NEXUS loader to work with FastAPI UploadFile data.

    """



    def __init__(

        self,

        name: str,

        content: bytes,

        content_type: str | None = None,

    ):

        self.name = name

        self.filename = name

        self.type = content_type or ""

        self.content_type = content_type or ""

        self._content = content



    def getvalue(self) -> bytes:

        return self._content



    def read(self) -> bytes:

        return self._content



    def getbuffer(self):

        return memoryview(self._content)



    @property

    def size(self) -> int:

        return len(self._content)





# =========================================================

# JSON SAFE CONVERSION

# =========================================================



def _json_safe(value: Any) -> Any:

    """

    Convert pandas / NumPy / Python objects into values that

    FastAPI can safely serialize into JSON.

    """



    if value is None:

        return None



    # Standard Python values

    if isinstance(

        value,

        (str, int, bool),

    ):

        return value



    # Python float

    if isinstance(value, float):

        if math.isnan(value):

            return None



        if math.isinf(value):

            return None



        return value



    # Path

    if isinstance(value, Path):

        return str(value)



    # Pandas timestamp

    if isinstance(value, pd.Timestamp):

        return value.isoformat()



    # Python datetime-like values

    if hasattr(value, "isoformat") and not isinstance(

        value,

        (str, bytes),

    ):

        try:

            return value.isoformat()

        except Exception:

            pass



    # NumPy scalar

    if isinstance(

        value,

        (

            np.integer,

            np.int64,

            np.int32,

        ),

    ):

        return int(value)



    if isinstance(

        value,

        (

            np.floating,

            np.float64,

            np.float32,

        ),

    ):

        number = float(value)



        if math.isnan(number):

            return None



        if math.isinf(number):

            return None



        return number



    if isinstance(value, np.bool_):

        return bool(value)



    # NumPy array

    if isinstance(value, np.ndarray):

        return [

            _json_safe(item)

            for item in value.tolist()

        ]



    # Dictionary

    if isinstance(value, dict):

        return {

            str(key): _json_safe(item)

            for key, item in value.items()

        }



    # List / tuple / set

    if isinstance(

        value,

        (

            list,

            tuple,

            set,

        ),

    ):

        return [

            _json_safe(item)

            for item in value

        ]



    # DataFrame

    if isinstance(

        value,

        pd.DataFrame,

    ):



        frame = value.copy()



        frame = frame.replace(

            {

                np.nan: None,

                np.inf: None,

                -np.inf: None,

            }

        )



        return {

            "type": "dataframe",

            "rows": int(frame.shape[0]),

            "columns_count": int(frame.shape[1]),

            "columns": [

                str(column)

                for column in frame.columns

            ],

            "data": [

                {

                    str(column): _json_safe(

                        row[column]

                    )

                    for column in frame.columns

                }

                for _, row in frame.iterrows()

            ],

        }



    # Series

    if isinstance(

        value,

        pd.Series,

    ):

        return {

            str(key): _json_safe(item)

            for key, item in value.items()

        }



    # Try NumPy-compatible .item()

    try:



        if hasattr(value, "item"):

            return _json_safe(

                value.item()

            )



    except Exception:

        pass



    # Last resort

    return str(value)





# =========================================================

# LOAD USING EXISTING NEXUS LOADER

# =========================================================



def _load_with_existing_loader(

    filename: str,

    content: bytes,

    content_type: str | None,

):

    """

    Reuse the existing NEXUS ingestion pipeline rather than

    creating a second ingestion system for the website.

    """



    adapter = UploadedFileAdapter(

        name=filename,

        content=content,

        content_type=content_type,

    )



    return load_file(adapter)





# =========================================================

# FIND BEST DATAFRAME

# =========================================================



def _best_dataframe(

    dataset: dict[str, Any],

):

    """

    Select the most useful dataframe from the loader result.



    For Excel files with multiple sheets, the largest non-empty

    dataframe is selected.

    """



    dataframes = dataset.get(

        "dataframes",

        {},

    )



    if not dataframes:

        return None, None



    candidates = []



    for sheet_name, frame in dataframes.items():



        if (

            isinstance(frame, pd.DataFrame)

            and not frame.empty

        ):



            candidates.append(

                (

                    str(sheet_name),

                    frame,

                )

            )



    if not candidates:

        return None, None



    # Select largest dataframe.

    candidates.sort(

        key=lambda item: (

            item[1].shape[0]

            * item[1].shape[1]

        ),

        reverse=True,

    )



    return candidates[0]





# =========================================================

# ENGINE IMPORT

# =========================================================



def _import_engine(

    module_name: str,

):

    """

    Dynamically import:



        analytics.<module>.engine

    """



    import importlib



    module_path = (

        f"analytics.{module_name}.engine"

    )



    try:



        return importlib.import_module(

            module_path

        )



    except Exception as exc:



        raise HTTPException(

            status_code=500,

            detail=(

                f"Could not import analytics "

                f"engine '{module_name}': {exc}"

            ),

        )





# =========================================================

# FIND COMPATIBLE ENGINE FUNCTIONS

# =========================================================



def _callable_engine_functions(

    module_name: str,

):

    """

    Find public functions inside the selected analytics

    engine that can receive a dataframe as their only

    required argument.

    """



    engine = _import_engine(

        module_name

    )



    functions = []



    for name, fn in inspect.getmembers(

        engine,

        inspect.isfunction,

    ):



        # Ignore imported functions.

        if fn.__module__ != engine.__name__:

            continue



        # Ignore private functions.

        if name.startswith("_"):

            continue



        try:



            signature = inspect.signature(

                fn

            )



        except (

            TypeError,

            ValueError,

        ):



            continue



        parameters = list(

            signature.parameters.values()

        )



        required = [

            parameter

            for parameter in parameters

            if (

                parameter.default

                is inspect.Parameter.empty

                and parameter.kind

                in (

                    inspect.Parameter.POSITIONAL_ONLY,

                    inspect.Parameter.POSITIONAL_OR_KEYWORD,

                )

            )

        ]



        # We currently call only functions with exactly

        # one required parameter.

        if len(required) != 1:

            continue



        first_parameter = required[0]



        allowed_dataframe_names = {

            "df",

            "data",

            "frame",

            "dataset",

            "dataframe",

        }



        if (

            first_parameter.name.lower()

            not in allowed_dataframe_names

        ):

            continue



        functions.append(

            (

                name,

                fn,

            )

        )



    return engine, functions





# =========================================================

# FUNCTION PRIORITY

# =========================================================



def _function_priority(

    name: str,

) -> int:



    lower = name.lower()



    if "kpi" in lower:

        return 0



    if "summary" in lower:

        return 1



    if "insight" in lower:

        return 2



    if "overview" in lower:

        return 3



    if "performance" in lower:

        return 4



    if "trend" in lower:

        return 5



    if "analysis" in lower:

        return 6



    if "analytics" in lower:

        return 7



    return 100





# =========================================================

# RUN ANALYTICS ENGINE

# =========================================================



def _run_engine(

    module_name: str,

    df: pd.DataFrame,

) -> dict[str, Any]:



    engine, functions = (

        _callable_engine_functions(

            module_name

        )

    )



    results: dict[str, Any] = {}

    errors: dict[str, str] = {}



    functions.sort(

        key=lambda item: (

            _function_priority(

                item[0]

            ),

            item[0],

        )

    )



    executed = 0



    # Prevent an engine with many functions from generating

    # an excessively large API response.

    max_functions = 12



    for name, fn in functions:



        if executed >= max_functions:

            break



        try:



            result = fn(

                df.copy()

            )



            results[name] = _json_safe(

                result

            )



            executed += 1



        except Exception as exc:



            errors[name] = str(exc)



    return {

        "engine": (

            f"analytics.{module_name}.engine"

        ),

        "functions_available": len(

            functions

        ),

        "functions_executed": executed,

        "results": results,

        "errors": errors,

    }





# =========================================================

# HEALTH

# =========================================================



@app.get("/api/health")

def health():



    return {

        "status": "ok",

        "service": (

            "NEXUS Analytics Studio API"

        ),

        "version": "1.0.0",

    }





# =========================================================

# ROOT

# =========================================================



@app.get("/")

def root():



    return {

        "name": "NEXUS Analytics Studio API",

        "status": "running",

        "version": "1.0.0",

        "health": "/api/health",

        "docs": "/docs",

        "modules": "/api/modules",

        "analyze": "/api/analyze",

    }





# =========================================================

# MODULES

# =========================================================



@app.get("/api/modules")

def modules():



    return {

        "modules": [

            {

                "name": name,

                "slug": slug,

            }

            for name, slug

            in MODULE_SLUGS.items()

        ]

    }





# =========================================================

# ANALYZE FILE

# =========================================================



@app.post("/api/analyze")

async def analyze(

    file: UploadFile = File(...),

    module: str = Form(

        "Sales Analytics"

    ),

):



    # =====================================================

    # VALIDATE MODULE

    # =====================================================



    if module not in MODULE_SLUGS:



        raise HTTPException(

            status_code=400,

            detail=(

                f"Unknown analytics module: "

                f"{module}"

            ),

        )



    # =====================================================

    # READ FILE

    # =====================================================



    filename = (

        file.filename

        or "uploaded_file"

    )



    content = await file.read()



    if not content:



        raise HTTPException(

            status_code=400,

            detail="Uploaded file is empty.",

        )



    # =====================================================

    # INGESTION

    # =====================================================



    try:



        dataset = (

            _load_with_existing_loader(

                filename,

                content,

                file.content_type,

            )

        )



    except Exception as exc:



        raise HTTPException(

            status_code=422,

            detail=(

                f"File ingestion failed: "

                f"{exc}"

            ),

        )



    if not isinstance(

        dataset,

        dict,

    ):



        raise HTTPException(

            status_code=500,

            detail=(

                "The NEXUS loader returned "

                "an unexpected result."

            ),

        )



    # =====================================================

    # FIND DATAFRAME

    # =====================================================



    sheet_name, df = _best_dataframe(

        dataset

    )



    # =====================================================

    # DOCUMENT-ONLY RESPONSE

    # =====================================================



    if df is None:



        extracted_text = dataset.get(

            "text",

            "",

        )



        return {

            "success": True,

            "filename": filename,

            "module": module,

            "sheet": None,

            "shape": None,

            "columns": [],

            "profile": {},

            "detected_domains": [],

            "analytics": {},

            "analytics_errors": {},

            "dashboard": {

                "domain": module,

                "kpis": [],

                "charts": [],

                "tables": [],

                "insights": ["No dataframe was available for dashboard generation."],

                "quality": {

                    "score": 0,

                    "grade": "Unavailable",

                    "rows": 0,

                    "columns": 0,

                    "missing_cells": 0,

                    "missing_rate": 0,

                    "duplicate_rows": 0,

                    "duplicate_rate": 0,

                    "outlier_rows": 0,

                    "invalid_date_values": 0,

                    "missing_columns": [],

                    "outliers": [],

                    "invalid_dates": [],

                    "constant_columns": [],

                    "high_cardinality": [],

                    "possible_pii": [],

                    "duplicate_keys": [],

                    "issues": [{

                        "severity": "info",

                        "type": "no_dataframe",

                        "message": "No dataframe was available for data-quality analysis.",

                        "recommendation": "Upload a tabular dataset for quality analysis."

                    }]

                }

            },

            "preview": {"columns": [], "rows": []},

            "document": {

                "text": (

                    extracted_text[:20000]

                    if extracted_text

                    else ""

                ),

                "pdf_tables_detected": dataset.get(

                    "pdf_tables_detected",

                    False,

                ),

                "pdf_table_count": dataset.get(

                    "pdf_table_count",

                    0,

                ),

            },

        }



    # =====================================================

    # CLEAN DATAFRAME

    # =====================================================



    df = df.copy()



    # Remove completely empty rows.

    df = df.dropna(

        how="all"

    )



    # Convert column names to strings.

    df.columns = [

        str(column)

        for column in df.columns

    ]



    if df.empty:



        raise HTTPException(

            status_code=422,

            detail=(

                "The detected dataframe is empty "

                "after removing empty rows."

            ),

        )



    # =====================================================

    # PROFILE

    # =====================================================



    try:



        profile = profile_dataframe(

            df

        )



    except Exception as exc:



        profile = {

            "rows": int(

                df.shape[0]

            ),

            "columns": int(

                df.shape[1]

            ),

            "error": str(exc),

        }



    # =====================================================

    # DOMAIN DETECTION

    # =====================================================



    try:



        rankings = detect_modules(

            df.columns

        )



        detection = []



        for ranking in rankings[:5]:



            try:



                name = ranking[0]

                score = ranking[1]



                detection.append(

                    {

                        "module": str(

                            name

                        ),

                        "score": float(

                            score

                        ),

                    }

                )



            except (

                IndexError,

                TypeError,

                ValueError,

            ):



                continue



    except Exception as exc:



        detection = [

            {

                "error": str(exc)

            }

        ]



    # =====================================================

    # RUN SELECTED ANALYTICS ENGINE

    # =====================================================



    module_slug = MODULE_SLUGS[

        module

    ]



    try:



        analytics = _run_engine(

            module_slug,

            df,

        )



    except HTTPException:



        raise



    except Exception as exc:



        analytics = {

            "engine": (

                f"analytics."

                f"{module_slug}."

                f"engine"

            ),

            "functions_available": 0,

            "functions_executed": 0,

            "results": {},

            "errors": {

                "engine": str(exc)

            },

        }



    # =====================================================

    # BUILD DOMAIN DASHBOARD FROM THE FULL DATAFRAME

    # =====================================================



    try:

        # Enterprise dashboard is generated directly from the uploaded

        # dataframe so domain KPIs use the real column names in the file.

        dashboard = build_enterprise_dashboard(

            df,

            module,

        )



        # Phase 3: Data Quality Intelligence

        dashboard["quality"] = analyze_data_quality(df)

    except Exception as exc:

        dashboard = {

            "domain": module,

            "kpis": [],

            "charts": [],

            "tables": [],

            "insights": [

                f"Dashboard generation error: {exc}"

            ],

        }



    # =====================================================



    # AI ANALYST FULL-DATA CONTEXT



    # =====================================================




    try:



        ai_context = build_ai_context(



            df,



            module,



        )




    except Exception as exc:



        ai_context = {



            "version": "1.1",



            "available": False,



            "module": module,



            "error": str(exc),



        }





    # =====================================================

    # NEXT.JS TABLE PREVIEW

    # =====================================================



    preview_frame = df.head(25).copy()

    preview_frame = preview_frame.replace({

        np.nan: None,

        np.inf: None,

        -np.inf: None,

    })



    preview = {

        "columns": [

            str(column)

            for column in preview_frame.columns

        ],

        "rows": [

            [

                _json_safe(value)

                for value in row

            ]

            for row in preview_frame.itertuples(

                index=False,

                name=None,

            )

        ],

    }



    # =====================================================

    # FINAL RESPONSE

    # =====================================================



    return {

        "success": True,



        "filename": filename,



        "module": module,



        "module_slug": module_slug,



        "sheet": sheet_name,



        "shape": {

            "rows": int(

                df.shape[0]

            ),

            "columns": int(

                df.shape[1]

            ),

        },



        "columns": [

            str(column)

            for column in df.columns

        ],



        "profile": _json_safe(

            profile

        ),



        "detected_domains": detection,



        "analytics": analytics,



        "dashboard": dashboard,



        "ai_context": ai_context,

        "preview": preview,

    }





# =========================================================

# STARTUP

# =========================================================



@app.on_event("startup")

async def startup_event():



    print()

    print("=" * 60)

    print(" NEXUS Analytics Studio API")

    print("=" * 60)

    print(

        " API:     http://127.0.0.1:8000"

    )

    print(

        " Docs:    http://127.0.0.1:8000/docs"

    )

    print(

        " Health:  http://127.0.0.1:8000/api/health"

    )

    print(

        " Modules: http://127.0.0.1:8000/api/modules"

    )

    print("=" * 60)

    print()



from api.ai_routes import router as ai_analyst_router

app.include_router(ai_analyst_router)







# =====================================================

# CROSS-FILE INTELLIGENCE

# =====================================================



@app.post("/api/cross-file/analyze")

async def cross_file_analyze(

    files: list[UploadFile] = File(...)

):

    """

    Analyze relationships between multiple uploaded datasets.



    The endpoint:

    - loads all supported files through the existing NEXUS loader

    - extracts DataFrames

    - detects likely relationships between datasets

    - identifies possible join keys

    - performs safe relationship analysis

    """



    if not files:

        raise HTTPException(

            status_code=400,

            detail="At least two files are required.",

        )



    datasets = []



    for uploaded_file in files:



        filename = uploaded_file.filename or "uploaded_file"



        content = await uploaded_file.read()



        if not content:

            continue



        try:

            dataset = _load_with_existing_loader(

                filename,

                content,

                uploaded_file.content_type,

            )

        except Exception as exc:

            raise HTTPException(

                status_code=422,

                detail=f"File ingestion failed for {filename}: {exc}",

            )



        sheet_name, df = _best_dataframe(dataset)



        if df is None or df.empty:

            continue



        datasets.append(

            {

                "filename": filename,

                "sheet": sheet_name,

                "dataframe": df,

            }

        )



    if len(datasets) < 2:

        raise HTTPException(

            status_code=400,

            detail="At least two files containing tabular data are required.",

        )



    result = analyze_cross_file_relationships(datasets)



    return {

        "success": True,

        **result,

    }



class AdvancedAIRequest(BaseModel):

    question: str

    dataset: Optional[dict] = None
    conversation: Optional[list] = None





@app.post("/api/ai/v2/ask")

async def advanced_ai_ask(payload: AdvancedAIRequest):

    return analyze_question(payload.question, payload.dataset or {}, payload.conversation or [])



@app.post("/api/executive/intelligence")

async def executive_intelligence(payload: dict):

    return build_executive_intelligence(payload or {})


