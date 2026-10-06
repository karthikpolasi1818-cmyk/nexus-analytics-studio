from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .ai_analyst import analyze_question, load_dataframe

router = APIRouter(prefix="/api/ai", tags=["AI Analyst"])


@router.get("/health")
def ai_health():
    return {
        "status": "ok",
        "service": "NEXUS AI Analyst",
        "version": "1.0.0",
    }


@router.post("/ask")
async def ask_ai(
    file: UploadFile = File(...),
    question: str = Form(...),
    module: str = Form("Sales Analytics"),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    filename = file.filename or "uploaded_file"
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        sheet_name, df = load_dataframe(filename, content)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not load the dataset: {exc}",
        ) from exc

    if df is None:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file did not contain a usable dataframe.",
        )

    result = analyze_question(df, question, module)
    result["filename"] = filename
    result["sheet"] = sheet_name
    return result
