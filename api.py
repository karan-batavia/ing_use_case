from __future__ import annotations

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import os
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from fastapi.templating import Jinja2Templates
# --- project services
from src.mongodb_service import MongoDBService
from src.dependencies import get_current_user, get_current_admin
from src.gemini_service import get_gemini_service
from src.sensitivity_classifier import get_classifier_service

# file handlers
from src.file_handler.docx_to_txt import DOCXToTextConverter
from src.file_handler.html_to_txt import HTMLToTextConverter
from src.file_handler.read_pdf_file import PDFToTextConverter
from src.file_handler.read_png_file import ImageToTextConverter

# auth
from src.auth import (
    TokenResponse,
    UserCreate,
    UserLogin,
    UserInDB,
    verify_password,
    get_password_hash,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# audit
from src.audit_log import (
    save_document,
    append_audit_event,
    build_client_info,
    init_audit_env,
    update_audit_event,
    NDJSON_LOG,
    HTML_LOG,
    ORIGINAL_DIR,
    SCRUBBED_DIR,
    PREDICTION_DIR,
    DESCRUBBED_DIR,
)

# =========================
# Paths & FastAPI init
# =========================
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"          # <— absolute
TMP_DIR    = BASE_DIR / "tmp"
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
TMP_DIR.mkdir(exist_ok=True)

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title="ING Prompt Scrubber API",
    description="API for scrubbing, AI predictions, and descrubbing.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount /static using an absolute path
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

mongodb_service = MongoDBService()

# =========================
# Models
# =========================
class TextScrubRequest(BaseModel):
    text: str

class TextScrubResponse(BaseModel):
    scrubbed_text: str
    matches_found: int
    original_length: int
    scrubbed_length: int
    reduction_percentage: float
    processed_at: datetime
    success: bool
    session_id: Optional[str] = None

class DescrubRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class DescrubResponse(BaseModel):
    descrubbed_text: str
    success: bool
    processed_at: datetime

class PredictionRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    model_name: Optional[str] = "gemini-pro"
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.4
    session_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    model_used: str
    success: bool
    session_id: str
    processed_at: datetime
    mask_summary: Dict[str, int] = Field(default_factory=dict)
    category: Optional[str] = None
    category_distribution: Dict[str, float] = Field(default_factory=dict)


class CategoryStat(BaseModel):
    id: str
    description: str
    support: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class CategoryStatsResponse(BaseModel):
    accuracy: Optional[float] = None
    categories: List[CategoryStat]

class FileScrubResponse(BaseModel):
    filename: str
    scrubbed_text: str
    original_text: str
    matches_found: int
    reduction_percentage: float
    processed_at: datetime
    success: bool
    session_id: Optional[str] = None

# =========================
# Startup: ensure audit env
# =========================
@app.on_event("startup")
def _startup():
    init_audit_env()  # ensures logs folder, NDJSON + HTML exist

# =========================
# Root & health
# =========================
@app.get("/", response_class=HTMLResponse)
def serve_frontend(request: Request):
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return {
        "message": "ING Prompt Scrubber API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mongodb_connected": mongodb_service.is_connected(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# =========================
# Auth
# =========================
@app.post("/auth/register", response_model=Dict[str, str])
def register(user: UserCreate, request: Request):
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    hashed_password = get_password_hash(user.password)
    result = mongodb_service.create_user(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_admin=getattr(user, "is_admin", False),
        role=getattr(user, "role", "user"),
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    ci = build_client_info(request)
    append_audit_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **ci,
        "action": "register",
        "user_email": user.email,
        "user_name": user.full_name or "",
        "original_file": "",
        "scrubbed_file": "",
        "descrubbed_file": "",
        "prediction_file": "",
    })
    return {"message": "User registered successfully", "user_id": result["user_id"]}

@app.post("/auth/login", response_model=TokenResponse)
def login(user_credentials: UserLogin, request: Request):
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    user_data = mongodb_service.get_user_by_email(user_credentials.email)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_in_db = UserInDB(**user_data)
    if not verify_password(user_credentials.password, user_in_db.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = user_data.get("user_id", user_data.get("_id", "unknown"))
    role = user_data.get("role", "user")

    access_token = create_access_token(
        data={"sub": user_in_db.email, "user_id": user_id, "role": role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    ci = build_client_info(request)
    append_audit_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **ci,
        "action": "login",
        "user_email": user_in_db.email,
        "user_name": user_in_db.full_name or "",
        "original_file": "",
        "scrubbed_file": "",
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "role": role,
    }

# =========================
# Scrub / Descrub
# =========================
@app.post("/scrub", response_model=TextScrubResponse)
async def scrub_text(
    payload: TextScrubRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        classifier = get_classifier_service()
        redaction = classifier.redact_sensitive_info(payload.text)

        scrubbed_text = redaction.redacted_text
        matches_found = redaction.total_redacted
        orig_len = len(payload.text)
        scrub_len = len(scrubbed_text)
        reduction = round((1 - scrub_len / orig_len) * 100, 2) if orig_len else 0

        session_id = mongodb_service.create_session(current_user["user_id"], "scrub")

        mongodb_service.save_redaction_record(
            session_id=session_id,
            user_id=current_user["user_id"],
            original_text=payload.text,
            redacted_text=scrubbed_text,
            detections=redaction.detections,
        )

        base = current_user.get("email", "user").replace("@", "_")
        orig_path = save_document("original", f"{base}_orig", payload.text)
        scrubbed_path = save_document("scrubbed", f"{base}_scrub", scrubbed_text)

        ci = build_client_info(request)
        append_audit_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **ci,
            "action": "scrub_text",
            "session_id": session_id,
            "user_email": current_user.get("email", ""),
            "user_name": current_user.get("full_name", ""),
            "original_file": orig_path,
            "scrubbed_file": scrubbed_path,
        })

        return TextScrubResponse(
            scrubbed_text=scrubbed_text,
            matches_found=matches_found,
            original_length=orig_len,
            scrubbed_length=scrub_len,
            reduction_percentage=reduction,
            processed_at=datetime.now(),
            success=True,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrubbing failed: {e}")

@app.post("/scrub-file", response_model=FileScrubResponse)
async def scrub_file(
    file: UploadFile = File(...),
    request: Request = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        content = await file.read()
        original_path = Path(file.filename or "uploaded_file")
        ext = (original_path.suffix or "").lower()
        ext_clean = ext.lstrip(".")
        tmp_path = TMP_DIR / f"{uuid.uuid4().hex[:8]}_{original_path.name or 'upload'}"
        with open(tmp_path, "wb") as f:
            f.write(content)

        # extract text
        if ext_clean == "pdf":
            text = PDFToTextConverter().extract_text_from_pdf(str(tmp_path))
        elif ext_clean == "docx":
            conv = DOCXToTextConverter()
            tmp_txt = tmp_path.with_suffix(".txt")
            text = ""
            if conv.convert_docx_to_txt(str(tmp_path), str(tmp_txt)):
                text = tmp_txt.read_text(encoding="utf-8", errors="ignore")
                tmp_txt.unlink(missing_ok=True)
        elif ext_clean in {"html", "htm"}:
            conv = HTMLToTextConverter()
            tmp_txt = tmp_path.with_suffix(".txt")
            text = ""
            if conv.convert_html_to_txt(str(tmp_path), str(tmp_txt)):
                text = tmp_txt.read_text(encoding="utf-8", errors="ignore")
                tmp_txt.unlink(missing_ok=True)
        elif ext_clean in {"png", "jpg", "jpeg"}:
            text = ImageToTextConverter(languages="eng+fra+nld").extract_text_from_image(str(tmp_path))
        else:
            text = content.decode("utf-8", errors="ignore")
        tmp_path.unlink(missing_ok=True)

        classifier = get_classifier_service()
        redaction = classifier.redact_sensitive_info(text)
        scrubbed_text = redaction.redacted_text
        matches_found = redaction.total_redacted

        orig_len = len(text)
        scrub_len = len(scrubbed_text)
        reduction = round((1 - scrub_len / orig_len) * 100, 2) if orig_len else 0

        session_id = mongodb_service.create_session(current_user["user_id"], "scrub-file")
        mongodb_service.save_redaction_record(
            session_id=session_id,
            user_id=current_user["user_id"],
            original_text=text,
            redacted_text=scrubbed_text,
            detections=redaction.detections,
        )
        original_stem = original_path.stem or current_user.get("email", "user").replace("@", "_")
        orig_filename = original_path.name or f"{original_stem}{ext or '.bin'}"
        scrub_ext = ext if ext in {".txt"} else ".txt"
        scrub_filename = f"{original_stem}_scrubbed{scrub_ext}"

        orig_path = save_document(
            "original",
            original_stem,
            content,
            ext=ext or ".bin",
            filename=orig_filename,
        )
        scrubbed_path = save_document(
            "scrubbed",
            f"{original_stem}_scrubbed",
            scrubbed_text,
            ext=scrub_ext,
            filename=scrub_filename,
        )

        ci = build_client_info(request)
        append_audit_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **ci,
            "action": "scrub_file",
            "session_id": session_id,
            "user_email": current_user.get("email", ""),
            "user_name": current_user.get("full_name", ""),
            "original_file": orig_path,
            "scrubbed_file": scrubbed_path,
        })

        return FileScrubResponse(
            filename=file.filename,
            scrubbed_text=scrubbed_text,
            original_text=text,
            matches_found=matches_found,
            reduction_percentage=reduction,
            processed_at=datetime.now(),
            success=True,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/descrub", response_model=DescrubResponse)
async def descrub_text(
    payload: DescrubRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        restored = mongodb_service.descrub_text(payload.text, payload.session_id)
        if not restored:
            raise HTTPException(status_code=404, detail="No mapping found for descrubbing")

        base = current_user.get("email", "user").replace("@", "_")
        descrub_path = save_document("descrubbed", f"{base}_descrub", restored)

        if payload.session_id:
            updated = update_audit_event(
                payload.session_id,
                {
                    "descrubbed_file": descrub_path,
                },
            )
            if not updated:
                print(f"[AUDIT] No audit entry updated for session {payload.session_id}")

        return DescrubResponse(
            descrubbed_text=restored,
            success=True,
            processed_at=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Descrubbing failed: {e}")

# =========================
# Prediction
# =========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        gemini = get_gemini_service()
        classifier = get_classifier_service()
        incoming_session = payload.session_id

        prompt_mask_summary: Dict[str, int] = {}
        classification_result = None
        if incoming_session:
            stored_redaction = mongodb_service.get_redaction_record_by_session(incoming_session)
            if stored_redaction:
                for detection in stored_redaction.get("detections", []) or []:
                    mask_type = detection.get("type")
                    if mask_type:
                        prompt_mask_summary[mask_type] = prompt_mask_summary.get(mask_type, 0) + 1

                original_text = stored_redaction.get("original_text")
                if original_text:
                    try:
                        classification_result = classifier.classify_text(original_text)
                    except Exception as classify_err:
                        print(f"[PREDICT] Classification failed for session {incoming_session}: {classify_err}")

        session_id = incoming_session or str(uuid.uuid4())
        raw_prediction = await asyncio.wait_for(
            gemini.generate_prediction(
                prompt=payload.prompt,
                context=payload.context,
                model_name=payload.model_name,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
            ),
            timeout=30.0,
        )

        redaction = classifier.redact_sensitive_info(raw_prediction)
        prediction = redaction.redacted_text

        mask_summary = prompt_mask_summary or (redaction.detection_summary or {})

        if classification_result is None:
            classification_source = payload.prompt or prediction
            if classification_source:
                try:
                    classification_result = classifier.classify_text(classification_source)
                except Exception as classify_err:
                    print(f"[PREDICT] Classification failed: {classify_err}")

        category_prediction = None
        category_distribution: Dict[str, float] = {}
        if classification_result:
            category_prediction = classification_result.prediction
            category_distribution = classification_result.probabilities or {}

        base = current_user.get("email", "user").replace("@", "_")
        pred_path = save_document("prediction", f"{base}_pred", prediction)

        updates = {
            "prediction_file": pred_path,
            "mask_summary": mask_summary,
            "prediction_model": payload.model_name,
        }

        if category_prediction:
            updates["category_prediction"] = category_prediction
        if category_distribution:
            updates["category_distribution"] = category_distribution

        updated = False
        if incoming_session:
            updated = update_audit_event(incoming_session, updates)

        if not updated:
            ci = build_client_info(request)
            append_audit_event({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **ci,
                "action": "predict",
                "session_id": session_id,
                "user_email": current_user.get("email", ""),
                "user_name": current_user.get("full_name", ""),
                "original_file": "",
                "scrubbed_file": "",
                "prediction_file": pred_path,
                "descrubbed_file": "",
                "mask_summary": mask_summary,
                "prediction_model": payload.model_name,
                "category_prediction": category_prediction,
                "category_distribution": category_distribution,
            })

        return PredictionResponse(
            prediction=prediction,
            model_used=payload.model_name,
            success=True,
            session_id=session_id,
            processed_at=datetime.now(),
            mask_summary=mask_summary,
            category=category_prediction,
            category_distribution=category_distribution,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# =========================
# Category statistics
# =========================


@app.get("/stats/categories", response_model=CategoryStatsResponse)
def get_category_statistics():
    classifier = get_classifier_service()
    metrics = classifier.get_model_metrics()
    category_info = classifier.get_category_info()

    categories: List[CategoryStat] = []
    for code, description in category_info.items():
        precision = metrics.precision.get(code) if metrics else None
        recall = metrics.recall.get(code) if metrics else None
        f1 = metrics.f1_score.get(code) if metrics else None
        support = metrics.support.get(code, 0) if metrics else 0

        categories.append(
            CategoryStat(
                id=code,
                description=description,
                support=int(support),
                precision=round(precision, 4) if precision is not None else None,
                recall=round(recall, 4) if recall is not None else None,
                f1_score=round(f1, 4) if f1 is not None else None,
            )
        )

    accuracy = None
    if metrics and metrics.accuracy is not None:
        accuracy = round(metrics.accuracy, 4)

    return CategoryStatsResponse(accuracy=accuracy, categories=categories)

# =========================
# Audit viewers
# =========================
@app.get("/audit/log")
def get_audit_log(
    current_admin: Dict[str, Any] = Depends(get_current_admin),
):
    if not os.path.exists(HTML_LOG):
        raise HTTPException(status_code=404, detail="Audit log not found")
    return FileResponse(HTML_LOG, media_type="text/html")

@app.get("/audit/ndjson")
def get_audit_ndjson(
    current_admin: Dict[str, Any] = Depends(get_current_admin),
):
    if not os.path.exists(NDJSON_LOG):
        raise HTTPException(status_code=404, detail="Audit NDJSON not found")
    return FileResponse(NDJSON_LOG, media_type="application/x-ndjson")

@app.get("/audit/log/download")
def download_audit_log(
    current_admin: Dict[str, Any] = Depends(get_current_admin),
):
    if not os.path.exists(NDJSON_LOG):
        raise HTTPException(status_code=404, detail="Audit NDJSON not found")
    return FileResponse(NDJSON_LOG, media_type="application/x-ndjson", filename="audit_log.ndjson")

@app.get("/audit/file/{kind}/{filename}")
def get_audit_file(
    kind: str,
    filename: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin),
):
    base_dirs = {
        "original": ORIGINAL_DIR,
        "scrubbed": SCRUBBED_DIR,
        "prediction": PREDICTION_DIR,
        "descrubbed": DESCRUBBED_DIR,
    }
    if kind not in base_dirs:
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = base_dirs[kind] / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="text/plain", filename=filename)

# =========================
# Main
# =========================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
