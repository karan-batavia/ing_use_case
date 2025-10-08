from typing import Union, Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import os
import uuid
import asyncio
import subprocess
from datetime import datetime, timedelta
from src.mongodb_service import MongoDBService
from src.dependencies import get_current_user, get_current_admin
from src.gemini_service import get_gemini_service
from src.sensitivity_classifier import get_classifier_service
from src.file_handler.docx_to_txt import DOCXToTextConverter
from src.file_handler.html_to_txt import HTMLToTextConverter
from src.file_handler.read_pdf_file import PDFToTextConverter
from src.file_handler.read_png_file import ImageToTextConverter
from src.file_handler.write_docx_file import DOCXWriter
from src.file_handler.write_pdf_file import PDFWriter
from src.file_handler.write_html_file import HTMLWriter
from src.file_handler.write_txt_file import TXTWriter
from src.auth import (
    Token,
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserInDB,
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token,
    verify_admin_token,
    authenticate_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# Initialize FastAPI app
app = FastAPI(
    title="ING Prompt Scrubber API",
    description="API for text scrubbing and anonymization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize services
mongodb_service = MongoDBService()


# Pydantic models for request/response
class TextScrubRequest(BaseModel):
    text: str


class TextScrubResponse(BaseModel):
    scrubbed_text: str
    original_length: int
    scrubbed_length: int
    matches_found: int
    reduction_percentage: float
    processed_at: datetime
    # Sensitivity classification fields
    original_classification: Optional[str] = None
    scrubbed_classification: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_explanation: Optional[str] = None
    classification_available: bool = False


class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    timestamp: datetime


# File upload models
class FileUploadResponse(BaseModel):
    filename: str
    file_type: str
    file_size: int
    extracted_text: str
    text_length: int
    processed_at: datetime


class FileScrubRequest(BaseModel):
    session_id: Optional[str] = None
    output_format: Optional[str] = "text"  # "text", "original", "both"


class FileScrubResponse(BaseModel):
    filename: str
    file_type: str
    file_size: int
    original_text: str
    scrubbed_text: str
    original_length: int
    scrubbed_length: int
    matches_found: int
    reduction_percentage: float
    matches_detected: Dict[str, List[str]]
    processed_at: datetime
    session_id: str
    download_url: Optional[str] = None


class FileDownloadInfo(BaseModel):
    download_url: str
    filename: str
    file_type: str
    expires_at: datetime


class PredictionRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    model_name: Optional[str] = "gemini-pro"  # Stable Gemini model
    max_tokens: Optional[int] = 150  # Reduced for faster responses
    temperature: Optional[float] = 0.3  # Lower for faster, more focused responses
    session_id: Optional[str] = (
        None  # Optional session ID to maintain workflow consistency
    )


class PredictionResponse(BaseModel):
    prediction: str
    model_used: str
    success: bool
    session_id: str
    processed_at: datetime


# Sensitivity Classification Models
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    explanation: str
    success: bool


class RedactRequest(BaseModel):
    text: str


class Detection(BaseModel):
    type: str
    original: str
    placeholder: str


class RedactResponse(BaseModel):
    session_id: str
    original_text: str
    redacted_text: str
    detections: List[Detection]
    total_redacted: int
    detection_summary: Dict[str, int]
    success: bool


class TrainModelRequest(BaseModel):
    data_file: str = "raw_prompts.txt"
    model_type: str = "logreg"
    test_size: float = 0.2
    ngram_max: int = 2
    random_seed: int = 42


class TrainModelResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    support: Dict[str, int]
    confusion_matrix: List[List[int]]
    available: bool


# De-scrubbing Models
class DescrubRequest(BaseModel):
    session_id: str
    scrubbed_text: str


class RedactionRecord(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime
    original_text: str
    redacted_text: str
    detections: List[Detection]
    total_redacted: int


class DescrubResponse(BaseModel):
    success: bool
    session_id: str
    original_text: str
    redacted_text: str
    restored_text: str
    detections_restored: int
    message: str


class RedactionRecordsResponse(BaseModel):
    success: bool
    records: List[Dict[str, Any]]
    total_count: int


@app.get("/", response_model=Dict[str, str])
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "ING Prompt Scrubber API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        mongodb_connected=mongodb_service.is_connected(),
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def generate_prediction(
    request: PredictionRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate a prediction using Gemini AI based on prompt and context.
    Requires user authentication.
    """
    try:
        # Get Gemini service instance
        gemini_service = get_gemini_service()

        # Use provided session_id or generate a new one for this request
        session_id = request.session_id or str(uuid.uuid4())

        # Generate prediction using Gemini with timeout
        try:
            prediction_text = await asyncio.wait_for(
                gemini_service.generate_prediction(
                    prompt=request.prompt,
                    context=request.context,
                    model_name=request.model_name,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="AI generation timed out. Please try a shorter prompt or try again later.",
            )

        # Log the prediction request for the current user
        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="prediction_generated",
            details={
                "prompt_length": len(request.prompt),
                "context_provided": bool(request.context),
                "model_used": request.model_name or "llama3.2:1b",
                "endpoint": "/predict",
            },
        )

        return PredictionResponse(
            prediction=prediction_text,
            model_used=request.model_name or "llama3.2:1b",
            success=True,
            session_id=session_id,
            processed_at=datetime.now(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating prediction: {str(e)}"
        )


# Authentication Routes
@app.post("/auth/register", response_model=Dict[str, str])
def register(user: UserCreate):
    """Register a new user"""
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    # Hash the password
    hashed_password = get_password_hash(user.password)

    # Create user in database
    result = mongodb_service.create_user(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_admin=user.is_admin,
    )

    if not result["success"]:
        if "already exists" in result["error"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )

    return {"message": "User registered successfully", "user_id": result["user_id"]}


@app.post("/auth/login", response_model=TokenResponse)
def login(user_credentials: UserLogin):
    """Login user and return JWT token"""
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    # Get user from database
    user_data = mongodb_service.get_user_by_email(user_credentials.email)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_in_db = UserInDB(**user_data)

    # Verify password
    if not verify_password(user_credentials.password, user_in_db.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user_in_db.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Update last login
    mongodb_service.update_last_login(user_credentials.email)

    # Ensure we have a user_id (fallback in case migration failed)
    user_id = user_data.get("user_id", user_data.get("_id", "unknown"))

    # Create access token with user_id in payload
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_in_db.email, "user_id": user_id},
        expires_delta=access_token_expires,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
    }


@app.get("/stats/user/{user_id}")
def get_user_stats(user_id: str, admin_user: dict = Depends(get_current_admin)):
    """Get statistics for a specific user (Admin only)"""
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    stats = mongodb_service.get_user_stats(user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")

    return stats


@app.get("/stats/sessions")
def get_recent_sessions(limit: int = 10, admin_user: dict = Depends(get_current_admin)):
    """Get recent session statistics (Admin only)"""
    if not mongodb_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")

    # This would need to be implemented in mongodb_service
    return {"message": "Recent sessions endpoint - implementation needed"}


# File Upload Endpoints
@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    """Upload and extract text from file (Authenticated users only)"""
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Check file type
        allowed_types = {
            "text/plain": "txt",
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "text/html": "html",
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "text/csv": "csv",
            "application/csv": "csv",
        }

        file_extension = (
            file.filename.lower().split(".")[-1] if "." in file.filename else ""
        )
        content_type = file.content_type or ""

        if content_type not in allowed_types and file_extension not in [
            "txt",
            "pdf",
            "docx",
            "html",
            "htm",
            "png",
            "jpg",
            "jpeg",
            "csv",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, HTML, PNG, JPG, CSV",
            )

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Generate unique temp filename
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{file.filename}"
        temp_file_path = f"/tmp/{temp_filename}"

        # Save temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        extracted_text = ""

        try:
            # Extract text based on file type
            if content_type == "text/plain" or file_extension == "txt":
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()

            elif content_type == "application/pdf" or file_extension == "pdf":
                converter = PDFToTextConverter()
                extracted_text = converter.extract_text_from_pdf(temp_file_path)

            elif (
                content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or file_extension == "docx"
            ):
                converter = DOCXToTextConverter()
                temp_txt_path = temp_file_path.replace(".docx", ".txt")
                success = converter.convert_docx_to_txt(temp_file_path, temp_txt_path)
                if success:
                    with open(temp_txt_path, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                    os.remove(temp_txt_path)

            elif content_type == "text/html" or file_extension in ["html", "htm"]:
                converter = HTMLToTextConverter(
                    clean_whitespace=True, remove_empty_lines=True
                )
                temp_txt_path = temp_file_path.replace(".html", ".txt").replace(
                    ".htm", ".txt"
                )
                success = converter.convert_html_to_txt(temp_file_path, temp_txt_path)
                if success:
                    with open(temp_txt_path, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                    os.remove(temp_txt_path)

            elif content_type.startswith("image/") or file_extension in [
                "png",
                "jpg",
                "jpeg",
            ]:
                converter = ImageToTextConverter(
                    languages="eng+fra+nld", preprocess=True  # English, French, Dutch
                )
                extracted_text = converter.extract_text_from_image(temp_file_path)

            elif (
                content_type in ["text/csv", "application/csv"]
                or file_extension == "csv"
            ):
                import pandas as pd

                try:
                    # Read CSV and convert to text representation
                    df = pd.read_csv(temp_file_path, encoding="utf-8")
                    # Convert DataFrame to string representation with headers
                    extracted_text = df.to_string(index=False)
                except UnicodeDecodeError:
                    # Try alternative encodings
                    for encoding in ["latin-1", "iso-8859-1", "cp1252"]:
                        try:
                            df = pd.read_csv(temp_file_path, encoding=encoding)
                            extracted_text = df.to_string(index=False)
                            break
                        except:
                            continue
                    if not extracted_text:
                        raise HTTPException(
                            status_code=400,
                            detail="Could not read CSV file with any encoding",
                        )
                except Exception as e:
                    raise HTTPException(
                        status_code=400, detail=f"Error reading CSV file: {str(e)}"
                    )

            if not extracted_text:
                raise HTTPException(
                    status_code=400, detail="Could not extract text from file"
                )

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return FileUploadResponse(
            filename=file.filename,
            file_type=content_type or f"file/{file_extension}",
            file_size=file_size,
            extracted_text=extracted_text,
            text_length=len(extracted_text),
            processed_at=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/scrub-file", response_model=FileScrubResponse)
async def scrub_file(
    file: UploadFile = File(...),
    request: FileScrubRequest = FileScrubRequest(),
    current_user: dict = Depends(get_current_user),
):
    """Upload file, extract text, and scrub sensitive information (Authenticated users only)"""
    try:
        # First extract text from file (reuse upload logic)
        upload_response = await upload_file(file, current_user)

        # Extract user_id from the authenticated user data
        authenticated_user_id = current_user.get(
            "user_id", current_user.get("_id", "unknown")
        )

        # Use provided session_id or create new one for workflow consistency
        session_id = request.session_id or mongodb_service.create_session(
            authenticated_user_id, "api"
        )

        # Scrub the extracted text using sensitivity classifier
        classifier_service = get_classifier_service()
        redaction_result = classifier_service.redact_sensitive_info(
            upload_response.extracted_text
        )
        scrubbed_text = redaction_result.redacted_text

        # Store redaction record for potential de-scrubbing using existing method
        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=authenticated_user_id,
            action="file_redacted",
            details={
                "filename": file.filename,
                "file_type": upload_response.file_type,
                "original_text": upload_response.extracted_text,
                "redacted_text": scrubbed_text,
                "detections": redaction_result.detections,
                "total_redacted": redaction_result.total_redacted,
                "endpoint": "/scrub-file",
            },
        )

        # Extract matches information from redaction result
        matches = {
            detection_type: [
                d["original"]
                for d in redaction_result.detections
                if d["type"] == detection_type
            ]
            for detection_type in redaction_result.detection_summary.keys()
        }

        # Calculate metrics
        original_length = len(upload_response.extracted_text)
        scrubbed_length = len(scrubbed_text)
        matches_found = redaction_result.total_redacted

        reduction_percentage = (
            round((1 - scrubbed_length / original_length) * 100, 2)
            if original_length > 0
            else 0
        )

        # Log the more detailed interaction for audit purposes
        mongodb_service.log_interaction(
            session_id,
            authenticated_user_id,
            "api_file_scrubbing",
            {
                "filename": file.filename,
                "file_type": upload_response.file_type,
                "file_size": upload_response.file_size,
                "input_length": original_length,
                "output_length": scrubbed_length,
                "matches_found": matches_found,
                "reduction_percentage": reduction_percentage,
                "detection_summary": redaction_result.detection_summary,
                "endpoint": "/scrub-file",
            },
        )

        return FileScrubResponse(
            filename=file.filename or "unknown_file",
            file_type=upload_response.file_type,
            file_size=upload_response.file_size,
            original_text=upload_response.extracted_text,
            scrubbed_text=scrubbed_text,
            original_length=original_length,
            scrubbed_length=scrubbed_length,
            matches_found=matches_found,
            reduction_percentage=reduction_percentage,
            matches_detected=matches,
            processed_at=datetime.now(),
            session_id=session_id,
            download_url=None,  # Will be implemented with file download functionality
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/scrub-file-download")
async def scrub_file_download(
    file: UploadFile = File(...),
    request: FileScrubRequest = FileScrubRequest(),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload file, scrub sensitive information, and download the scrubbed file in original format.
    Supports PDF, DOCX, TXT, HTML, CSV formats.
    """
    try:
        import io

        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # First scrub the file
        scrub_response = await scrub_file(file, request, current_user)

        # Get file extension for processing
        file_extension = (
            file.filename.lower().split(".")[-1] if "." in file.filename else "txt"
        )
        scrubbed_filename = f"scrubbed_{file.filename}"

        # Create scrubbed file in appropriate format
        if file_extension == "txt":
            writer = TXTWriter()
            file_bytes = writer.create_txt_from_text(scrub_response.scrubbed_text)
            content_type = "text/plain"

        elif file_extension == "docx":
            writer = DOCXWriter()
            file_bytes = writer.create_docx_from_text(scrub_response.scrubbed_text)
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        elif file_extension == "html":
            writer = HTMLWriter()
            file_bytes = writer.create_html_from_text(
                text_content=scrub_response.scrubbed_text
            )
            content_type = "text/html"

        elif file_extension == "csv":
            # For CSV, just save as text with original structure preserved
            file_bytes = scrub_response.scrubbed_text.encode("utf-8")
            content_type = "text/csv"

        else:
            # Default to text file
            file_bytes = scrub_response.scrubbed_text.encode("utf-8")
            content_type = "text/plain"

        # Return file as download
        headers = {
            "Content-Disposition": f'attachment; filename="{scrubbed_filename}"',
            "Content-Type": content_type,
        }

        return StreamingResponse(
            io.BytesIO(file_bytes), headers=headers, media_type=content_type
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating scrubbed file: {str(e)}"
        )


# Sensitivity Classification Endpoints
@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Classify text sensitivity level (C1-C4).
    Requires user authentication.
    """
    try:
        classifier_service = get_classifier_service()

        if not classifier_service.is_model_available():
            raise HTTPException(
                status_code=503,
                detail="Classification model not available. Please train the model first.",
            )

        # Classify the text
        result = classifier_service.classify_text(request.text)

        # Log the classification
        mongodb_service = MongoDBService()
        session_id = str(uuid.uuid4())

        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="text_classified",
            details={
                "text_length": len(request.text),
                "prediction": result.prediction,
                "confidence": result.confidence,
                "endpoint": "/classify",
            },
        )

        return ClassifyResponse(
            prediction=result.prediction,
            probabilities=result.probabilities,
            confidence=result.confidence,
            explanation=result.explanation,
            success=True,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying text: {str(e)}")


@app.post("/redact", response_model=RedactResponse)
async def redact_sensitive_data(
    request: RedactRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Redact sensitive information from text.
    Requires user authentication.
    """
    try:
        classifier_service = get_classifier_service()

        # Redact sensitive information
        result = classifier_service.redact_sensitive_info(request.text)

        # Log the redaction
        mongodb_service = MongoDBService()
        session_id = str(uuid.uuid4())

        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="text_redacted",
            details={
                "text_length": len(request.text),
                "total_redacted": result.total_redacted,
                "detection_types": list(result.detection_summary.keys()),
                "detections": result.detections,
                "endpoint": "/redact",
            },
        )

        # Convert detections to Pydantic models
        detections = [
            Detection(
                type=d["type"], original=d["original"], placeholder=d["placeholder"]
            )
            for d in result.detections
        ]

        return RedactResponse(
            session_id=session_id,
            original_text=result.original_text,
            redacted_text=result.redacted_text,
            detections=detections,
            total_redacted=result.total_redacted,
            detection_summary=result.detection_summary,
            success=True,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error redacting text: {str(e)}")


@app.post("/train-model", response_model=TrainModelResponse)
async def train_classification_model(
    request: TrainModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_admin),
):
    """
    Train the sensitivity classification model.
    Requires admin authentication.
    """
    try:
        import subprocess
        import sys

        # Set environment variables for training
        env = os.environ.copy()
        env["MODEL"] = request.model_type
        env["DATA_PATH"] = request.data_file
        env["TEST_SIZE"] = str(request.test_size)
        env["NGRAM_MAX"] = str(request.ngram_max)
        env["SEED"] = str(request.random_seed)

        # Run training script
        result = subprocess.run(
            [sys.executable, "ml_setup.py", "--train"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout
        )

        # Log the training attempt
        mongodb_service = MongoDBService()
        session_id = str(uuid.uuid4())

        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="model_training",
            details={
                "model_type": request.model_type,
                "data_file": request.data_file,
                "success": result.returncode == 0,
                "endpoint": "/train-model",
            },
        )

        if result.returncode == 0:
            # Parse metrics from output if available
            metrics = None
            try:
                # Try to load metrics file if created
                classifier_service = get_classifier_service()
                classifier_service._load_model()  # Reload model
                model_metrics = classifier_service.get_model_metrics()
                if model_metrics:
                    metrics = {
                        "accuracy": model_metrics.accuracy,
                        "precision": model_metrics.precision,
                        "recall": model_metrics.recall,
                        "f1_score": model_metrics.f1_score,
                    }
            except Exception:
                pass

            return TrainModelResponse(
                success=True, message="Model trained successfully", metrics=metrics
            )
        else:
            return TrainModelResponse(
                success=False,
                message="Model training failed",
                error=result.stderr or result.stdout,
            )

    except subprocess.TimeoutExpired:
        return TrainModelResponse(
            success=False,
            message="Training timeout exceeded (5 minutes)",
            error="Training took too long",
        )
    except FileNotFoundError:
        return TrainModelResponse(
            success=False,
            message="Training script not found",
            error="ml_setup.py not found in current directory",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/model-metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get model performance metrics.
    Requires user authentication.
    """
    try:
        classifier_service = get_classifier_service()
        model_metrics = classifier_service.get_model_metrics()

        if model_metrics is None:
            return ModelMetricsResponse(
                accuracy=0.0,
                precision={},
                recall={},
                f1_score={},
                support={},
                confusion_matrix=[],
                available=False,
            )

        return ModelMetricsResponse(
            accuracy=model_metrics.accuracy,
            precision=model_metrics.precision,
            recall=model_metrics.recall,
            f1_score=model_metrics.f1_score,
            support=model_metrics.support,
            confusion_matrix=model_metrics.confusion_matrix,
            available=True,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model metrics: {str(e)}"
        )


@app.post("/de-scrub", response_model=DescrubResponse)
async def de_scrub_text(
    request: DescrubRequest, current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """
    De-scrub (restore) redacted text using stored redaction records.
    Takes scrubbed text with placeholders and replaces them with original values.
    Admin only endpoint for security.
    """
    try:
        # Get the redaction record from MongoDB to get the detections
        redaction_record = mongodb_service.get_redaction_record_by_session(
            request.session_id
        )

        if not redaction_record:
            raise HTTPException(
                status_code=404,
                detail=f"No redaction record found for session {request.session_id}",
            )

        # Get the detections from the stored record
        details = redaction_record.get("details", {})

        # Handle case where details might be stored as a JSON string
        if isinstance(details, str):
            import json

            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid details format in redaction record"
                )

        detections = details.get("detections", [])

        if not detections:
            raise HTTPException(
                status_code=400, detail="No detections found for de-scrubbing"
            )

        # Perform de-scrubbing: replace each placeholder with its original value
        restored_text = request.scrubbed_text
        detections_restored = 0

        for detection in detections:
            placeholder = detection.get("placeholder", "")
            original = detection.get("original", "")

            if placeholder and original and placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, original)
                detections_restored += 1

        # Store the de-scrubbing action in the database
        # Use the provided session_id to maintain workflow consistency
        session_id = request.session_id
        mongodb_service.log_interaction(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="text_de_scrubbed",
            details={
                "original_session_id": request.session_id,
                "scrubbed_text": request.scrubbed_text,
                "restored_text": restored_text,
                "detections_restored": detections_restored,
                "endpoint": "/de-scrub",
            },
        )

        return DescrubResponse(
            success=True,
            session_id=session_id,
            original_text=redaction_record.get("details", {}).get("original_text", ""),
            redacted_text=request.scrubbed_text,
            restored_text=restored_text,
            detections_restored=detections_restored,
            message=f"Successfully restored {detections_restored} redacted items",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during de-scrubbing: {str(e)}"
        )


@app.get("/redaction-records", response_model=RedactionRecordsResponse)
async def get_redaction_records(
    user_id: Optional[str] = None,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_admin),
):
    """
    Get list of redaction records from the database.
    Admin only endpoint.
    """
    try:
        # Limit the maximum number of records to prevent large responses
        limit = min(limit, 100)

        records = mongodb_service.get_redaction_records(user_id=user_id, limit=limit)

        return RedactionRecordsResponse(
            success=True, records=records, total_count=len(records)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving redaction records: {str(e)}"
        )


@app.get("/classification-categories", response_model=Dict[str, str])
async def get_classification_categories():
    """
    Get information about classification categories.
    Public endpoint.
    """
    classifier_service = get_classifier_service()
    return classifier_service.get_category_info()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
