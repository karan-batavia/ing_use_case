from typing import Union, Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import os
import uuid
import asyncio
from datetime import datetime, timedelta
from src.mongodb_service import MongoDBService
from src.prompt_scrubber import PromptScrubber
from src.dependencies import get_current_user, get_current_admin
from src.ollama_service import get_ollama_service
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
prompt_scrubber = PromptScrubber()


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


class FileDownloadInfo(BaseModel):
    download_url: str
    filename: str
    file_type: str
    expires_at: datetime


class PredictionRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    model_name: Optional[str] = "llama3.2:1b"  # Optimized smaller model
    max_tokens: Optional[int] = 150  # Reduced for faster responses
    temperature: Optional[float] = 0.3  # Lower for faster, more focused responses


class PredictionResponse(BaseModel):
    prediction: str
    model_used: str
    success: bool
    session_id: str
    processed_at: datetime


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


@app.post("/scrub", response_model=TextScrubResponse)
def scrub_text(
    request: TextScrubRequest, current_user: dict = Depends(get_current_user)
):
    """Scrub sensitive information from text (Authenticated users only)"""
    try:
        # Extract user_id from the authenticated user data
        authenticated_user_id = current_user.get(
            "user_id", current_user.get("_id", "unknown")
        )

        # Create session for authenticated user
        session_id = mongodb_service.create_session(authenticated_user_id, "api")

        # Scrub the text
        matches = prompt_scrubber.scrub(request.text)
        scrubbed_text = prompt_scrubber.scrub_prompt(request.text)

        # Calculate metrics
        original_length = len(request.text)
        scrubbed_length = len(scrubbed_text)
        matches_found = (
            sum(len(found_values) for found_values in matches.values())
            if matches
            else 0
        )
        reduction_percentage = (
            round((1 - scrubbed_length / original_length) * 100, 2)
            if original_length > 0
            else 0
        )

        # Log the interaction
        mongodb_service.log_interaction(
            session_id,
            authenticated_user_id,
            "api_text_scrubbing",
            {
                "input_length": original_length,
                "output_length": scrubbed_length,
                "matches_found": matches_found,
                "reduction_percentage": reduction_percentage,
                "endpoint": "/scrub",
            },
        )

        return TextScrubResponse(
            scrubbed_text=scrubbed_text,
            original_length=original_length,
            scrubbed_length=scrubbed_length,
            matches_found=matches_found,
            reduction_percentage=reduction_percentage,
            processed_at=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def generate_prediction(
    request: PredictionRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate a prediction using Ollama AI based on prompt and context.
    Requires user authentication.
    """
    try:
        # Get Ollama service instance
        ollama_service = get_ollama_service()

        # Validate that Ollama service is available
        if not ollama_service.validate_connection():
            raise HTTPException(
                status_code=503,
                detail="Ollama AI service is not available. Please check that Ollama is running.",
            )

        # Generate session ID for this request
        session_id = str(uuid.uuid4())

        # Generate prediction using Ollama with timeout
        try:
            prediction_text = await asyncio.wait_for(
                ollama_service.generate_prediction(
                    prompt=request.prompt,
                    context=request.context,
                    model_name=request.model_name,
                ),
                timeout=300.0,  # 5 minutes timeout
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
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, HTML, PNG, JPG",
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
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    """Upload file, extract text, and scrub sensitive information (Authenticated users only)"""
    try:
        # First extract text from file (reuse upload logic)
        upload_response = await upload_file(file, current_user)

        # Extract user_id from the authenticated user data
        authenticated_user_id = current_user.get(
            "user_id", current_user.get("_id", "unknown")
        )

        # Create session for authenticated user
        session_id = mongodb_service.create_session(authenticated_user_id, "api")

        # Scrub the extracted text
        matches = prompt_scrubber.scrub(upload_response.extracted_text)
        scrubbed_text = prompt_scrubber.scrub_prompt(upload_response.extracted_text)

        # Calculate metrics
        original_length = len(upload_response.extracted_text)
        scrubbed_length = len(scrubbed_text)
        matches_found = sum(len(values) for values in matches.values())

        reduction_percentage = (
            round((1 - scrubbed_length / original_length) * 100, 2)
            if original_length > 0
            else 0
        )

        # Log the interaction
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
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
