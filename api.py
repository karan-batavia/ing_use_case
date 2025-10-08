from typing import Union, Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
import uvicorn
import os
import uuid
import asyncio
import subprocess
import hashlib
import json
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


# Enhanced Audit Context Models
class AuditContext(BaseModel):
    """Enhanced audit context for banking compliance"""

    device_id: Optional[str] = None
    browser_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    session_fingerprint: Optional[str] = None
    client_timestamp: Optional[datetime] = None
    risk_level: Optional[str] = "medium"  # low, medium, high, critical


class CustomerDataOperation(BaseModel):
    """Classification of customer data operations for compliance"""

    operation_type: str  # read, write, update, delete, export, scrub, de-scrub
    data_categories: List[str]  # pii, financial, transaction, sensitive, etc.
    risk_classification: str  # C1, C2, C3, C4
    justification: Optional[str] = None
    requires_approval: bool = False


def extract_audit_context(request: Request) -> AuditContext:
    """Extract audit context from HTTP request for compliance tracking"""
    try:
        # Extract client IP (handle proxy headers)
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Extract user agent
        user_agent = request.headers.get("user-agent", "unknown")

        # Generate device fingerprint from headers and IP
        device_components = [
            client_ip,
            user_agent,
            request.headers.get("accept", ""),
            request.headers.get("accept-language", ""),
            request.headers.get("accept-encoding", ""),
        ]
        device_fingerprint = hashlib.sha256(
            "|".join(device_components).encode()
        ).hexdigest()[:16]

        # Extract browser/session info
        browser_id = request.headers.get("x-browser-id", device_fingerprint)
        session_fingerprint = request.headers.get("x-session-id", str(uuid.uuid4()))

        return AuditContext(
            device_id=device_fingerprint,
            browser_id=browser_id,
            user_agent=user_agent,
            ip_address=client_ip,
            session_fingerprint=session_fingerprint,
            client_timestamp=datetime.now(),
            risk_level="medium",  # Default risk level
        )
    except Exception as e:
        # Fallback audit context if extraction fails
        return AuditContext(
            device_id="unknown",
            browser_id="unknown",
            user_agent="unknown",
            ip_address="unknown",
            session_fingerprint=str(uuid.uuid4()),
            client_timestamp=datetime.now(),
            risk_level="high",  # Higher risk for unknown contexts
        )


def get_request_context(request: Request) -> AuditContext:
    """Dependency to extract audit context from request"""
    return extract_audit_context(request)


def detect_customer_data_operation(
    action: str, details: Dict[str, Any]
) -> CustomerDataOperation:
    """Detect and classify customer data operations for banking compliance"""

    # Get text content for analysis
    text_content = details.get("original_text", "")

    # Use existing classifier service for sensitivity classification
    try:
        classifier_service = get_classifier_service()
        if classifier_service.is_model_available() and text_content:
            classification_result = classifier_service.classify_text(text_content)
            risk_classification = classification_result.prediction
        else:
            # Fallback to pattern-based detection if model not available
            risk_classification = _fallback_classification(text_content)
    except Exception:
        # Fallback in case of classification errors
        risk_classification = _fallback_classification(text_content)

    # Extract data categories from redaction results if available
    data_categories = []
    detections = details.get("detections", [])
    if detections:
        # Use actual detection types from redaction
        detected_types = {d.get("type", "") for d in detections}
        # Map detection types to data categories
        for detection_type in detected_types:
            if detection_type in ["EMAIL", "PHONE_EU", "SSN_LIKE", "NATIONAL_ID"]:
                data_categories.append("pii")
            elif detection_type in ["IBAN", "ACCOUNT_NUM", "AMOUNT"]:
                data_categories.append("financial")
            elif detection_type in ["BIOMETRIC"]:
                data_categories.append("authentication")
    else:
        # Fallback to text analysis
        data_categories = _extract_data_categories_from_text(text_content)

    # Determine if approval is required based on classification and operation
    requires_approval = action in [
        "text_de_scrubbed",
        "file_de_scrubbed",
    ] or risk_classification in ["C3", "C4"]

    return CustomerDataOperation(
        operation_type=action,
        data_categories=data_categories,
        risk_classification=risk_classification,
        requires_approval=requires_approval,
    )


def _fallback_classification(text_content: str) -> str:
    """Fallback classification when ML model is not available"""
    text_lower = text_content.lower()

    # Define sensitive patterns for fallback
    if any(
        pattern in text_lower
        for pattern in [
            "account",
            "balance",
            "transaction",
            "payment",
            "loan",
            "credit",
            "password",
            "token",
            "credential",
        ]
    ):
        return "C4"  # Highest sensitivity
    elif any(
        pattern in text_lower
        for pattern in ["name", "address", "phone", "email", "ssn", "id"]
    ):
        return "C3"
    elif any(
        pattern in text_lower
        for pattern in ["salary", "income", "medical", "legal", "confidential"]
    ):
        return "C2"
    else:
        return "C1"  # Default


def _extract_data_categories_from_text(text_content: str) -> List[str]:
    """Extract data categories from text when redaction data is not available"""
    text_lower = text_content.lower()
    data_categories = []

    if any(
        pattern in text_lower
        for pattern in [
            "account",
            "balance",
            "transaction",
            "payment",
            "loan",
            "credit",
        ]
    ):
        data_categories.append("financial")
    if any(
        pattern in text_lower
        for pattern in ["name", "address", "phone", "email", "ssn", "id", "passport"]
    ):
        data_categories.append("pii")
    if any(
        pattern in text_lower
        for pattern in ["password", "token", "key", "credential", "auth"]
    ):
        data_categories.append("authentication")
    if any(
        pattern in text_lower
        for pattern in ["salary", "income", "medical", "legal", "confidential"]
    ):
        data_categories.append("sensitive")

    return data_categories


def _redact_with_custom_patterns(text: str, pattern_names: List[str]):
    """
    Custom redaction function that only applies specific patterns based on sensitivity level.
    Returns a result object similar to classifier_service.redact_sensitive_info()
    """
    import re
    from dataclasses import dataclass

    # Define patterns (should match what's in sensitivity_classifier.py)
    FALLBACK_PATTERNS = {
        "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "PHONE_EU": r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}",
        "SSN_LIKE": r"\b\d{6}[- ]?\d{2,4}[\.]?\d{0,2}\b",
        "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        "ACCOUNT_NUM": r"\b(?:acct|account)\s*\d{3,}\b",
        "AMOUNT": r"(?:USD|EUR|GBP|€|\$|£)\s?\d{1,3}(?:[, \u00A0]\d{3})*(?:\.\d{2})?",
        "DOB": r"\b\d{4}-\d{2}-\d{2}\b",
        "NATIONAL_ID": r"\bID[:\s-]?[A-Z0-9]{6,}\b",
        "BIOMETRIC": r"\b(FaceID|fingerprint|iris|biometric)\b",
    }

    PLACEHOLDERS = {
        "EMAIL": "[EMAIL]",
        "PHONE_EU": "[PHONE]",
        "SSN_LIKE": "[SSN]",
        "IBAN": "[IBAN]",
        "ACCOUNT_NUM": "[ACCOUNT_NUMBER]",
        "AMOUNT": "[AMOUNT]",
        "DOB": "[DATE_OF_BIRTH]",
        "NATIONAL_ID": "[NATIONAL_ID]",
        "BIOMETRIC": "[BIOMETRIC_DATA]",
    }

    @dataclass
    class CustomRedactionResult:
        original_text: str
        redacted_text: str
        detections: List[Dict[str, Any]]
        total_redacted: int
        detection_summary: Dict[str, int]

    redacted_text = text
    detections = []

    # Track all matches with their positions (in reverse order to maintain indices)
    matches = []

    for pattern_name in pattern_names:
        if pattern_name in FALLBACK_PATTERNS:
            pattern = re.compile(FALLBACK_PATTERNS[pattern_name], re.IGNORECASE)
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "type": pattern_name,
                        "placeholder": PLACEHOLDERS.get(
                            pattern_name, f"[{pattern_name}]"
                        ),
                    }
                )

    # Sort by position (reverse order to maintain string indices during replacement)
    matches.sort(key=lambda x: x["start"], reverse=True)

    # Replace matches with placeholders
    for match in matches:
        redacted_text = (
            redacted_text[: match["start"]]
            + match["placeholder"]
            + redacted_text[match["end"] :]
        )
        detections.append(
            {
                "type": match["type"],
                "original": match["text"],
                "placeholder": match["placeholder"],
            }
        )

    # Reverse detections to show in original order
    detections.reverse()

    # Create detection summary
    detection_summary = {}
    for detection in detections:
        det_type = detection["type"]
        detection_summary[det_type] = detection_summary.get(det_type, 0) + 1

    return CustomRedactionResult(
        original_text=text,
        redacted_text=redacted_text,
        detections=detections,
        total_redacted=len(detections),
        detection_summary=detection_summary,
    )


def enhanced_audit_log(
    session_id: str,
    user_id: str,
    action: str,
    details: Dict[str, Any],
    audit_context: AuditContext,
    endpoint: str,
) -> None:
    """Enhanced audit logging with banking compliance features"""

    # Detect customer data operation
    customer_operation = detect_customer_data_operation(action, details)

    # Create comprehensive audit record
    enhanced_details = {
        **details,
        "audit_context": {
            "device_id": audit_context.device_id,
            "browser_id": audit_context.browser_id,
            "user_agent": audit_context.user_agent,
            "ip_address": audit_context.ip_address,
            "session_fingerprint": audit_context.session_fingerprint,
            "client_timestamp": (
                audit_context.client_timestamp.isoformat()
                if audit_context.client_timestamp
                else None
            ),
            "risk_level": audit_context.risk_level,
        },
        "customer_data_operation": {
            "operation_type": customer_operation.operation_type,
            "data_categories": customer_operation.data_categories,
            "risk_classification": customer_operation.risk_classification,
            "requires_approval": customer_operation.requires_approval,
        },
        "compliance_metadata": {
            "endpoint": endpoint,
            "server_timestamp": datetime.now().isoformat(),
            "audit_version": "2.0",
            "banking_compliance": True,
        },
    }

    # Log using existing MongoDB service with enhanced details
    mongodb_service.log_interaction(
        session_id=session_id, user_id=user_id, action=action, details=enhanced_details
    )


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
    sensitivity_level: Optional[str] = (
        None  # C1, C2, C3, C4 - if None, uses full redaction
    )

    @field_validator("sensitivity_level")
    @classmethod
    def validate_sensitivity_level(cls, v):
        if v is not None and v not in ["C1", "C2", "C3", "C4"]:
            raise ValueError("sensitivity_level must be one of: C1, C2, C3, C4")
        return v


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
    sensitivity_level: Optional[str] = (
        None  # C1, C2, C3, C4 - if None, uses full redaction
    )

    @field_validator("sensitivity_level")
    @classmethod
    def validate_sensitivity_level(cls, v):
        if v is not None and v not in ["C1", "C2", "C3", "C4"]:
            raise ValueError("sensitivity_level must be one of: C1, C2, C3, C4")
        return v


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
    request: PredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_context: AuditContext = Depends(get_request_context),
):
    """
    Generate a prediction using Gemini AI based on prompt and context.
    Requires user authentication.
    """
    try:
        # Audit context is provided by dependency injection

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

        # Enhanced audit logging
        enhanced_audit_log(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="prediction_generated",
            details={
                "prompt_length": len(request.prompt),
                "context_provided": bool(request.context),
                "model_used": request.model_name or "gemini-pro",
                "original_text": request.prompt,  # For customer data detection
                "prediction_length": len(prediction_text),
            },
            audit_context=audit_context,
            endpoint="/predict",
        )

        return PredictionResponse(
            prediction=prediction_text,
            model_used=request.model_name or "gemini-pro",
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


class EnhancedTokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    session_id: str
    expires_in: Optional[int] = None


@app.post("/auth/login", response_model=EnhancedTokenResponse)
def login(user_credentials: UserLogin, request: Request):
    """Login user and return JWT token"""
    try:
        # Extract audit context for login tracking
        audit_context = extract_audit_context(request)

        if not mongodb_service.is_connected():
            raise HTTPException(status_code=503, detail="Database not available")

        # Get user from database
        user_data = mongodb_service.get_user_by_email(user_credentials.email)
        if not user_data:
            # Log failed login attempt for security
            enhanced_audit_log(
                session_id=str(uuid.uuid4()),
                user_id=user_credentials.email,  # Use email for failed attempts
                action="login_failed",
                details={
                    "reason": "user_not_found",
                    "attempted_email": user_credentials.email,
                    "original_text": "",  # No sensitive data in login failure
                },
                audit_context=audit_context,
                endpoint="/auth/login",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_in_db = UserInDB(**user_data)

        # Verify password
        if not verify_password(user_credentials.password, user_in_db.hashed_password):
            # Log failed login attempt for security
            enhanced_audit_log(
                session_id=str(uuid.uuid4()),
                user_id=user_in_db.email,
                action="login_failed",
                details={
                    "reason": "invalid_password",
                    "attempted_email": user_credentials.email,
                    "original_text": "",  # No sensitive data in login failure
                },
                audit_context=audit_context,
                endpoint="/auth/login",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is active
        if not user_in_db.is_active:
            # Log failed login attempt for inactive user
            enhanced_audit_log(
                session_id=str(uuid.uuid4()),
                user_id=user_in_db.email,
                action="login_failed",
                details={
                    "reason": "inactive_user",
                    "attempted_email": user_credentials.email,
                    "original_text": "",  # No sensitive data in login failure
                },
                audit_context=audit_context,
                endpoint="/auth/login",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
            )

        # Update last login
        mongodb_service.update_last_login(user_credentials.email)

        # Ensure we have a user_id (fallback in case migration failed)
        user_id = user_data.get("user_id", user_data.get("_id", "unknown"))

        # Create session for login tracking
        login_session_id = mongodb_service.create_session(user_id, "authentication")

        # Create access token with user_id and session_id in payload
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user_in_db.email,
                "user_id": user_id,
                "session_id": login_session_id,
            },
            expires_delta=access_token_expires,
        )

        # Log successful login for audit trail
        enhanced_audit_log(
            session_id=login_session_id,
            user_id=user_id,
            action="login_successful",
            details={
                "email": user_credentials.email,
                "is_admin": user_in_db.is_admin,
                "token_expires": access_token_expires.total_seconds(),
                "original_text": "",  # No sensitive data in successful login
            },
            audit_context=audit_context,
            endpoint="/auth/login",
        )

        return EnhancedTokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            session_id=login_session_id,
            expires_in=int(access_token_expires.total_seconds()),
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log system error during login
        enhanced_audit_log(
            session_id=str(uuid.uuid4()),
            user_id=user_credentials.email,
            action="login_error",
            details={
                "error": str(e),
                "attempted_email": user_credentials.email,
                "original_text": "",
            },
            audit_context=extract_audit_context(request),
            endpoint="/auth/login",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login system error",
        )


@app.post("/auth/logout")
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_context: AuditContext = Depends(get_request_context),
):
    """Logout user and invalidate session"""
    try:
        user_id = current_user.get("user_id", "unknown")
        session_id = current_user.get("session_id", str(uuid.uuid4()))

        # Log logout for audit trail
        enhanced_audit_log(
            session_id=session_id,
            user_id=user_id,
            action="logout_successful",
            details={
                "email": current_user.get("email", "unknown"),
                "session_closed": True,
                "original_text": "",  # No sensitive data in logout
            },
            audit_context=audit_context,
            endpoint="/auth/logout",
        )

        # Note: In a production system, you would:
        # 1. Add the token to a blacklist/revocation list
        # 2. Clear any server-side session data
        # 3. Notify other services about the logout

        return {
            "message": "Successfully logged out",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Log logout error
        enhanced_audit_log(
            session_id=str(uuid.uuid4()),
            user_id=current_user.get("user_id", "unknown"),
            action="logout_error",
            details={
                "error": str(e),
                "original_text": "",
            },
            audit_context=audit_context,
            endpoint="/auth/logout",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout system error",
        )


class SessionStatusResponse(BaseModel):
    active: bool
    user_id: str
    session_id: str
    expires_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None


@app.get("/auth/session-status", response_model=SessionStatusResponse)
async def get_session_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_context: AuditContext = Depends(get_request_context),
):
    """Get current session status"""
    try:
        user_id = current_user.get("user_id", "unknown")
        session_id = current_user.get("session_id", str(uuid.uuid4()))

        # Log session status check (low priority audit event)
        enhanced_audit_log(
            session_id=session_id,
            user_id=user_id,
            action="session_status_checked",
            details={
                "email": current_user.get("email", "unknown"),
                "original_text": "",  # No sensitive data
            },
            audit_context=audit_context,
            endpoint="/auth/session-status",
        )

        return SessionStatusResponse(
            active=True,
            user_id=user_id,
            session_id=session_id,
            expires_at=None,  # Would need to extract from JWT
            last_activity=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session status error",
        )


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
    current_user: dict = Depends(get_current_user),
    request: FileScrubRequest = FileScrubRequest(),
    audit_context: AuditContext = Depends(get_request_context),
):
    """Upload file, extract text, and scrub sensitive information (Authenticated users only)"""
    try:
        # Audit context is provided by dependency injection

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

        # Scrub the extracted text using sensitivity classifier with level-based filtering
        classifier_service = get_classifier_service()

        # Define sensitivity level pattern mappings
        sensitivity_patterns = {
            "C1": [],  # No redaction for public information
            "C2": ["EMAIL", "PHONE_EU"],  # Basic PII redaction
            "C3": [
                "EMAIL",
                "PHONE_EU",
                "IBAN",
                "ACCOUNT_NUM",
                "AMOUNT",
            ],  # Add financial
            "C4": None,  # Full redaction (all patterns)
        }

        # Determine which patterns to use based on sensitivity level
        if (
            request.sensitivity_level
            and request.sensitivity_level in sensitivity_patterns
        ):
            if request.sensitivity_level == "C1":
                # No redaction for C1 (public)
                scrubbed_text = upload_response.extracted_text
                redaction_result = None
                detections = []
                total_redacted = 0
                detection_summary = {}
            else:
                # Custom redaction based on sensitivity level
                patterns_to_use = sensitivity_patterns[request.sensitivity_level]
                if patterns_to_use is None:
                    # C4 - use full redaction
                    redaction_result = classifier_service.redact_sensitive_info(
                        upload_response.extracted_text
                    )
                else:
                    # Custom redaction for C2, C3
                    redaction_result = _redact_with_custom_patterns(
                        upload_response.extracted_text, patterns_to_use
                    )

                scrubbed_text = redaction_result.redacted_text
                detections = redaction_result.detections
                total_redacted = redaction_result.total_redacted
                detection_summary = redaction_result.detection_summary
        else:
            # Default: full redaction if no sensitivity level specified
            redaction_result = classifier_service.redact_sensitive_info(
                upload_response.extracted_text
            )
            scrubbed_text = redaction_result.redacted_text
            detections = redaction_result.detections
            total_redacted = redaction_result.total_redacted
            detection_summary = redaction_result.detection_summary

        # Enhanced audit logging for file redaction
        enhanced_audit_log(
            session_id=session_id,
            user_id=authenticated_user_id,
            action="file_redacted",
            details={
                "filename": file.filename,
                "file_type": upload_response.file_type,
                "original_text": upload_response.extracted_text,
                "redacted_text": scrubbed_text,
                "detections": detections,
                "total_redacted": total_redacted,
                "file_size": upload_response.file_size,
                "sensitivity_level": request.sensitivity_level,
            },
            audit_context=audit_context,
            endpoint="/scrub-file",
        )

        # Extract matches information from detections
        matches = {
            detection_type: [
                d["original"] for d in detections if d["type"] == detection_type
            ]
            for detection_type in detection_summary.keys()
        }

        # Calculate metrics
        original_length = len(upload_response.extracted_text)
        scrubbed_length = len(scrubbed_text)
        matches_found = total_redacted

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
                "detection_summary": detection_summary,
                "sensitivity_level": request.sensitivity_level,
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
    audit_context: AuditContext = Depends(get_request_context),
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

        # First scrub the file (pass audit_context via dependency)
        scrub_response = await scrub_file(file, current_user, request, audit_context)

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
    request: RedactRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_context: AuditContext = Depends(get_request_context),
):
    """
    Redact sensitive information from text with optional sensitivity level filtering.
    Sensitivity levels:
    - C1: Public information (no redaction)
    - C2: Internal use (basic redaction - emails, phones)
    - C3: Confidential (moderate redaction - adds financial data)
    - C4: Highly sensitive (full redaction - all patterns)
    - None: Full redaction (default)
    Requires user authentication.
    """
    try:
        # Audit context is provided by dependency injection

        classifier_service = get_classifier_service()

        # Define sensitivity level pattern mappings
        sensitivity_patterns = {
            "C1": [],  # No redaction for public information
            "C2": ["EMAIL", "PHONE_EU"],  # Basic PII redaction
            "C3": [
                "EMAIL",
                "PHONE_EU",
                "IBAN",
                "ACCOUNT_NUM",
                "AMOUNT",
            ],  # Add financial
            "C4": None,  # Full redaction (all patterns)
        }

        # Determine which patterns to use based on sensitivity level
        if (
            request.sensitivity_level
            and request.sensitivity_level in sensitivity_patterns
        ):
            if request.sensitivity_level == "C1":
                # No redaction for C1 (public)
                result_text = request.text
                detections = []
                total_redacted = 0
                detection_summary = {}
            else:
                # Custom redaction based on sensitivity level
                patterns_to_use = sensitivity_patterns[request.sensitivity_level]
                if patterns_to_use is None:
                    # C4 - use full redaction
                    result = classifier_service.redact_sensitive_info(request.text)
                else:
                    # Custom redaction for C2, C3
                    result = _redact_with_custom_patterns(request.text, patterns_to_use)

                result_text = result.redacted_text
                detections = result.detections
                total_redacted = result.total_redacted
                detection_summary = result.detection_summary
        else:
            # Default: full redaction if no sensitivity level specified
            result = classifier_service.redact_sensitive_info(request.text)
            result_text = result.redacted_text
            detections = result.detections
            total_redacted = result.total_redacted
            detection_summary = result.detection_summary

        # Generate session ID for this redaction
        session_id = str(uuid.uuid4())

        # Enhanced audit logging for text redaction
        enhanced_audit_log(
            session_id=session_id,
            user_id=current_user["user_id"],
            action="text_redacted",
            details={
                "text_length": len(request.text),
                "sensitivity_level": request.sensitivity_level,
                "total_redacted": total_redacted,
                "detection_types": list(detection_summary.keys()),
                "detections": detections,
                "original_text": request.text,
                "redacted_text": result_text,
            },
            audit_context=audit_context,
            endpoint="/redact",
        )

        # Convert detections to Pydantic models
        detection_models = [
            Detection(
                type=d["type"], original=d["original"], placeholder=d["placeholder"]
            )
            for d in detections
        ]

        return RedactResponse(
            session_id=session_id,
            original_text=request.text,
            redacted_text=result_text,
            detections=detection_models,
            total_redacted=total_redacted,
            detection_summary=detection_summary,
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
    request: DescrubRequest,
    current_user: Dict[str, Any] = Depends(get_current_admin),
    audit_context: AuditContext = Depends(get_request_context),
):
    """
    De-scrub (restore) redacted text using stored redaction records.
    Takes scrubbed text with placeholders and replaces them with original values.
    Admin only endpoint for security.
    """
    try:
        # Audit context is provided by dependency injection

        # Get the redaction record from MongoDB to get the detections
        redaction_record = mongodb_service.get_redaction_record_by_session(
            request.session_id
        )

        if not redaction_record:
            # Log failed de-scrub attempt for security
            enhanced_audit_log(
                session_id=request.session_id,
                user_id=current_user["user_id"],
                action="de_scrub_failed",
                details={
                    "reason": "redaction_record_not_found",
                    "session_id": request.session_id,
                    "scrubbed_text": request.scrubbed_text,
                },
                audit_context=audit_context,
                endpoint="/de-scrub",
            )
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

        # Enhanced audit logging for de-scrubbing (critical operation)
        enhanced_audit_log(
            session_id=request.session_id,
            user_id=current_user["user_id"],
            action="text_de_scrubbed",
            details={
                "original_session_id": request.session_id,
                "scrubbed_text": request.scrubbed_text,
                "restored_text": restored_text,
                "detections_restored": detections_restored,
                "original_text": redaction_record.get("details", {}).get(
                    "original_text", ""
                ),
                "admin_action": True,
                "critical_operation": True,
            },
            audit_context=audit_context,
            endpoint="/de-scrub",
        )

        return DescrubResponse(
            success=True,
            session_id=request.session_id,
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


# Enhanced Audit Endpoints
@app.get("/audit/sessions")
async def get_audit_sessions(
    user_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    operation_type: Optional[str] = None,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_admin),
):
    """
    Get enhanced audit sessions with banking compliance information.
    Admin only endpoint.
    """
    try:
        # This would need to be implemented in mongodb_service
        # For now, return a placeholder
        return {
            "message": "Enhanced audit sessions endpoint",
            "filters": {
                "user_id": user_id,
                "risk_level": risk_level,
                "operation_type": operation_type,
                "limit": limit,
            },
            "note": "Full implementation requires MongoDB service enhancement",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving audit sessions: {str(e)}"
        )


@app.get("/audit/risk-dashboard")
async def get_risk_dashboard(
    time_range: str = "24h",  # 1h, 24h, 7d, 30d
    current_user: Dict[str, Any] = Depends(get_current_admin),
):
    """
    Get risk dashboard with banking compliance metrics.
    Admin only endpoint.
    """
    try:
        # This would analyze audit logs to provide risk metrics
        return {
            "message": "Risk dashboard endpoint",
            "time_range": time_range,
            "metrics": {
                "total_operations": 0,
                "high_risk_operations": 0,
                "de_scrub_requests": 0,
                "customer_data_operations": 0,
                "compliance_violations": 0,
            },
            "note": "Full implementation requires audit log analysis",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating risk dashboard: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
