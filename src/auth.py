"""
Authentication utilities for JWT tokens and password hashing
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str


class TokenData(BaseModel):
    email: Optional[str] = None


class User(BaseModel):
    user_id: Optional[str] = None
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    is_admin: bool = False

    @field_validator("password")
    def validate_password(cls, v):
        if len(v) < 3:
            raise ValueError("Password must be at least 3 characters long")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password too long (max 72 bytes)")
        return v


class UserLogin(BaseModel):
    email: str
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    # Ensure password is not too long for bcrypt (72 bytes max)
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password too long (max 72 bytes)")

    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return email"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Please provide Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Token may be expired or invalid.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
        return str(email)
    except JWTError as e:
        # Check if it's an expired token
        if "expired" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please login again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )


def authenticate_user(
    email: str, password: str, user_db_function
) -> Union[UserInDB, bool]:
    """Authenticate a user against the database"""
    user = user_db_function(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def verify_admin_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify JWT token and ensure user is admin"""
    # First verify the token is valid
    email = verify_token(credentials)

    # Import here to avoid circular imports
    from src.mongodb_service import MongoDBService

    # Get user data and check admin status
    mongodb_service = MongoDBService()
    if not mongodb_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    user_data = mongodb_service.get_user_by_email(email)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if not user_data.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )

    return email
