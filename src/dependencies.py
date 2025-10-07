"""
Authentication dependencies for FastAPI endpoints
Fixed to not show parameters in OpenAPI docs
"""

from typing import Dict, Any
from fastapi import HTTPException, Depends, status
from src.auth import verify_token, verify_admin_token
from src.mongodb_service import MongoDBService

# Shared MongoDB service instance
mongodb_service = MongoDBService()


class GetCurrentUser:
    """
    Class-based dependency to get current authenticated user data.
    This approach prevents FastAPI from showing internal parameters in the docs.
    """

    def __call__(self, token_email: str = Depends(verify_token)) -> Dict[str, Any]:
        """
        Get current authenticated user data from JWT token.
        JWT token is automatically extracted from Authorization header.
        """
        if not mongodb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Get user data
        user_data = mongodb_service.get_user_by_email(token_email)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Check if user is active
        if not user_data.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated"
            )

        return user_data


class GetCurrentAdmin:
    """
    Class-based dependency to get current authenticated admin data.
    This approach prevents FastAPI from showing internal parameters in the docs.
    """

    def __call__(
        self, admin_token_email: str = Depends(verify_admin_token)
    ) -> Dict[str, Any]:
        """
        Get current authenticated admin data from JWT token.
        JWT token is automatically extracted from Authorization header.
        """
        if not mongodb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Get admin user data
        admin_data = mongodb_service.get_user_by_email(admin_token_email)
        if not admin_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Admin user not found"
            )

        # Double-check admin status
        if not admin_data.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

        # Check if admin is active
        if not admin_data.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is deactivated",
            )

        return admin_data


# Create instances of the dependency classes
get_current_user = GetCurrentUser()
get_current_admin = GetCurrentAdmin()
