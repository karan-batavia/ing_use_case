import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import json


class MongoDBService:
    """Service class for MongoDB operations to track user sessions and interactions"""

    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = None
        self.db = None
        self.sessions_collection = None
        self.interactions_collection = None
        self.users_collection = None

        # MongoDB connection settings
        self.mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DB_NAME", "ing_prompt_scrubber")

        self._connect()

    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.admin.command("ping")

            self.db = self.client[self.db_name]
            self.sessions_collection = self.db["user_sessions"]
            self.interactions_collection = self.db["user_interactions"]
            self.users_collection = self.db["users"]

            # Create indexes for better performance
            self._create_indexes()

            print(f"✅ Connected to MongoDB: {self.mongo_uri}")

        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            print("💡 Running without database logging")
            self.client = None
        except Exception as e:
            print(f"❌ MongoDB setup error: {e}")
            print("💡 Running without database logging")
            self.client = None

    def _create_indexes(self):
        """Create database indexes for better performance"""
        if not self.client:
            return

        try:
            # Session indexes
            self.sessions_collection.create_index("session_id", unique=True)
            self.sessions_collection.create_index("user_id")
            self.sessions_collection.create_index("created_at")

            # Interaction indexes
            self.interactions_collection.create_index("session_id")
            self.interactions_collection.create_index("user_id")
            self.interactions_collection.create_index("timestamp")
            self.interactions_collection.create_index("action_type")

            # User indexes
            self.users_collection.create_index("user_id", unique=True)

        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")

    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self.client is not None

    def create_session(self, user_id: str, user_role: str = "user") -> str:
        """Create a new user session"""
        if not self.client:
            return str(uuid.uuid4())  # Return UUID even without DB

        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "user_role": user_role,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "is_active": True,
            "interactions_count": 0,
        }

        try:
            self.sessions_collection.insert_one(session_data)
            print(f"📝 Created session: {session_id} for user: {user_id}")
            return session_id
        except PyMongoError as e:
            print(f"Error creating session: {e}")
            return session_id

    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for a session"""
        if not self.client or not session_id:
            return

        try:
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {"last_activity": datetime.now(timezone.utc)},
                    "$inc": {"interactions_count": 1},
                },
            )
        except PyMongoError as e:
            print(f"Error updating session activity: {e}")

    def end_session(self, session_id: str):
        """Mark a session as ended"""
        if not self.client or not session_id:
            return

        try:
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"is_active": False, "ended_at": datetime.now(timezone.utc)}},
            )
            print(f"🔚 Ended session: {session_id}")
        except PyMongoError as e:
            print(f"Error ending session: {e}")

    def log_interaction(
        self,
        session_id: str,
        user_id: str,
        action_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a user interaction"""
        if not self.client:
            return

        interaction_data = {
            "interaction_id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "action_type": action_type,
            "timestamp": datetime.now(timezone.utc),
            "details": details or {},
        }

        try:
            self.interactions_collection.insert_one(interaction_data)
            self.update_session_activity(session_id)

        except PyMongoError as e:
            print(f"Error logging interaction: {e}")

    def log_file_upload(
        self,
        session_id: str,
        user_id: str,
        filename: str,
        file_type: str,
        file_size: Optional[int] = None,
    ):
        """Log file upload interaction"""
        details = {"filename": filename, "file_type": file_type, "file_size": file_size}
        self.log_interaction(session_id, user_id, "file_upload", details)

    def log_text_scrubbing(
        self,
        session_id: str,
        user_id: str,
        input_length: int,
        output_length: int,
        matches_found: int,
        scrubbing_method: str = "presidio",
    ):
        """Log text scrubbing interaction"""
        details = {
            "input_length": input_length,
            "output_length": output_length,
            "matches_found": matches_found,
            "scrubbing_method": scrubbing_method,
            "reduction_percentage": (
                round((1 - output_length / input_length) * 100, 2)
                if input_length > 0
                else 0
            ),
        }
        self.log_interaction(session_id, user_id, "text_scrubbing", details)

    def log_file_download(
        self, session_id: str, user_id: str, filename: str, file_type: str
    ):
        """Log file download interaction"""
        details = {"filename": filename, "file_type": file_type}
        self.log_interaction(session_id, user_id, "file_download", details)

    def log_login(self, user_id: str, user_role: str):
        """Log user login"""
        # Create or update user record
        if self.client:
            try:
                self.users_collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "user_id": user_id,
                            "user_role": user_role,
                            "last_login": datetime.now(timezone.utc),
                        },
                        "$inc": {"login_count": 1},
                        "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
                    },
                    upsert=True,
                )
            except PyMongoError as e:
                print(f"Error logging user login: {e}")

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session"""
        if not self.client:
            return None

        try:
            # Get session info
            session = self.sessions_collection.find_one({"session_id": session_id})
            if not session:
                return None

            # Get interaction count
            interaction_count = self.interactions_collection.count_documents(
                {"session_id": session_id}
            )

            # Get interaction types
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {"_id": "$action_type", "count": {"$sum": 1}}},
            ]
            interaction_types = list(self.interactions_collection.aggregate(pipeline))

            return {
                "session_id": session_id,
                "user_id": session["user_id"],
                "user_role": session["user_role"],
                "created_at": session["created_at"],
                "last_activity": session["last_activity"],
                "total_interactions": interaction_count,
                "interaction_breakdown": {
                    item["_id"]: item["count"] for item in interaction_types
                },
            }

        except PyMongoError as e:
            print(f"Error getting session stats: {e}")
            return None

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific user"""
        if not self.client:
            return None

        try:
            # Get user info
            user = self.users_collection.find_one({"user_id": user_id})
            if not user:
                return None

            # Get session count
            session_count = self.sessions_collection.count_documents(
                {"user_id": user_id}
            )

            # Get total interactions
            interaction_count = self.interactions_collection.count_documents(
                {"user_id": user_id}
            )

            return {
                "user_id": user_id,
                "user_role": user.get("user_role", "unknown"),
                "created_at": user.get("created_at"),
                "last_login": user.get("last_login"),
                "login_count": user.get("login_count", 0),
                "total_sessions": session_count,
                "total_interactions": interaction_count,
            }

        except PyMongoError as e:
            print(f"Error getting user stats: {e}")
            return None

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")


# Streamlit session state integration
def get_mongodb_service() -> MongoDBService:
    """Get or create MongoDB service instance in Streamlit session state"""
    if "mongodb_service" not in st.session_state:
        st.session_state.mongodb_service = MongoDBService()
    return st.session_state.mongodb_service


def ensure_session_tracking():
    """Ensure session tracking is set up in Streamlit session state"""
    mongodb_service = get_mongodb_service()

    # Create session ID if not exists
    if "session_id" not in st.session_state:
        user_id = st.session_state.get("user_id", "anonymous")
        user_role = st.session_state.get("user_role", "user")

        session_id = mongodb_service.create_session(user_id, user_role)
        st.session_state.session_id = session_id

        # Log the session creation
        mongodb_service.log_login(user_id, user_role)

    return st.session_state.session_id


def log_app_interaction(action_type: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function to log interactions from anywhere in the app"""
    try:
        mongodb_service = get_mongodb_service()
        session_id = st.session_state.get("session_id")
        user_id = st.session_state.get("user_id", "anonymous")

        if session_id:
            mongodb_service.log_interaction(session_id, user_id, action_type, details)
    except Exception as e:
        print(f"Error logging interaction: {e}")
