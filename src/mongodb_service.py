"""
MongoDB service for session tracking and user analytics
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import uuid


class MongoDBService:
    def __init__(self):
        """Initialize MongoDB connection with environment-based configuration"""
        # Get MongoDB URL from environment variable or use default
        self.mongodb_url = os.getenv(
            "MONGODB_URL",
            "mongodb://inguser:ingpassword@localhost:27017/ing_prompt_scrubber?authSource=admin",
        )

        self.database_name = "ing_prompt_scrubber"
        self.client = None
        self.db = None
        self.connected = False

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize connection
        self._connect()

    def _connect(self):
        """Establish MongoDB connection with error handling"""
        try:
            self.client = MongoClient(
                self.mongodb_url,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=10,
            )

            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.connected = True

            # Ensure collections exist
            self._ensure_collections()

            self.logger.info(f"Successfully connected to MongoDB: {self.database_name}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.warning(f"MongoDB connection failed: {e}")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False

    def _ensure_collections(self):
        """Create collections if they don't exist"""
        if not self.connected:
            return

        try:
            collections = ["sessions", "interactions", "user_stats", "users"]
            existing_collections = self.db.list_collection_names()

            for collection in collections:
                if collection not in existing_collections:
                    self.db.create_collection(collection)
                    self.logger.info(f"Created collection: {collection}")

            # Create indexes for better performance
            self.db.sessions.create_index("session_id", unique=True)
            self.db.sessions.create_index("user_id")
            self.db.sessions.create_index("created_at")

            self.db.interactions.create_index("session_id")
            self.db.interactions.create_index("user_id")
            self.db.interactions.create_index("timestamp")

            self.db.user_stats.create_index("user_id", unique=True)

            # User collection indexes
            self.db.users.create_index("email", unique=True)
            self.db.users.create_index("created_at")

        except Exception as e:
            self.logger.error(f"Error ensuring collections: {e}")

    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        if not self.connected:
            return False

        try:
            self.client.admin.command("ping")
            return True
        except:
            self.connected = False
            return False

    def create_session(self, user_id: str, user_role: str = "user") -> str:
        """Create a new session and return session ID"""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot create session")
            return str(uuid.uuid4())  # Return a UUID even if DB is down

        try:
            session_id = str(uuid.uuid4())
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "user_role": user_role,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "status": "active",
            }

            self.db.sessions.insert_one(session_data)
            self.logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return str(uuid.uuid4())  # Fallback to UUID

    def log_interaction(
        self,
        session_id: str,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log user interaction"""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot log interaction")
            return

        try:
            interaction_data = {
                "session_id": session_id,
                "user_id": user_id,
                "action": action,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc),
            }

            self.db.interactions.insert_one(interaction_data)

            # Update session last activity
            self.db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"last_activity": datetime.now(timezone.utc)}},
            )

            self.logger.debug(f"Logged interaction: {action} for session {session_id}")

        except Exception as e:
            self.logger.error(f"Error logging interaction: {e}")

    def update_user_stats(self, user_id: str, stats_update: Dict[str, Any]):
        """Update user statistics"""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot update user stats")
            return

        try:
            # Use upsert to create or update user stats
            self.db.user_stats.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "last_updated": datetime.now(timezone.utc),
                        **stats_update,
                    },
                    "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
                },
                upsert=True,
            )

            self.logger.debug(f"Updated stats for user {user_id}")

        except Exception as e:
            self.logger.error(f"Error updating user stats: {e}")

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user statistics"""
        if not self.is_connected():
            return None

        try:
            stats = self.db.user_stats.find_one({"user_id": user_id})
            if stats:
                # Remove MongoDB ObjectId for JSON serialization
                stats.pop("_id", None)
            return stats

        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return None

    def get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a session"""
        if not self.is_connected():
            return []

        try:
            interactions = list(
                self.db.interactions.find({"session_id": session_id}).sort(
                    "timestamp", 1
                )
            )

            # Remove MongoDB ObjectIds
            for interaction in interactions:
                interaction.pop("_id", None)

            return interactions

        except Exception as e:
            self.logger.error(f"Error getting session interactions: {e}")
            return []

    def close_session(self, session_id: str):
        """Mark session as closed"""
        if not self.is_connected():
            return

        try:
            self.db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"status": "closed", "closed_at": datetime.now(timezone.utc)}},
            )

            self.logger.info(f"Closed session {session_id}")

        except Exception as e:
            self.logger.error(f"Error closing session: {e}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_connected():
            return {"connected": False, "error": "Not connected to MongoDB"}

        try:
            stats = {
                "connected": True,
                "database": self.database_name,
                "collections": {},
                "server_info": self.client.server_info(),
            }

            # Get collection counts
            for collection_name in ["sessions", "interactions", "user_stats", "users"]:
                try:
                    count = self.db[collection_name].count_documents({})
                    stats["collections"][collection_name] = count
                except Exception as e:
                    stats["collections"][collection_name] = f"Error: {e}"

            return stats

        except Exception as e:
            return {"connected": False, "error": str(e)}

    # User Management Methods
    def create_user(
        self,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        """Create a new user"""
        if not self.connected:
            return {"success": False, "error": "Database not connected"}

        try:
            # Check if user already exists
            existing_user = self.db.users.find_one({"email": email})
            if existing_user:
                return {"success": False, "error": "User already exists"}

            # Generate UUID for user_id
            user_id = str(uuid.uuid4())

            user_data = {
                "user_id": user_id,
                "email": email,
                "hashed_password": hashed_password,
                "full_name": full_name,
                "is_active": True,
                "is_admin": is_admin,
                "created_at": datetime.now(timezone.utc),
                "last_login": None,
            }

            result = self.db.users.insert_one(user_data)

            if result.inserted_id:
                # Initialize user stats with the UUID
                self.db.user_stats.insert_one(
                    {
                        "user_id": user_id,
                        "total_sessions": 0,
                        "total_interactions": 0,
                        "total_text_processed": 0,
                        "total_chars_scrubbed": 0,
                        "last_activity": datetime.now(timezone.utc),
                        "created_at": datetime.now(timezone.utc),
                    }
                )

                self.logger.info(
                    f"User created successfully: {email} with ID: {user_id}"
                )
                return {"success": True, "user_id": user_id}
            else:
                return {"success": False, "error": "Failed to create user"}

        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return {"success": False, "error": str(e)}

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        if not self.connected:
            return None

        try:
            user = self.db.users.find_one({"email": email})
            if user:
                user["_id"] = str(user["_id"])  # Convert ObjectId to string
            return user
        except Exception as e:
            self.logger.error(f"Error getting user by email: {e}")
            return None

    def update_last_login(self, email: str) -> bool:
        """Update user's last login timestamp"""
        if not self.connected:
            return False

        try:
            result = self.db.users.update_one(
                {"email": email}, {"$set": {"last_login": datetime.now(timezone.utc)}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating last login: {e}")
            return False

    def deactivate_user(self, email: str) -> bool:
        """Deactivate a user account"""
        if not self.connected:
            return False

        try:
            result = self.db.users.update_one(
                {"email": email}, {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error deactivating user: {e}")
            return False

    def get_all_users(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all users with pagination"""
        if not self.connected:
            return []

        try:
            users = list(
                self.db.users.find({}, {"hashed_password": 0})  # Exclude password hash
                .skip(skip)
                .limit(limit)
                .sort("created_at", -1)
            )

            # Convert ObjectId to string
            for user in users:
                user["_id"] = str(user["_id"])

            return users
        except Exception as e:
            self.logger.error(f"Error getting all users: {e}")
            return []

    def migrate_users_add_user_id(self) -> Dict[str, Any]:
        """Migrate existing users to add user_id field"""
        if not self.connected:
            return {"success": False, "error": "Database not connected"}

        try:
            # Find users without user_id field
            users_to_migrate = list(self.db.users.find({"user_id": {"$exists": False}}))

            if not users_to_migrate:
                return {
                    "success": True,
                    "message": "No users need migration",
                    "migrated_count": 0,
                }

            migrated_count = 0
            for user in users_to_migrate:
                user_id = str(uuid.uuid4())

                # Update user document
                result = self.db.users.update_one(
                    {"_id": user["_id"]}, {"$set": {"user_id": user_id}}
                )

                if result.modified_count > 0:
                    # Update user_stats to use new user_id instead of email
                    self.db.user_stats.update_one(
                        {"user_id": user["email"]},  # Old format used email as user_id
                        {"$set": {"user_id": user_id}},
                    )
                    migrated_count += 1
                    self.logger.info(
                        f"Migrated user {user['email']} with new user_id: {user_id}"
                    )

            return {
                "success": True,
                "message": f"Migrated {migrated_count} users",
                "migrated_count": migrated_count,
            }

        except Exception as e:
            self.logger.error(f"Error migrating users: {e}")
            return {"success": False, "error": str(e)}

    def get_redaction_records(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get redaction records from the database.

        Args:
            user_id: Optional user ID to filter records
            limit: Maximum number of records to return

        Returns:
            List of redaction records with detections
        """
        if not self.connected:
            self.logger.error("Not connected to MongoDB")
            return []

        try:
            query = {"action": "text_redacted"}
            if user_id:
                query["user_id"] = user_id

            records = list(
                self.db.interactions.find(query).sort("timestamp", -1).limit(limit)
            )
            print(f"Found {len(records)} redaction records", records)

            # Convert ObjectId to string for JSON serialization
            for record in records:
                if "_id" in record:
                    record["_id"] = str(record["_id"])

            return records

        except Exception as e:
            self.logger.error(f"Error getting redaction records: {e}")
            return []

    def get_redaction_record_by_session(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific redaction record by session ID.

        Args:
            session_id: Session ID of the redaction

        Returns:
            Redaction record or None if not found
        """
        if not self.connected:
            self.logger.error("Not connected to MongoDB")
            return None

        try:
            record = self.db.interactions.find_one(
                {"session_id": session_id, "action": "text_redacted"}
            )

            if record and "_id" in record:
                record["_id"] = str(record["_id"])

            return record

        except Exception as e:
            self.logger.error(f"Error getting redaction record: {e}")
            return None

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("MongoDB connection closed")
