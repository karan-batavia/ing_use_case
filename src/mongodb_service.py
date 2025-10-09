"""
MongoDB service for session tracking, users, roles, and analytics
- Accepts `role` on create_user (user|auditor|admin) and keeps legacy `is_admin`
- Conflict-safe index creation (no crashes on IndexOptionsConflict)
"""

from __future__ import annotations

import os
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from pymongo import MongoClient, ASCENDING
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    DuplicateKeyError,
)

def _extract_db_name(uri: str, default: str = "ing_prompt_scrubber") -> str:
    """Infer DB name from URI path; fall back to default."""
    try:
        tail = uri.rsplit("/", 1)[-1]
        name = tail.split("?", 1)[0].strip()
        return name or default
    except Exception:
        return default


class MongoDBService:
    def __init__(self) -> None:
        # Prefer env var, fall back to localhost
        self.mongodb_url = os.getenv(
            "MONGODB_URL",
            "mongodb://inguser:ingpassword@localhost:27017/ing_prompt_scrubber?authSource=admin",
        )
        self.database_name = _extract_db_name(self.mongodb_url, "ing_prompt_scrubber")

        self.client: Optional[MongoClient] = None
        self.db = None
        self.connected: bool = False

        # Bound after connect
        self.users = None
        self.sessions = None
        self.interactions = None
        self.user_stats = None
        self.redactions = None

        self.logger = logging.getLogger(__name__)
        self._connect()

    # ------------------------------------------------------------------ #
    # Connection & health
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        """Establish MongoDB connection with error handling."""
        try:
            self.client = MongoClient(
                self.mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=10,
            )
            # Ping to verify
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.connected = True

            # Bind collections
            self.users = self.db["users"]
            self.sessions = self.db["sessions"]
            self.interactions = self.db["interactions"]
            self.user_stats = self.db["user_stats"]
            self.redactions = self.db["redactions"]

            # Ensure collections & indexes
            self._ensure_collections()

            self.logger.info(f"Successfully connected to MongoDB: {self.database_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.warning(f"MongoDB connection failed: {e}")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False

    def is_connected(self) -> bool:
        """Live connectivity check (safe to call anywhere)."""
        if not self.connected or not self.client:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            self.connected = False
            return False

    # ------------------------------------------------------------------ #
    # Index helpers (conflict-safe)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_keys(keys):
        return tuple((k, int(v)) for k, v in keys)

    def _ensure_index(
        self,
        coll,
        keys,
        *,
        unique: bool | None = None,
        sparse: bool | None = None,
        name: str | None = None,
    ):
        """
        Create index only if an equivalent one doesn't already exist.
        If an index with the same keys but different options exists:
          - If MONGO_AUTOFIX_INDEXES=true, drop & recreate
          - Otherwise, log a warning and keep the existing index
        """
        try:
            target_keys = self._normalize_keys(keys)
            info = coll.index_information()  # {name: {'key': [('field', 1)], ...}}

            existing_name = None
            existing_opts = None
            for iname, spec in info.items():
                spec_keys = self._normalize_keys(spec.get("key", []))
                if spec_keys == target_keys:
                    existing_name = iname
                    existing_opts = spec
                    break

            if existing_name:
                existing_unique = bool(existing_opts.get("unique", False))
                existing_sparse = bool(existing_opts.get("sparse", False))
                desired_unique = bool(unique) if unique is not None else existing_unique
                desired_sparse = bool(sparse) if sparse is not None else existing_sparse

                if existing_unique == desired_unique and existing_sparse == desired_sparse:
                    # Already compatible
                    return existing_name

                # Options differ
                if os.getenv("MONGO_AUTOFIX_INDEXES", "false").lower() in ("1", "true", "yes"):
                    self.logger.warning(
                        f"Recreating index on {coll.name} {target_keys}: "
                        f"existing(name={existing_name}, unique={existing_unique}, sparse={existing_sparse}) "
                        f"→ desired(unique={desired_unique}, sparse={desired_sparse})"
                    )
                    try:
                        coll.drop_index(existing_name)
                    except Exception as de:
                        self.logger.warning(f"Drop index failed ({existing_name}): {de}")
                    # fall through to create
                else:
                    self.logger.warning(
                        f"Index exists on {coll.name} {target_keys} with different options "
                        f"(name={existing_name}, unique={existing_unique}, sparse={existing_sparse}). "
                        "Leaving as-is. Set MONGO_AUTOFIX_INDEXES=true to auto-recreate."
                    )
                    return existing_name

            # Create (avoid specifying name unless necessary)
            return coll.create_index(
                keys,
                unique=bool(unique) if unique else False,
                sparse=bool(sparse) if sparse else False,
                name=name,
            )

        except Exception as e:
            msg = str(e)
            if "IndexOptionsConflict" in msg or "already exists with a different name" in msg:
                self.logger.warning(f"Index already present (conflict handled) on {coll.name} {keys}: {e}")
                return None
            self.logger.error(f"Error ensuring index on {coll.name} {keys}: {e}")
            return None

    def _ensure_collections(self):
        """Create collections and reconcile indexes (idempotent & conflict-safe)."""
        if not self.is_connected():
            return

        try:
            needed = ["sessions", "interactions", "user_stats", "users", "redactions"]
            existing = set(self.db.list_collection_names())
            for c in needed:
                if c not in existing:
                    self.db.create_collection(c)
                    self.logger.info(f"Created collection: {c}")

            # Sessions
            self._ensure_index(self.sessions, [("session_id", 1)], unique=True)
            self._ensure_index(self.sessions, [("user_id", 1)])
            self._ensure_index(self.sessions, [("created_at", -1)])

            # Interactions
            self._ensure_index(self.interactions, [("session_id", 1)])
            self._ensure_index(self.interactions, [("user_id", 1)])
            self._ensure_index(self.interactions, [("timestamp", -1)])

            # User stats
            self._ensure_index(self.user_stats, [("user_id", 1)], unique=True)

            # Users
            self._ensure_index(self.users, [("email", 1)], unique=True)
            self._ensure_index(self.users, [("user_id", 1)], unique=True)
            self._ensure_index(self.users, [("created_at", -1)])

            # Redactions
            self._ensure_index(self.redactions, [("session_id", 1)], unique=True)
            self._ensure_index(self.redactions, [("redacted_hash", 1)])
        except Exception as e:
            self.logger.error(f"Error ensuring collections/indexes: {e}")

    # ------------------------------------------------------------------ #
    # Users
    # ------------------------------------------------------------------ #
    def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        role: Optional[str] = None,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        """Create a new user. Accepts `role` (preferred) or legacy `is_admin`."""
        if not self.is_connected():
            return {"success": False, "error": "Database not connected"}

        try:
            normalized_email = (email or "").strip().lower()
            if not normalized_email:
                return {"success": False, "error": "Email is required"}

            existing_user = self.users.find_one({"email": normalized_email})
            if existing_user:
                return {"success": False, "error": "User already exists"}

            # Resolve role (keep legacy in sync)
            role = (role or "").strip().lower() if role else ("admin" if is_admin else "user")
            if role not in {"user", "auditor", "admin"}:
                role = "user"
            is_admin = bool(is_admin or role == "admin")

            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            user_data = {
                "user_id": user_id,
                "email": normalized_email,
                "hashed_password": hashed_password,
                "full_name": full_name,
                "is_active": True,
                "is_admin": is_admin,  # legacy
                "role": role,          # canonical
                "created_at": now,
                "last_login": None,
            }

            self.users.insert_one(user_data)

            # Initialize user stats
            self.user_stats.update_one(
                {"user_id": user_id},
                {
                    "$setOnInsert": {
                        "user_id": user_id,
                        "total_sessions": 0,
                        "total_interactions": 0,
                        "total_text_processed": 0,
                        "total_chars_scrubbed": 0,
                        "last_activity": now,
                        "created_at": now,
                    }
                },
                upsert=True,
            )

            self.logger.info(f"User created successfully: {normalized_email} with ID: {user_id}")
            return {"success": True, "user_id": user_id}

        except DuplicateKeyError:
            return {"success": False, "error": "User already exists"}
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return {"success": False, "error": str(e)}

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email (normalized); ensure role is normalized and _id is string."""
        if not self.is_connected():
            return None
        try:
            doc = self.users.find_one({"email": (email or "").strip().lower()})
            if not doc:
                return None
            out = dict(doc)
            if "_id" in out:
                out["_id"] = str(out["_id"])
            out["user_id"] = out.get("user_id") or out.get("_id")
            out["role"] = (out.get("role") or ("admin" if out.get("is_admin", False) else "user")).lower()
            return out
        except Exception as e:
            self.logger.error(f"Error getting user by email: {e}")
            return None

    def update_last_login(self, email: str) -> bool:
        """Update user's last login timestamp."""
        if not self.is_connected():
            return False
        try:
            res = self.users.update_one(
                {"email": (email or "").strip().lower()},
                {"$set": {"last_login": datetime.now(timezone.utc)}},
            )
            return res.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating last login: {e}")
            return False

    def deactivate_user(self, email: str) -> bool:
        """Deactivate a user account."""
        if not self.is_connected():
            return False
        try:
            res = self.users.update_one(
                {"email": (email or "").strip().lower()},
                {"$set": {"is_active": False}},
            )
            return res.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error deactivating user: {e}")
            return False

    def get_all_users(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all users (excluding password hash)."""
        if not self.is_connected():
            return []
        try:
            users = list(
                self.users.find({}, {"hashed_password": 0})
                .skip(skip)
                .limit(limit)
                .sort("created_at", -1)
            )
            for u in users:
                if "_id" in u:
                    u["_id"] = str(u["_id"])
                u["role"] = (u.get("role") or ("admin" if u.get("is_admin", False) else "user")).lower()
            return users
        except Exception as e:
            self.logger.error(f"Error getting all users: {e}")
            return []

    def migrate_users_add_user_id(self) -> Dict[str, Any]:
        """Ensure every user has user_id."""
        if not self.is_connected():
            return {"success": False, "error": "Database not connected"}
        try:
            to_fix = list(self.users.find({"user_id": {"$exists": False}}))
            migrated = 0
            for u in to_fix:
                uid = str(uuid.uuid4())
                res = self.users.update_one({"_id": u["_id"]}, {"$set": {"user_id": uid}})
                if res.modified_count > 0:
                    migrated += 1
            return {"success": True, "migrated_count": migrated}
        except Exception as e:
            self.logger.error(f"Error migrating users: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Sessions & interactions
    # ------------------------------------------------------------------ #
    def create_session(self, user_id: str, source: str = "api", user_role: str = "user") -> str:
        """Create a new session and return session ID."""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, returning local session id")
            return f"offline-{uuid.uuid4()}"

        try:
            session_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            self.sessions.insert_one(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "source": source,
                    "user_role": user_role,
                    "created_at": now,
                    "last_activity": now,
                    "status": "active",
                }
            )
            # bump stats
            self.user_stats.update_one(
                {"user_id": user_id},
                {"$inc": {"total_sessions": 1}, "$set": {"last_activity": now}},
                upsert=True,
            )
            return session_id
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return str(uuid.uuid4())  # fallback

    def log_interaction(
        self,
        session_id: str,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log user interaction and bump stats."""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot log interaction")
            return
        try:
            now = datetime.now(timezone.utc)
            self.interactions.insert_one(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "action": action,
                    "details": details or {},
                    "timestamp": now,
                }
            )
            self.sessions.update_one(
                {"session_id": session_id}, {"$set": {"last_activity": now}}
            )
            self.user_stats.update_one(
                {"user_id": user_id},
                {"$inc": {"total_interactions": 1}, "$set": {"last_activity": now}},
                upsert=True,
            )
        except Exception as e:
            self.logger.error(f"Error logging interaction: {e}")

    # ------------------------------------------------------------------ #
    # Redaction records
    # ------------------------------------------------------------------ #
    def save_redaction_record(
        self,
        *,
        session_id: str,
        user_id: str,
        original_text: str,
        redacted_text: str,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Persist redaction mapping for later descrubbing."""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot persist redaction record")
            return None

        try:
            redacted_hash = hashlib.sha256(redacted_text.encode("utf-8")).hexdigest()
            now = datetime.now(timezone.utc)
            record = {
                "session_id": session_id,
                "user_id": user_id,
                "original_text": original_text,
                "redacted_text": redacted_text,
                "redacted_hash": redacted_hash,
                "detections": detections or [],
                "created_at": now,
                "updated_at": now,
            }

            self.redactions.update_one(
                {"session_id": session_id},
                {"$set": record},
                upsert=True,
            )
            return session_id
        except Exception as e:
            self.logger.error(f"Error saving redaction record: {e}")
            return None

    def get_redaction_record_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            return None
        try:
            record = self.redactions.find_one({"session_id": session_id})
            if record and "_id" in record:
                record["_id"] = str(record["_id"])
            return record
        except Exception as e:
            self.logger.error(f"Error retrieving redaction record: {e}")
            return None

    def descrub_text(self, scrubbed_text: str, session_id: Optional[str] = None) -> Optional[str]:
        """Restore original text using stored redaction mapping."""
        if not self.is_connected():
            return None

        try:
            record: Optional[Dict[str, Any]] = None

            if session_id:
                record = self.redactions.find_one({"session_id": session_id})

            if not record:
                redacted_hash = hashlib.sha256(scrubbed_text.encode("utf-8")).hexdigest()
                record = self.redactions.find_one({"redacted_hash": redacted_hash})

            if not record:
                return None

            detections = record.get("detections", []) or []
            restored_text = scrubbed_text
            for detection in detections:
                placeholder = detection.get("placeholder")
                original = detection.get("original")
                if placeholder and original is not None:
                    restored_text = restored_text.replace(placeholder, original, 1)
            return restored_text
        except Exception as e:
            self.logger.error(f"Error descrubbing text: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Stats helpers
    # ------------------------------------------------------------------ #
    def update_user_stats(self, user_id: str, stats_update: Dict[str, Any]) -> None:
        """Update user statistics (upsert)."""
        if not self.is_connected():
            self.logger.warning("MongoDB not connected, cannot update user stats")
            return
        try:
            self.user_stats.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        **stats_update,
                        "last_updated": datetime.now(timezone.utc),
                    },
                    "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
                },
                upsert=True,
            )
        except Exception as e:
            self.logger.error(f"Error updating user stats: {e}")

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user statistics."""
        if not self.is_connected():
            return None
        try:
            stats = self.user_stats.find_one({"user_id": user_id})
            if stats:
                stats.pop("_id", None)
            return stats
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return None

    def get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a session."""
        if not self.is_connected():
            return []
        try:
            rows = list(self.interactions.find({"session_id": session_id}).sort("timestamp", 1))
            for r in rows:
                r.pop("_id", None)
            return rows
        except Exception as e:
            self.logger.error(f"Error getting session interactions: {e}")
            return []

    def close_session(self, session_id: str) -> None:
        """Mark session as closed."""
        if not self.is_connected():
            return
        try:
            self.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"status": "closed", "closed_at": datetime.now(timezone.utc)}},
            )
        except Exception as e:
            self.logger.error(f"Error closing session: {e}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics (lightweight)."""
        if not self.is_connected():
            return {"connected": False, "error": "Not connected to MongoDB"}
        try:
            stats: Dict[str, Any] = {
                "connected": True,
                "database": self.database_name,
                "collections": {},
                "server_info": self.client.server_info(),
            }
            for col in ["sessions", "interactions", "user_stats", "users"]:
                try:
                    stats["collections"][col] = self.db[col].count_documents({})
                except Exception as e:
                    stats["collections"][col] = f"Error: {e}"
            return stats
        except Exception as e:
            return {"connected": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close_connection(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("MongoDB connection closed")
