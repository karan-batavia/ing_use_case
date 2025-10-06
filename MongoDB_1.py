import logging
import os
import json
import warnings
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
import collections.abc


# --- Utility: Deep Merge Dicts ---
def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# --- Secure Configuration Loader ---
class SecurePromptConfig:
    """Configuration manager for MongoDB metadata logging system."""

    def __init__(self, config_file: str = "logging_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        default_config = {
            "mongodb": {
                "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                "database": os.getenv("MONGODB_DB", "ing_document_processing"),
                "collection": os.getenv("MONGODB_COLLECTION", "processing_logs"),
            },
            "logging": {
                "level": "INFO",
                "log_file": "document_processing.log",
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
            },
            "processing": {
                "supported_formats": [
                    ".txt", ".docx", ".html", ".png", ".jpg", ".jpeg", ".pdf"
                ],
                "output_directory": "processed_documents",
            },
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    user_config = json.load(f)
                return deep_update(default_config, user_config)
            except Exception as e:
                warnings.warn(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            self._save_config(default_config)
            return default_config

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            warnings.warn(f"Error saving config: {e}")

    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'mongodb.uri')."""
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def update(self, key_path: str, value: Any):
        """Update config value and save."""
        keys = key_path.split(".")
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config(self.config)


# --- Safe Default Context for LogRecords ---
old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    if not hasattr(record, "context"):
        record.context = "general"
    if not hasattr(record, "file_info"):
        record.file_info = {}
    if not hasattr(record, "processing_stats"):
        record.processing_stats = {}
    if not hasattr(record, "session_id"):
        record.session_id = None
    return record
logging.setLogRecordFactory(record_factory)


# --- Custom Formatter ---
class MetadataFormatter(logging.Formatter):
    """Formatter that adds metadata context to log output."""

    def format(self, record):
        base_msg = super().format(record)
        metadata_parts = []

        # File info
        if record.file_info:
            file_parts = []
            if "file_size" in record.file_info:
                file_parts.append(f"Size:{record.file_info['file_size']/1024/1024:.1f}MB")
            if "file_type" in record.file_info:
                file_parts.append(f"Type:{record.file_info['file_type']}")
            if file_parts:
                metadata_parts.append(f"File({', '.join(file_parts)})")

        # Processing stats
        if record.processing_stats:
            stats = record.processing_stats
            stat_parts = []
            if "processing_time" in stats:
                stat_parts.append(f"Time:{stats['processing_time']:.1f}s")
            if "confidence" in stats:
                stat_parts.append(f"Confidence:{stats['confidence']:.1%}")
            if "characters_extracted" in stats:
                stat_parts.append(f"Chars:{stats['characters_extracted']}")
            if "output_length" in stats:
                stat_parts.append(f"Output:{stats['output_length']}")
            if "pages_processed" in stats:
                stat_parts.append(f"Pages:{stats['pages_processed']}")
            if stat_parts:
                metadata_parts.append(f"Stats({', '.join(stat_parts)})")

        # Session ID
        if record.session_id:
            metadata_parts.append(f"Session:{record.session_id}")

        return f"{base_msg} | {' | '.join(metadata_parts)}" if metadata_parts else base_msg


# --- MongoDB Logging Handler ---
class DocumentProcessingMongoHandler(logging.Handler):
    """Handler that writes log metadata to MongoDB."""

    _shared_instance = None  # Shared handler (singleton-style reuse)

    def __new__(cls, config: SecurePromptConfig):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
        return cls._shared_instance

    def __init__(self, config: SecurePromptConfig):
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self.config = config
        self.connected = False
        self.collection = None
        try:
            uri = config.get("mongodb.uri")
            self.client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            self.client.server_info()
            db_name = config.get("mongodb.database")
            coll_name = config.get("mongodb.collection")
            self.collection = self.client[db_name][coll_name]
            self.connected = True
            logging.getLogger(__name__).info(f"Connected to MongoDB: {db_name}.{coll_name}")
        except Exception as e:
            warnings.warn(f"MongoDB not available ({e}). Using file logging only.")
        self._initialized = True

    def emit(self, record):
        if not self.connected:
            return
        try:
            log_entry = {
                "timestamp": datetime.utcnow(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "context": record.context,
                "source": {
                    "filename": record.pathname,
                    "line": record.lineno,
                    "function": record.funcName,
                },
                "file_info": record.file_info,
                "processing_stats": record.processing_stats,
                "session_id": record.session_id,
            }
            self.collection.insert_one(log_entry)
        except Exception as e:
            self.handleError(record)

    def close(self):
        if self.connected:
            self.client.close()
            self.connected = False
        super().close()


# --- Document Processing Logger Factory ---
class DocumentProcessingLogger:
    """Factory for context-specific document processing loggers."""

    def __init__(self, config_file: str = "logging_config.json"):
        self.config = SecurePromptConfig(config_file)
        self.loggers = {}
        self.mongo_handler = DocumentProcessingMongoHandler(self.config)

    def get_logger(self, name: str, context: str = "general") -> logging.Logger:
        key = f"{name.lower()}_{context.lower()}"
        if key in self.loggers:
            return self.loggers[key]

        logger = logging.getLogger(key)
        logger.setLevel(getattr(logging, self.config.get("logging.level", "INFO")))
        logger.handlers.clear()

        log_file = f"{context}_{self.config.get('logging.log_file')}"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.get("logging.max_file_size", 10 * 1024 * 1024),
            backupCount=self.config.get("logging.backup_count", 5),
        )
        fmt = MetadataFormatter("%(asctime)s - %(name)s - %(levelname)s - [%(context)s] - %(message)s")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        if self.mongo_handler.connected:
            self.mongo_handler.setLevel(logging.INFO)
            logger.addHandler(self.mongo_handler)

        if os.getenv("ENVIRONMENT", "production") == "development":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(MetadataFormatter("%(levelname)s - [%(context)s] - %(message)s"))
            logger.addHandler(console_handler)

        self.loggers[key] = logger
        return logger

    def log_document_processing(self, logger_name: str, context: str, message: str,
                                level: str = "INFO", **kwargs):
        logger = self.get_logger(logger_name, context)
        extra = {
            "context": context,
            "file_info": kwargs.get("file_info", {}),
            "processing_stats": kwargs.get("processing_stats", {}),
            "session_id": kwargs.get("session_id"),
        }
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message, extra=extra)


# --- Helper factory function ---
def setup_document_logger(config_file: str = "logging_config.json") -> DocumentProcessingLogger:
    return DocumentProcessingLogger(config_file)