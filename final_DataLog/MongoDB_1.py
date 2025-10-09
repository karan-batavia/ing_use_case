import logging
import os
import json
import warnings
import threading
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Mapping, MutableMapping
from pathlib import Path
from logging.handlers import RotatingFileHandler
import collections.abc
import uuid
import platform
import socket

# Optional pandas import for DataFrame helpers
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    pd = None
    HAS_PANDAS = False

try:
    from pymongo import MongoClient
    HAS_PYMONGO = True
except Exception:
    MongoClient = None
    HAS_PYMONGO = False
    warnings.warn("pymongo not installed; MongoDB logging disabled. File + JSON logging remain enabled.")


# --- Utility: Deep Merge Dicts ---
def deep_update(d: MutableMapping[str, Any], u: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# --- Secure Configuration Loader ---
class SecurePromptConfig:
    def __init__(self, config_file: str = "logging_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {
            "mongodb": {
                "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                "database": os.getenv("MONGODB_DB", "ing_document_processing"),
                "collection": os.getenv("MONGODB_COLLECTION", "processing_logs"),
            },
            "logging": {
                "level": "INFO",
                "log_file": "document_processing.log",
                # Optional logs directory; if empty or not set, Path.cwd() is used.
                "dir": "",
                "max_file_size": 10 * 1024 * 1024,
                "backup_count": 5,
            },
        }
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                return deep_update(default_config, user_config)
            except Exception as exc:
                warnings.warn(f"Error loading config: {exc}. Using defaults.")
                return default_config
        else:
            try:
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=2)
            except Exception as exc:
                warnings.warn(f"Unable to create config file: {exc}")
            return default_config

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


# --- Helpers ---
def hash_prompt(prompt: str) -> str:
    if prompt is None:
        return None
    return f"sha256:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"


def validate_classification(c: Optional[str]) -> str:
    allowed = {"C1", "C2", "C3", "C4"}
    if not c:
        return "C2"
    c = c.strip().upper()
    return c if c in allowed else "C2"


# --- Safe LogRecord Factory ---
old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    # Do not predefine keys that will be passed via `extra=` to logger.log()
    for attr in ["context", "file_info", "processing_stats", "session_id"]:
        if not hasattr(record, attr):
            setattr(record, attr, None)
    return record
logging.setLogRecordFactory(record_factory)


# --- Formatter ---
class MetadataFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base_msg = super().format(record)
        metadata_parts = []

        ctx = getattr(record, "dp_context", getattr(record, "context", None))
        if ctx:
            metadata_parts.append(f"Ctx:{ctx}")

        file_info = getattr(record, "dp_file_info", getattr(record, "file_info", None))
        if file_info:
            parts = []
            if "file_size" in file_info:
                try:
                    parts.append(f"Size:{file_info['file_size']/1024/1024:.1f}MB")
                except Exception:
                    parts.append(f"Size:{file_info.get('file_size')}")
            if "file_type" in file_info:
                parts.append(f"Type:{file_info['file_type']}")
            if parts:
                metadata_parts.append(f"File({', '.join(parts)})")

        stats = getattr(record, "dp_processing_stats", getattr(record, "processing_stats", None))
        if stats:
            parts = []
            for key in ["processing_time", "confidence", "characters_extracted", "output_length", "pages_processed"]:
                if key in stats:
                    parts.append(f"{key}:{stats[key]}")
            if parts:
                metadata_parts.append(f"Stats({', '.join(parts)})")

        session = getattr(record, "dp_session_id", getattr(record, "session_id", None))
        if session:
            metadata_parts.append(f"Session:{session}")

        return f"{base_msg} | {' | '.join(metadata_parts)}" if metadata_parts else base_msg


# --- MongoDB Handler + JSON Logging ---
class DocumentProcessingHandler(logging.Handler):
    _shared_instance: Optional['DocumentProcessingHandler'] = None
    _lock = threading.Lock()

    def __new__(cls, config: SecurePromptConfig):
        if cls._shared_instance is None:
            with cls._lock:
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
        self.client = None
        # Determine logs directory: prefer explicit config -> cwd
        cfg_dir = (self.config.get('logging.dir') or '').strip()
        if cfg_dir:
            try:
                resolved = Path(cfg_dir)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                resolved.mkdir(parents=True, exist_ok=True)
                self.logs_dir = resolved
            except Exception:
                self.logs_dir = Path.cwd()
        else:
            self.logs_dir = Path.cwd()

        # JSON file paths resolved under logs_dir
        self.JSON_FILE = self.logs_dir / "secure_processing_logs.json"
        # app metadata
        self.app_version = self.config.get('app.version', os.getenv('APP_VERSION', 'dev'))

        if HAS_PYMONGO:
            try:
                self.client = MongoClient(config.get("mongodb.uri"))
                db_name = config.get("mongodb.database")
                coll_name = config.get("mongodb.collection")
                self.collection = self.client[db_name][coll_name]
                self.connected = True
            except Exception:
                warnings.warn("MongoDB connection failed; only file/JSON logging will be used.")

        self._initialized = True
        # Ensure JSON file exists (in current working directory)
        try:
            if not self.JSON_FILE.exists():
                with open(self.JSON_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        # Use fixed timezone offset +02:00 for timestamps
        try:
            tz_plus2 = timezone(datetime.timedelta(hours=2))
        except Exception:
            # fallback if datetime.timedelta not available in typing
            from datetime import timedelta as _td
            tz_plus2 = timezone(_td(hours=2))

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(tz_plus2).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "dp_context", getattr(record, "context", "general")),
            "file_info": getattr(record, "dp_file_info", getattr(record, "file_info", {})),
            "processing_stats": getattr(record, "dp_processing_stats", getattr(record, "processing_stats", {})),
            "session_id": getattr(record, "dp_session_id", getattr(record, "session_id", None)),
        }

        extra = getattr(record, "dp_extra_fields", {}) or {}

        # Classification
        log_entry["classification"] = validate_classification(extra.get("classification"))

        # Prompt/Response metadata
        prompt_meta = extra.get("prompt_metadata", {})
        if "original_prompt" in prompt_meta:
            orig = prompt_meta.pop("original_prompt")
            prompt_meta["original_prompt_hash"] = hash_prompt(orig)
            # Do NOT store original prompt in clear; but keep scrubbed/descrubbed if provided
            # If caller included 'scrubbed' or 'descrubbed' we copy them to top-level audit fields
            # so the UI can display them safely.
            if 'scrubbed_text' in prompt_meta:
                log_entry['scrubbed_text'] = prompt_meta.get('scrubbed_text')
            if 'descrubbed_text' in prompt_meta:
                log_entry['descrubbed_text'] = prompt_meta.get('descrubbed_text')
            if 'redacted_text' in prompt_meta:
                log_entry['redacted_text'] = prompt_meta.get('redacted_text')
        log_entry["prompt_metadata"] = prompt_meta
        log_entry["response_metadata"] = extra.get("response_metadata", {})

        # Model fields: performance, confidence, classification
        model_info = extra.get('model_info', {}) or {}
        if 'model_name' in model_info:
            log_entry['model_name'] = model_info.get('model_name')
        if 'model_version' in model_info:
            log_entry['model_version'] = model_info.get('model_version')
        if 'performance' in model_info:
            # e.g., {'accuracy':0.93, 'f1':0.88}
            log_entry['model_performance'] = model_info.get('performance')
        if 'confidence' in model_info:
            # numeric between 0-1 or percentage
            log_entry['model_confidence'] = model_info.get('confidence')
        if 'classification' in model_info:
            log_entry['classification'] = validate_classification(model_info.get('classification'))

        # Security/audit
        audit = extra.get("security_audit", {})
        if "ip_address" in audit:
            try:
                parts = audit["ip_address"].split(".")
                if len(parts) == 4:
                    audit["masked_ip"] = f"{parts[0]}.{parts[1]}.x.x"
            except Exception:
                audit["ip_hash"] = hash_prompt(audit.get("ip_address"))
            audit.pop("ip_address", None)
        log_entry["security_audit"] = audit

        # Add extended audit metadata
        try:
            log_entry['_audit_id'] = uuid.uuid4().hex
            log_entry['_host'] = platform.node() or socket.gethostname()
            log_entry['_pid'] = os.getpid()
            log_entry['_app_version'] = self.app_version
            # infer source
            lname = (record.name or '').lower()
            if 'ui' in lname:
                log_entry['_source'] = 'ui'
            elif 'cli' in lname or 'script' in lname:
                log_entry['_source'] = 'cli'
            else:
                log_entry['_source'] = 'app'
        except Exception:
            pass

        # Optional top-level extra fields
        for k in ["tags", "pipeline_step", "business_unit"]:
            if k in extra:
                log_entry[k] = extra[k]

        # Decide whether this entry should be persisted to local JSONL / Mongo.
        extra = getattr(record, 'dp_extra_fields', {}) or {}
        omit_persistence = False
        # Honor explicit caller instruction
        if extra.get('omit_persistence') or extra.get('transient') or extra.get('do_not_persist'):
            omit_persistence = True
        # No heuristic skipping: persistence is controlled only by explicit flags in extra_fields

        # Prepare a sanitized copy for persistence (never include raw original prompt)
        def _sanitize_for_persistence(entry: Dict[str, Any]) -> Dict[str, Any]:
            e = dict(entry)
            # Remove any raw prompt if present
            if 'prompt' in e:
                e.pop('prompt', None)
            pm = e.get('prompt_metadata', {}) or {}
            # remove any plain-text original prompt
            if 'original_prompt' in pm:
                pm.pop('original_prompt', None)
            e['prompt_metadata'] = pm
            # Scrub any obviously sensitive keys
            for k in list(e.keys()):
                if k.lower().startswith('secret') or k.lower().startswith('password') or k.lower().startswith('token'):
                    e.pop(k, None)
            return e

        # Also append a JSON-line to a local JSONL file for easy tailing by the UI
        if not omit_persistence:
            try:
                sanitized = _sanitize_for_persistence(log_entry)
                local_jsonl = self.logs_dir / "local_processing_logs.jsonl"
                with open(local_jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
            except Exception:
                # Never raise from the logger
                pass

            # Additionally, write the full (unsanitized) structured entry to a per-context log file
            try:
                ctx = (log_entry.get('context') or 'general')
                safe_ctx = ''.join(c for c in (ctx or '') if c.isalnum() or c in ('_','-')) or 'general'
                per_context_log = self.logs_dir / f"{safe_ctx}_document_processing.log"
                with open(per_context_log, 'a', encoding='utf-8') as pf:
                    pf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # --- Write to MongoDB ---
        # --- Write to MongoDB ---
        if not omit_persistence:
            if self.connected and self.collection is not None:
                try:
                    self.collection.insert_one(log_entry)
                except Exception:
                    self.handleError(record)


# -----------------------------
# DocumentProcessingLogger (factory wrapper)
# -----------------------------
class DocumentProcessingLogger:
    def __init__(self, config_file: str = 'logging_config.json'):
        self.config = SecurePromptConfig(config_file)
        self.handler = DocumentProcessingHandler(self.config)
        # backward-compatible alias used by other modules
        self.mongo_handler = self.handler
        self.loggers: Dict[str, logging.Logger] = {}

    def _sanitize_context(self, context: str) -> str:
        return ''.join(c for c in (context or '') if c.isalnum() or c in ('_','-')) or 'general'

    def get_logger(self, name: str, context: str = 'general') -> logging.Logger:
        key = f"{name.lower()}_{(context or 'general').lower()}"
        if key in self.loggers:
            return self.loggers[key]
        logger = logging.getLogger(key)
        logger.handlers.clear()
        logger.propagate = False
        level_name = self.config.get('logging.level', 'INFO')
        logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))

        safe_context = self._sanitize_context(context)
        base = Path(self.config.get('logging.log_file') or 'document_processing.log').name
        logfile = f"{safe_context}_{base}"
        # prefer handler's cwd if present (writes to app CWD), else fallback to Path.cwd()
        handler_cwd = getattr(self.handler, 'cwd', Path.cwd())
        logfile_path = str(Path(handler_cwd) / logfile)
        fh = RotatingFileHandler(logfile_path,
                                 maxBytes=int(self.config.get('logging.max_file_size', 10*1024*1024)),
                                 backupCount=int(self.config.get('logging.backup_count', 5)))
        fh.setFormatter(MetadataFormatter("%(asctime)s - %(name)s - %(levelname)s - [%(context)s] - %(message)s"))
        logger.addHandler(fh)

        if getattr(self.handler, 'connected', False):
            logger.addHandler(self.handler)

        # add a console handler in dev
        if os.getenv('ENVIRONMENT', 'production').strip().lower() in {'dev', 'development'}:
            ch = logging.StreamHandler()
            ch.setFormatter(MetadataFormatter("%(levelname)s - [%(context)s] - %(message)s"))
            logger.addHandler(ch)

        self.loggers[key] = logger
        return logger

    def log_document_processing(self, logger_name: str, context: str, message: str, level: str = 'INFO', * ,
                                file_info: Optional[Dict[str,Any]] = None,
                                processing_stats: Optional[Dict[str,Any]] = None,
                                session_id: Optional[str] = None,
                                extra_fields: Optional[Dict[str,Any]] = None):
        logger = self.get_logger(logger_name, context)
        payload = {
            'dp_context': context,
            'dp_file_info': file_info or {},
            'dp_processing_stats': processing_stats or {},
            'dp_session_id': session_id,
            'dp_extra_fields': extra_fields or {}
        }
        levelno = getattr(logging, level.upper(), logging.INFO)
        logger.log(levelno, message, extra=payload)


# Module-level factory accessor
_FACTORY: Optional[DocumentProcessingLogger] = None

def get_logger_factory(config_file: str = 'logging_config.json') -> DocumentProcessingLogger:
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = DocumentProcessingLogger(config_file)
    return _FACTORY


# -----------------------------
# Helpers: load local JSONL and Mongo query results into pandas DataFrame
# -----------------------------
def _safe_parse_jsonl(path: Path):
    try:
        # if path is relative, resolve against cwd
        if not path.is_absolute():
            path = Path.cwd() / path
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue
    except FileNotFoundError:
        return


def load_local_logs_as_dataframe(jsonl_path: str = 'local_processing_logs.jsonl'):
    """Return a pandas DataFrame built from the local JSONL log file.

    If pandas is not available, returns a list of dicts.
    """
    path = Path(jsonl_path)
    records = list(_safe_parse_jsonl(path))
    if HAS_PANDAS:
        try:
            df = pd.json_normalize(records)
            # Prefer readable timestamp column
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    pass
            return df
        except Exception:
            return records
    return records


def load_mongo_logs_as_dataframe(factory: DocumentProcessingLogger, query: Optional[Dict[str, Any]] = None, limit: int = 1000):
    """Query Mongo (if available) and return a pandas DataFrame or list of dicts.

    Provide a `DocumentProcessingLogger` instance (from get_logger_factory()) so we can use its handler.
    """
    if not hasattr(factory, 'handler') or not getattr(factory.handler, 'connected', False):
        return []
    coll = factory.handler.collection
    q = query or {}
    try:
        docs = list(coll.find(q).sort([('timestamp', -1)]).limit(int(limit)))
        # convert ObjectId and datetimes
        for d in docs:
            if '_id' in d:
                d['_id'] = str(d['_id'])
        if HAS_PANDAS:
            try:
                df = pd.json_normalize(docs)
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception:
                        pass
                return df
            except Exception:
                return docs
        return docs
    except Exception:
        return []

