import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_BASE = BASE_DIR / "audit_logs"

# Directories for files
ORIGINAL_DIR = AUDIT_BASE / "original"
SCRUBBED_DIR = AUDIT_BASE / "scrubbed"
DESCRUBBED_DIR = AUDIT_BASE / "descrubbed"
PREDICTION_DIR = AUDIT_BASE / "predictions"


# Log files
NDJSON_LOG = AUDIT_BASE / "audit.ndjson"
HTML_LOG = AUDIT_BASE / "audit_log.html"

# ---------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------

def init_audit_env() -> Dict[str, str]:
    """Ensure directories and files exist"""
    AUDIT_BASE.mkdir(parents=True, exist_ok=True)
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SCRUBBED_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    DESCRUBBED_DIR.mkdir(parents=True, exist_ok=True)
    NDJSON_LOG.touch(exist_ok=True)
    HTML_LOG.touch(exist_ok=True)

    print(f"[AUDIT] NDJSON_LOG   : {NDJSON_LOG.resolve()}")
    print(f"[AUDIT] HTML_LOG     : {HTML_LOG.resolve()}")
    print(f"[AUDIT] ORIGINAL_DIR : {ORIGINAL_DIR.resolve()}")
    print(f"[AUDIT] SCRUBBED_DIR : {SCRUBBED_DIR.resolve()}")
    print(f"[AUDIT] DESCRUBBED_DIR: {DESCRUBBED_DIR.resolve()}")
    print(f"[AUDIT] PREDICTIONS_DIR: {PREDICTION_DIR.resolve()}")

    return {
        "ndjson": str(NDJSON_LOG),
        "html": str(HTML_LOG),
        "original_dir": str(ORIGINAL_DIR),
        "scrubbed_dir": str(SCRUBBED_DIR),
        "descrubbed_dir": str(DESCRUBBED_DIR),
        "predictions_dir": str(PREDICTION_DIR),
    }

# ---------------------------------------------------------------------
# DOCUMENT SAVING
# ---------------------------------------------------------------------
def save_document(
    kind: str,
    name: str,
    text: Union[str, bytes],
    ext: str = ".txt",
    filename: Optional[str] = None,
) -> str:
    """Save a document by type and return its path"""
    mapping = {
        "original": ORIGINAL_DIR,
        "scrubbed": SCRUBBED_DIR,
        "prediction": PREDICTION_DIR,
        "descrubbed": DESCRUBBED_DIR,
    }
    directory = mapping.get(kind)
    if directory is None:
        raise ValueError(f"Unknown document type: {kind}")
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_ext = ext if ext.startswith(".") else f".{ext}"

    if filename:
        safe_filename = _sanitize_filename(filename)
        candidate = Path(safe_filename)
        final_ext = candidate.suffix or target_ext
        safe_stem = candidate.stem or _sanitize_filename(name)
        candidate_name = f"{safe_stem}{final_ext}"
        path = directory / candidate_name
        if path.exists():
            candidate_name = f"{safe_stem}_{timestamp}{final_ext}"
            path = directory / candidate_name
    else:
        safe_name = _sanitize_filename(name)
        candidate_name = f"{timestamp}_{safe_name}{target_ext}"
        path = directory / candidate_name

    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(str(text), encoding="utf-8")

    print(f"[AUDIT] Saved {kind} document → {path.name}")
    return str(path)

# ---------------------------------------------------------------------
# DOCUMENT SAVING
# ---------------------------------------------------------------------
def _sanitize_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value or "document"


# ---------------------------------------------------------------------
# CLIENT INFO
# ---------------------------------------------------------------------


def build_client_info(request=None) -> Dict[str, Any]:
    """Collect minimal client info."""
    info = {
        "client_ip": "unknown",
        "user_agent": "unknown",
        "host": socket.gethostname(),
    }
    try:
        if request:
            info["client_ip"] = request.client.host if request.client else "unknown"
            info["user_agent"] = request.headers.get("user-agent", "unknown")
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------
# APPEND EVENT
# ---------------------------------------------------------------------


def append_audit_event(event: dict):
    """Add audit event and rebuild HTML"""
    try:
        NDJSON_LOG.parent.mkdir(parents=True, exist_ok=True)
        event.setdefault("timestamp", datetime.utcnow().isoformat())

        with open(NDJSON_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        print(f"[AUDIT] Logged event: {event.get('action', 'unknown')} ({event.get('user_email', '?')})")

        rebuild_audit_html()
    except Exception as e:
        print(f"[AUDIT ERROR] Failed to append event: {e}")


# ---------------------------------------------------------------------
# UPDATE EVENT
# ---------------------------------------------------------------------

def update_audit_event(session_id: str, updates: Dict[str, Any]) -> bool:
    """Update the most recent event for a session and rebuild HTML."""
    if not NDJSON_LOG.exists():
        return False

    try:
        raw_entries: List[Dict[str, Any]] = []
        with open(NDJSON_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        raw_entries.append(json.loads(line))
                    except Exception:
                        continue

        updated = False
        for entry in reversed(raw_entries):
            if entry.get("session_id") == session_id:
                entry.update(updates)
                entry["updated_at"] = datetime.utcnow().isoformat()
                updated = True
                break

        if not updated:
            return False

        with open(NDJSON_LOG, "w", encoding="utf-8") as f:
            for entry in raw_entries:
                f.write(json.dumps(entry) + "\n")

        rebuild_audit_html(raw_entries)
        return True
    except Exception as e:
        print(f"[AUDIT ERROR] Failed to update event for session {session_id}: {e}")
        return False

# ---------------------------------------------------------------------
# REBUILD HTML
# ---------------------------------------------------------------------

def rebuild_audit_html(entries: Optional[List[Dict[str, Any]]] = None):
    """Rebuild beautiful HTML table with clickable links"""
    try:
        if entries is None:
            entries = []
            with open(NDJSON_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue

        html = [
            "<!doctype html><html><head><meta charset='utf-8'><title>Audit Log</title>",
            "<style>",
            "body{font-family:Inter,system-ui,Arial,sans-serif;background:#fff7f1;color:#1a2433;padding:28px;}",
            "h2{color:#ff6200;margin-bottom:12px;}",
            "table{width:100%;border-collapse:collapse;box-shadow:0 4px 12px rgba(0,0,0,0.1);border-radius:12px;overflow:hidden;}",
            "th,td{padding:10px 14px;text-align:left;}",
            "th{background:#001a72;color:white;font-weight:600;}",
            "tr:nth-child(even){background:#fff3ec;}",
            "tr:hover{background:#ffe0d0;transition:0.2s;}",
            "a{color:#ff6200;text-decoration:none;font-weight:600;}",
            "a:hover{text-decoration:underline;}",
            ".footer{margin-top:20px;color:#59657a;font-size:0.9rem;text-align:center;}",
            "</style></head><body>",
            f"<h2>Audit Log ({len(entries)} entries)</h2>",
            "<table>",
            "<tr><th>Timestamp</th><th>Action</th><th>User</th><th>Original</th><th>Scrubbed</th><th>Prediction</th><th>Descrubbed</th><th>Client IP</th></tr>"
        ]

        for e in reversed(entries[-500:]):  # latest first
            orig = e.get("original_file", "")
            scrb = e.get("scrubbed_file", "")
            pred = e.get("prediction_file", "")
            descrb = e.get("descrubbed_file", "")
            client_ip = (e.get("client", {}) or {}).get("client_ip", "")
            user_email = e.get("user_email", "")
            action = e.get("action", "")
            timestamp = e.get("timestamp", "")

            orig_name = Path(orig).name if orig else ""
            scrb_name = Path(scrb).name if scrb else ""
            descrb_name = Path(descrb).name if descrb else ""
            pred_name = Path(pred).name if pred else ""

            orig_link = f"<a href='/audit/file/original/{orig_name}' target='_blank'>{orig_name}</a>" if orig_name else ""
            scrb_link = f"<a href='/audit/file/scrubbed/{scrb_name}' target='_blank'>{scrb_name}</a>" if scrb_name else ""
            pred_link = f"<a href='/audit/file/prediction/{pred_name}' target='_blank'>{pred_name}</a>" if pred_name else ""
            descrb_link = f"<a href='/audit/file/descrubbed/{descrb_name}' target='_blank'>{descrb_name}</a>" if descrb_name else ""

            html.append(
                "<tr>"
                f"<td>{timestamp}</td>"
                f"<td>{action}</td>"
                f"<td>{user_email}</td>"
                f"<td>{orig_link}</td>"
                f"<td>{scrb_link}</td>"

                f"<td>{pred_link}</td>"
                f"<td>{descrb_link}</td>"
                f"<td>{client_ip}</td>"
                "</tr>"
            )

        html.append("</table>")
        html.append("<p class='footer'>© ING – Audit Log • Click filenames to view contents</p>")
        html.append("</body></html>")

        HTML_LOG.write_text("\n".join(html), encoding="utf-8")
        print(f"[AUDIT] HTML log rebuilt ({len(entries)} entries)")

    except Exception as e:
        print(f"[AUDIT HTML ERROR] {e}")

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def unique_doc_name(base: str) -> str:
    """Generate a unique document name"""
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ---------------------------------------------------------------------
# AUTO INIT
# ---------------------------------------------------------------------

init_audit_env()
