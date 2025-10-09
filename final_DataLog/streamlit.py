import streamlit as st
import joblib
import openai
from ml_predict import auto_select_model, predict_with_confidence, hybrid_predict, MODEL_PATHS, DynamicScrubber, classify_text
from MongoDB_1 import get_logger_factory, load_local_logs_as_dataframe, load_mongo_logs_as_dataframe
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
import os
import signal

LOCAL_LOG_PATH = Path.cwd() / "local_processing_logs.jsonl"
TRAIN_LOG_PATH = Path.cwd() / "training.log"
TRAIN_PID_PATH = Path.cwd() / "training.pid"
STOP_FLAG_PATH = Path.cwd() / "training.stop"


def _fetch_recent_local_logs(limit: int = 100):
    if not LOCAL_LOG_PATH.exists():
        return []
    out = []
    try:
        with open(LOCAL_LOG_PATH, "r", encoding="utf-8") as f:
            for line in reversed(list(f)):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except Exception:
        return []
    return out


def _fetch_recent_logs(limit: int = 100):
    """Try Mongo first then fallback to local JSONL."""
    try:
        factory = get_logger_factory()
        mh = getattr(factory, "mongo_handler", None)
        if mh and getattr(mh, "connected", False) and getattr(mh, "collection", None) is not None:
            docs = list(mh.collection.find({}).sort("timestamp", -1).limit(limit))
            out = []
            for d in docs:
                d2 = dict(d)
                d2["_id"] = str(d2.get("_id"))
                if isinstance(d2.get("timestamp"), datetime):
                    d2["timestamp"] = d2["timestamp"].astimezone(timezone.utc).isoformat()
                out.append(d2)
            return out
    except Exception:
        pass
    return _fetch_recent_local_logs(limit)


def render_live_logs(limit: int = 50):
    st.sidebar.header("Live processing logs")
    auto = st.sidebar.checkbox("Auto-refresh logs (5s)", value=False)
    refresh_seconds = 5
    # allow user to change how many lines of the training log to see
    tail_lines = st.sidebar.number_input("Training log lines", min_value=10, max_value=2000, value=200, step=10)
    if st.sidebar.button("Refresh logs") or auto:
        # force refresh
        logs = _fetch_recent_logs(limit)
        st.session_state["recent_logs"] = logs

    logs = st.session_state.get("recent_logs")
    if logs is None:
        logs = _fetch_recent_logs(limit)
        st.session_state["recent_logs"] = logs

    for i, entry in enumerate(logs[:limit]):
        with st.sidebar.expander(f"{entry.get('timestamp', '')} - {entry.get('context', '')}", expanded=(i < 3)):
            st.json(entry)

    # Show a rolling view of the training log if present
    st.sidebar.markdown("---")
    st.sidebar.subheader("Training log")
    def tail_file(path: Path, lines: int = 200) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                end = f.tell()
                if end == 0:
                    return ""
                size = 1024
                data = b""
                while len(data.splitlines()) <= lines and end > 0:
                    read_size = min(size, end)
                    f.seek(end - read_size)
                    chunk = f.read(read_size)
                    data = chunk + data
                    end -= read_size
                    size *= 2
                text = data.decode("utf-8", errors="replace")
                return "\n".join(text.splitlines()[-lines:])
        except Exception:
            return ""

    if TRAIN_LOG_PATH.exists():
        tail_text = tail_file(TRAIN_LOG_PATH, tail_lines)
        st.sidebar.code(tail_text or "(no training output yet)", language="bash")

    # If auto-refresh requested, wait a bit and rerun to update the sidebar
    if auto:
        time.sleep(refresh_seconds)
        st.experimental_rerun()


def render_main_log_viewer():
    """Main-area live log viewer with filtering, search and download."""
    st.header("Live log viewer")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        source = st.selectbox("Log source", options=["Auto (Mongo or local)", "Local JSONL", "MongoDB", "Training log"], index=0)
    with col2:
        lines = st.number_input("Lines / items", min_value=10, max_value=2000, value=200, step=10)
    with col3:
        auto = st.checkbox("Auto-refresh", value=False)

    # Fetch base logs depending on source
    logs = []
    if source == "Training log":
        # Show training log tail only
        def tail_file(path: Path, lines: int = 200) -> str:
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    end = f.tell()
                    if end == 0:
                        return ""
                    size = 1024
                    data = b""
                    while len(data.splitlines()) <= lines and end > 0:
                        read_size = min(size, end)
                        f.seek(end - read_size)
                        chunk = f.read(read_size)
                        data = chunk + data
                        end -= read_size
                        size *= 2
                    text = data.decode("utf-8", errors="replace")
                    return "\n".join(text.splitlines()[-lines:])
            except Exception:
                return ""

        tail_text = tail_file(TRAIN_LOG_PATH, lines)
        st.code(tail_text or "(no training output yet)", language="bash")
        if st.button("Download training log"):
            try:
                with open(TRAIN_LOG_PATH, "rb") as f:
                    data = f.read()
                st.download_button("Download training.log", data=data, file_name=TRAIN_LOG_PATH.name, mime="text/plain")
            except Exception as e:
                st.error(f"Failed to read training log: {e}")
        if auto:
            time.sleep(3)
            st.experimental_rerun()
        return

    # For JSON logs (local or mongo/auto) fetch entries
    if source == "Local JSONL":
        logs = _fetch_recent_local_logs(lines)
    elif source == "MongoDB":
        logs = _fetch_recent_logs(lines)
    else:
        logs = _fetch_recent_logs(lines)

    # Extract available levels and contexts
    levels = sorted({(l.get("level") or "INFO") for l in logs}) if logs else []
    contexts = sorted({(l.get("context") or "") for l in logs}) if logs else []

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        level_filter = st.selectbox("Level", options=[None] + levels, index=0)
    with fcol2:
        context_filter = st.selectbox("Context", options=[None] + contexts, index=0)
    with fcol3:
        q = st.text_input("Search (full text)")

    # Filter logs
    def match_entry(entry: dict) -> bool:
        if level_filter and (entry.get("level") != level_filter):
            return False
        if context_filter and (entry.get("context") != context_filter):
            return False
        if q:
            s = json.dumps(entry, ensure_ascii=False)
            if q.lower() not in s.lower():
                return False
        return True

    filtered = [e for e in logs if match_entry(e)]

    st.markdown(f"**Showing {len(filtered)} of {len(logs)} entries**")

    # Download filtered view
    try:
        import io

        if filtered:
            buf = io.BytesIO()
            for e in filtered:
                buf.write((json.dumps(e, ensure_ascii=False) + "\n").encode("utf-8"))
            buf.seek(0)
            st.download_button("Download filtered logs (JSONL)", data=buf, file_name="filtered_logs.jsonl", mime="application/json")
    except Exception:
        pass

    # Render entries
    for i, entry in enumerate(filtered):
        with st.expander(f"{entry.get('timestamp', '')} - {entry.get('context', '')} - {entry.get('level','')}", expanded=(i < 3)):
            st.json(entry)

    if auto:
        time.sleep(3)
        st.experimental_rerun()


def _terminate_process(pid: int) -> bool:
    """Attempt to terminate a process and its children. Return True on success."""
    try:
        # Prefer psutil if available for robust child termination
        import psutil

        p = psutil.Process(pid)
        # kill children first
        for child in p.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        p.kill()
        return True
    except Exception:
        # Fallbacks
        try:
            if os.name == "nt":
                # Windows fallback using taskkill
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
            else:
                os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False


def stop_training_if_requested():
    """If the user clicked Stop training in the sidebar, try to kill the training process."""
    if st.sidebar.button("Stop training"):
        # Instead of forcibly killing processes from the UI, create a stop-flag file
        # that the trainer checks periodically. This allows for graceful shutdown.
        try:
            STOP_FLAG_PATH.write_text("stop\n", encoding="utf-8")
            st.sidebar.success("Stop flag created; trainer will attempt a graceful shutdown.")
        except Exception as e:
            st.sidebar.error(f"Failed to create stop flag: {e}")
        # Also attempt to signal the PID as a best-effort fallback (non-fatal)
        pid = None
        try:
            if TRAIN_PID_PATH.exists():
                pid = int(TRAIN_PID_PATH.read_text().strip())
        except Exception:
            pid = None
        if pid:
            try:
                _terminate_process(pid)
            except Exception:
                pass


st.set_page_config(page_title="Hybrid Sensitivity Classifier", page_icon="🔐", layout='wide')

# Artistic styling for a more professional, polished UI
st.markdown(
        """
        <style>
            /* Page gradient background */
            .stApp {
                background: linear-gradient(180deg, #0f172a 0%, #00263b 40%, #071f3a 100%);
                color: #e6f0f6;
            }
            /* Card like panels */
            .card {
                background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.04);
                box-shadow: 0 4px 20px rgba(2,6,23,0.6);
                margin-bottom: 1rem;
            }
            .panel-header { color: #a5f3fc; font-weight: 700; font-size: 1.05rem; margin-bottom:0.5rem }
            .hero { padding: 1rem; border-radius: 12px; margin-bottom: 1rem; }
            .hero-title { font-size: 1.6rem; font-weight: 800; color: #ffffff; margin: 0 }
            .hero-sub { color: #cfeff6; margin-top: 0.25rem }
            /* Buttons */
            .stButton>button {
                background: linear-gradient(90deg,#4ade80,#06b6d4);
                color: #042027;
                border: none;
                padding: 0.6rem 1rem;
                font-weight: 700;
            }
            /* Sidebar tweaks */
            .css-1d391kg { background: rgba(255,255,255,0.02); }
        </style>
        """,
        unsafe_allow_html=True,
)

# Hero header
st.markdown("""
<div class='hero card'>
    <div style='display:flex;align-items:center;gap:1rem'>
        <div style='font-size:2.2rem;'>🔐</div>
        <div>
            <div class='hero-title'>Hybrid Sensitivity Classifier</div>
            <div class='hero-sub'>Anonymize, classify, and audit prompts with placeholder preservation and structured logs.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Right-side floating audit panel (optional)
def _render_audit_panel(limit: int = 30, source: str = 'Local JSONL'):
    # Build a compact HTML table showing timestamp, context, classification, confidence, session_id, snippet
    try:
        if source == 'MongoDB':
            factory = get_logger_factory()
            df = load_mongo_logs_as_dataframe(factory, limit=limit)
        else:
            df = load_local_logs_as_dataframe('local_processing_logs.jsonl')
        # If pandas DataFrame, convert to list of dicts for compact rendering
        rows = df.to_dict(orient='records') if hasattr(df, 'to_dict') else (df or [])
    except Exception:
        rows = []

    # Build table rows
    html_rows = []
    for i, r in enumerate(rows[:limit] if isinstance(rows, list) else []):
        ts = r.get('timestamp', '')
        ctx = r.get('context', '')
        cls = r.get('classification', r.get('model_info.classification', ''))
        conf = r.get('model_confidence', '')
        sid = r.get('session_id', '')
        snippet = ''
        # Robust shortening helper for various value types (avoid len() on non-strings)
        def _shorten(val, limit=160):
            if val is None:
                return ''
            if isinstance(val, str):
                return (val[:limit] + '...') if len(val) > limit else val
            try:
                # convert numbers, booleans, and simple types to string
                s = str(val)
                return (s[:limit] + '...') if len(s) > limit else s
            except Exception:
                return ''

        if 'descrubbed_text' in r and r.get('descrubbed_text') is not None:
            snippet = _shorten(r.get('descrubbed_text'), 160)
        elif 'scrubbed_text' in r and r.get('scrubbed_text') is not None:
            snippet = _shorten(r.get('scrubbed_text'), 160)
        else:
            snippet = _shorten(r.get('message', ''), 120)
        # create a stable audit id: prefer Mongo _id when present, otherwise synthesize
        aid = None
        if isinstance(r, dict) and r.get('_id'):
            aid = str(r.get('_id'))
        else:
            # local logs may not have _id; create a reproducible id using index+timestamp
            aid = f"local-{i}-{urllib.parse.quote_plus(str(ts))}"
            # stash it so we can find it later
            r['_audit_id'] = aid

        view_href = f"?audit_id={urllib.parse.quote_plus(aid)}"
        view_link = f"<a href='{view_href}' style='color:#7dd3fc;text-decoration:none;font-weight:700;'>View</a>"
        html_rows.append(f"<tr><td>{ts}</td><td>{ctx}</td><td>{cls}</td><td>{conf}</td><td>{sid}</td><td>{snippet}</td><td>{view_link}</td></tr>")

        rows_html = "\n".join(html_rows) or "<tr><td colspan='6' style='color:#9fbfc9'>No recent audit entries</td></tr>"
        html_table = f"""
        <div style='position:fixed;right:18px;top:120px;z-index:9999;width:420px;max-height:70vh;overflow:auto;padding:10px;'>
            <div class='card' style='background:rgba(2,6,23,0.7);'>
                <div style='font-weight:700;color:#a5f3fc;margin-bottom:6px;'>Audit — recent events</div>
                <table style='width:100%;font-size:0.8rem;color:#e6f0f6;border-collapse:collapse;'>
                      <thead><tr><th style='text-align:left'>ts</th><th>ctx</th><th>cls</th><th>conf</th><th>sid</th><th>snippet</th><th></th></tr></thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    st.markdown(html_table, unsafe_allow_html=True)

# Sidebar toggle for audit panel
with st.sidebar.expander('Audit panel'):
    show_audit = st.checkbox('Show right audit column', value=True)
    audit_source = st.selectbox('Audit source', options=['Local JSONL', 'MongoDB'], index=0)
    audit_limit = st.number_input('Audit rows', min_value=5, max_value=200, value=30, step=5)

if 'show_audit' in locals() and show_audit:
    _render_audit_panel(limit=audit_limit, source=audit_source)

# If the user clicked a View link, show the full JSON for that audit entry
query_params = st.experimental_get_query_params()
audit_id = None
if 'audit_id' in query_params:
    audit_id = query_params.get('audit_id')[0]

if audit_id:
    # Try to locate the audit entry in Mongo or local logs
    found = None
    try:
        factory = get_logger_factory()
        if factory.handler and getattr(factory.handler, 'connected', False):
            # try Mongo first
            try:
                doc = factory.handler.collection.find_one({'_id': audit_id})
                if doc:
                    doc['_id'] = str(doc['_id'])
                    found = doc
            except Exception:
                found = None
    except Exception:
        found = None

    if not found:
        # search local JSONL for matching _audit_id or synthesized id
        rows = load_local_logs_as_dataframe('local_processing_logs.jsonl')
        if hasattr(rows, 'to_dict'):
            rows = rows.to_dict(orient='records')
        for r in (rows or []):
            if r.get('_audit_id') == audit_id or str(r.get('_id')) == audit_id:
                found = r
                break

    if found:
        with st.expander(f"Audit entry: {audit_id}", expanded=True):
            st.json(found)
            if st.button('Close audit view'):
                st.experimental_set_query_params()

data_dir = st.text_input("Training data directory:", "train_data")

# Simple local logging (JSONL) to record scrub/descrub events when Mongo logging
# from the larger app isn't available.
import json
from datetime import datetime, timezone
LOCAL_LOG_PATH = Path("./local_processing_logs.jsonl")


def _write_local_log(entry: dict) -> None:
    try:
        e = dict(entry)
        if isinstance(e.get("timestamp"), datetime):
            e["timestamp"] = e["timestamp"].astimezone(timezone.utc).isoformat()
        with open(LOCAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_doc_processing_simple(context: str, message: str, extra: dict | None = None):
    try:
        entry = {"timestamp": datetime.now(timezone.utc), "context": context, "message": message}
        if extra:
            entry.update(extra)
        _write_local_log(entry)
    except Exception:
        pass


def create_download_button_for_text(text: str, filename: str = "scrubbed.txt"):
    try:
        import io

        buf = io.BytesIO()
        buf.write(text.encode("utf-8"))
        buf.seek(0)
        return st.download_button("📥 Download scrubbed", data=buf, file_name=filename, mime="text/plain")
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


def demo_playground():
    st.markdown("""
    <style>
      .card { background: #ffffff; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.06); }
      .panel-header { color: #064e3b; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    st.header("Scrub / Fake LLM / Descrub Playground")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>1) Original Prompt</div>", unsafe_allow_html=True)
        upload = st.file_uploader("Upload a text file (.txt)", type=["txt"], key="pg_u1")
        if upload is not None:
            raw = upload.read().decode("utf-8", errors="replace")
            original = st.text_area("Original text", value=raw, height=220, key='original_area')
        else:
            default_text = (
                "Draft a short LinkedIn post promoting the 2024 Annual Report with this link: "
                "https://assets.ing.com/m/6e8ace8ade094690/original/Annual-report-2024.pdf. "
                "Also reference the account IBAN BE71 0961 2345 6769 for the sample transfer."
            )
            original = st.text_area("Original text", height=220, value=default_text, key='original_area')

        btn_l, btn_r = st.columns([1, 1])
        with btn_l:
            if original:
                st.download_button("Download original", data=original, file_name="original.txt", mime="text/plain")
        with btn_r:
            if st.button("Scrub", key="pg_scrub_btn"):
                    use_model = st.session_state.get("use_ml_model", True)
                    try:
                        if use_model:
                            ml_res = classify_text(original, use_model=True)
                            scrubbed = ml_res.get("scrubbed_text", original)
                            placeholder_map = ml_res.get("placeholder_mapping", {})
                            matches = ml_res.get("matches", {})
                            classification = ml_res.get("classification")
                            confidence = ml_res.get("confidence", {})
                        else:
                            ds = DynamicScrubber()
                            scrubbed, placeholder_map = ds.scrub(original)
                            matches = {}
                            classification = None
                            confidence = {}
                    except Exception:
                        scrubbed, placeholder_map = original, {}
                        matches = {}
                        classification = None
                        confidence = {}

                    st.session_state['scrubbed'] = scrubbed
                    st.session_state['scrubbed_area'] = scrubbed
                    st.session_state['placeholder_map'] = placeholder_map
                    st.session_state['scrub_matches'] = matches
                    st.session_state['scrub_classification'] = classification
                    st.session_state['scrub_confidence'] = confidence

                    # Try to log using Mongo factory if available
                    try:
                        factory = get_logger_factory()
                        factory.log_document_processing(
                            logger_name="streamlit_playground",
                            context="scrub",
                            message="User scrubbed a prompt via playground",
                            level="INFO",
                            file_info={"file_type": "prompt"},
                            processing_stats={"processing_time": 0.0},
                            session_id=st.session_state.get("session_id"),
                            extra_fields={
                                "prompt_metadata": {
                                    "original_prompt": original,
                                    "scrubbed_prompt": scrubbed,
                                    "sensitive_entities_detected": list(placeholder_map.keys()),
                                },
                                "classification": classification,
                            },
                        )
                    except Exception:
                        # fallback to local JSONL
                        log_doc_processing_simple("scrub", "User scrubbed a prompt via playground", {"sensitive_entities_detected": list(placeholder_map.keys())})

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>2) Scrubbed Output</div>", unsafe_allow_html=True)
        scrubbed = st.session_state.get('scrubbed', '')
        placeholder_map = st.session_state.get('placeholder_map', {})
        st.write(scrubbed)
        st.subheader("Placeholder map")
        st.json(placeholder_map)
        create_download_button_for_text(scrubbed, "scrubbed.txt")
        st.markdown("</div>", unsafe_allow_html=True)

    # LLM and Descrub
    st.markdown("---")
    st.header("LLM (simulated or OpenAI)")
    if st.button("Call LLM (simulated playground)"):
        # mirror scrubbed text
        s = st.session_state.get('scrubbed', '')
        st.session_state['llm_response_scrubbed'] = s

    if st.button("Descrub playground"):
        ds = DynamicScrubber()
        sc = st.session_state.get('llm_response_scrubbed', '')
        pm = st.session_state.get('placeholder_map', {})
        descrubbed = ds.descrub(sc, pm)
        st.session_state['llm_response_descrubbed'] = descrubbed
        # Auto-classify
        try:
            pred, conf = None, None
            sel_path = Path("model_selection.joblib")
            if sel_path.exists():
                sel = joblib.load(sel_path)
                best_model = sel.get('best_model_type')
                if best_model == 'deepnn':
                    pred, conf = predict_with_confidence(None, descrubbed, model_type='deepnn')
                else:
                    model_file = MODEL_PATHS.get(best_model)
                    if model_file and Path(model_file).exists():
                        model = joblib.load(model_file)
                        pred, conf = predict_with_confidence(model, descrubbed, model_type=best_model)
            elif st.session_state.get('loaded_model') is not None:
                pred, conf = predict_with_confidence(st.session_state.get('loaded_model'), descrubbed)
            else:
                # fallback heuristic
                from ml_predict import heuristic_label
                pred = heuristic_label(descrubbed)
                conf = {pred: 1.0}

            st.session_state['llm_descrub_pred'] = pred
            st.session_state['llm_descrub_conf'] = conf
        except Exception as e:
            st.error(f"Playground classification failed: {e}")

    st.subheader("LLM response (descrubbed)")
    st.write(st.session_state.get('llm_response_descrubbed', ''))
    if st.session_state.get('llm_descrub_pred'):
        st.markdown("**Predicted class:** " + str(st.session_state.get('llm_descrub_pred')))
        st.json(st.session_state.get('llm_descrub_conf', {}))


# Sidebar mode selector — App or Playground
mode = st.sidebar.selectbox("Mode", options=["App", "Playground"], index=0)
if mode == "Playground":
    demo_playground()
    st.stop()

# --- QA: emit test log and table view ---
def _emit_test_log():
    try:
        factory = get_logger_factory()
        extra = {
            "prompt_metadata": {
                "original_prompt": "Test original prompt with IBAN BE71 0961 2345 6769 and name John Doe",
                "scrubbed_text": "Test original prompt with <IBAN_1> and <NAME_1>",
                "descrubbed_text": "Test original prompt with John Doe and BE71...",
                "redacted_text": "[REDACTED]",
            },
            "model_info": {
                "model_name": "unit_test_model",
                "model_version": "0.0.1",
                "performance": {"accuracy": 0.99},
                "confidence": 0.95,
                "classification": "C4",
            },
            "response_metadata": {"generated_by": "ui_test"}
        }
        # ensure this UI emit is persisted unless deliberately marked otherwise
        extra['omit_persistence'] = False
        factory.log_document_processing(
            logger_name="ui_test",
            context="unit_test",
            message="Emit sample test log from Streamlit UI",
            level="INFO",
            file_info={"file_type": "test"},
            processing_stats={"processing_time": 0.0},
            session_id="ui-test-session",
            extra_fields=extra,
        )
        return True
    except Exception:
        return False


with st.expander("QA: Emit test log & view logs table"):
    st.write("Emit a structured test log entry (will write to local JSONL and Mongo if configured)")
    if st.button("Emit test log from UI"):
        ok = _emit_test_log()
        if ok:
            st.success("Test log emitted — check the table below or the Mongo collection.")
        else:
            st.error("Failed to emit test log.")

    # Select source for table view
    table_source = st.selectbox("Table source", options=["Local JSONL", "MongoDB"], index=0)
    table_limit = st.number_input("Table rows", min_value=10, max_value=5000, value=200, step=10)

    df = None
    if table_source == "Local JSONL":
        # read from configured logs dir if present
        factory = get_logger_factory()
        logs_dir = getattr(factory.handler, 'logs_dir', Path.cwd())
        jsonl_path = str(Path(logs_dir) / 'local_processing_logs.jsonl')
        df = load_local_logs_as_dataframe(jsonl_path)
    else:
        factory = get_logger_factory()
        df = load_mongo_logs_as_dataframe(factory, query=None, limit=table_limit)

    if df is None:
        st.info("No logs available for the selected source.")
    else:
        try:
            # If pandas DataFrame, show nicely and offer CSV download
            import pandas as _pd
            if isinstance(df, _pd.DataFrame):
                st.dataframe(df.head(table_limit))
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download table as CSV", data=csv, file_name="logs_table.csv", mime="text/csv")
            else:
                # df is likely a list of dicts
                st.write(f"Showing {len(df)} entries")
                for r in (df[:table_limit] if isinstance(df, list) else []):
                    st.json(r)
        except Exception:
            # best-effort rendering
            if isinstance(df, list):
                st.write(f"Showing {len(df)} entries")
                for r in df[:table_limit]:
                    st.json(r)
            else:
                st.write(df)

# Sidebar: model loader
st.sidebar.header("Model management")
# Load available model files at startup (do not retrain in the UI)
available = {k: v for k, v in MODEL_PATHS.items() if Path(v).exists()}
# Attempt to load model selection if present
SELECTION_PATH = Path("model_selection.joblib")
if SELECTION_PATH.exists():
    try:
        sel = joblib.load(SELECTION_PATH)
        st.session_state.setdefault("model_selection", sel)
    except Exception:
        st.session_state.setdefault("model_selection", None)
else:
    st.session_state.setdefault("model_selection", None)

# Auto-load best model from model_selection.joblib if available
if st.session_state.get("model_selection"):
    sel = st.session_state.get("model_selection")
    best = sel.get("best_model_type")
    if best:
        mpath = Path(MODEL_PATHS.get(best, ""))
        if mpath.exists():
            try:
                raw = joblib.load(mpath)
                if isinstance(raw, dict) and "pipeline" in raw:
                    st.session_state["loaded_model"] = raw["pipeline"]
                else:
                    st.session_state["loaded_model"] = raw
                st.session_state["loaded_model_path"] = str(mpath)
            except Exception:
                # leave unloaded
                pass
sel_key = st.sidebar.selectbox("Choose model to load:", [None] + list(available.keys()))
if st.sidebar.button("Load selected model") and sel_key:
    try:
        raw = joblib.load(available[sel_key])
        if isinstance(raw, dict) and "pipeline" in raw:
            st.session_state["loaded_model"] = raw["pipeline"]
        else:
            st.session_state["loaded_model"] = raw
        st.sidebar.success(f"Loaded: {sel_key}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if "loaded_model" not in st.session_state:
    st.session_state["loaded_model"] = None

# Render live logs in sidebar (from Mongo or local JSONL)
render_live_logs(limit=50)

# --- Single-model loader (joblib) ---
st.sidebar.markdown("### Single model loader")
# Discover joblib files in repo root
joblib_files = [p.name for p in Path('.').glob('*.joblib')] + [p.name for p in Path('.').glob('*_model.joblib')]
joblib_files = sorted(set(joblib_files))
selected_single = st.sidebar.selectbox("Choose a joblib model file:", options=[None] + joblib_files, index=0)
auto_load = st.sidebar.checkbox("Auto-load this model on start", value=False)
if auto_load and selected_single and st.session_state.get('loaded_model_path') is None:
    # Attempt to auto-load at first render
    try:
        mpath = Path(selected_single)
        raw = joblib.load(mpath)
        if isinstance(raw, dict) and 'pipeline' in raw:
            st.session_state['loaded_model'] = raw['pipeline']
        else:
            st.session_state['loaded_model'] = raw
        st.session_state['loaded_model_path'] = str(mpath)
        st.sidebar.success(f"Auto-loaded: {selected_single}")
    except Exception as e:
        st.sidebar.error(f"Auto-load failed: {e}")

if st.sidebar.button("Load this model") and selected_single:
    try:
        mpath = Path(selected_single)
        raw = joblib.load(mpath)
        if isinstance(raw, dict) and 'pipeline' in raw:
            st.session_state['loaded_model'] = raw['pipeline']
        else:
            st.session_state['loaded_model'] = raw
        st.session_state['loaded_model_path'] = str(mpath)
        st.sidebar.success(f"Loaded: {selected_single}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if st.sidebar.button("Unload model"):
    if 'loaded_model' in st.session_state:
        st.session_state['loaded_model'] = None
    if 'loaded_model_path' in st.session_state:
        del st.session_state['loaded_model_path']
    st.sidebar.info("Model unloaded")


# LLM provider settings
st.sidebar.markdown("### LLM settings")
llm_provider = st.sidebar.selectbox("LLM provider", ["Google Gemini", "OpenAI", "Simulated"], index=0)
if llm_provider == "Google Gemini":
    google_key = st.sidebar.text_input("Google API key", type="password")
    if google_key:
        st.session_state["google_api_key"] = google_key
        os.environ['GOOGLE_API_KEY'] = google_key
    google_model = st.sidebar.text_input("Google model", value="gemini-2.0-flash")
elif llm_provider == "OpenAI":
    openai_key = st.sidebar.text_input("OpenAI API key", type="password")
    if openai_key:
        # temporarily store in session state for the running app only
        st.session_state["openai_api_key"] = openai_key
        openai.api_key = openai_key
    openai_model = st.sidebar.selectbox("OpenAI model", ["gpt-4", "gpt-4o", "gpt-3.5-turbo"], index=2)
else:
    # Simulated: no keys required
    openai_model = None
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

if st.button("Train & Auto-Select Model"):
    # Run the training script externally so Streamlit doesn't retrain inline for each user action.
    try:
        import subprocess, sys
        train_script = Path("tools/train_and_select.py")
        if not train_script.exists():
            st.error("Training script not found: tools/train_and_select.py")
        else:
            st.info("Starting background training process. This may take some time—check the terminal.")
            # spawn a new process using the current python executable to run training
            # Redirect stdout/stderr to a rolling training.log so the UI can tail it
            TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Rotate existing training log to avoid overwriting older runs
            try:
                if TRAIN_LOG_PATH.exists() and TRAIN_LOG_PATH.stat().st_size > 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup = TRAIN_LOG_PATH.with_name(f"training_{ts}.log")
                    TRAIN_LOG_PATH.rename(backup)
            except Exception:
                # If rotation fails, proceed and open/append to the existing file
                pass
            # Open the (new) training log in append mode so the child can write to it
            # Remove any existing stop flag before launching
            try:
                if STOP_FLAG_PATH.exists():
                    STOP_FLAG_PATH.unlink()
            except Exception:
                pass

            log_f = open(TRAIN_LOG_PATH, "a", encoding="utf-8")
            # Use Popen so the UI doesn't block; file will be written by the child process
            proc = subprocess.Popen([sys.executable, str(train_script), "--data-dir", data_dir], stdout=log_f, stderr=log_f)
            # store PID for informational purposes
            try:
                TRAIN_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
            except Exception:
                pass
            st.session_state["training_pid"] = proc.pid
            st.success("Training started in background. When finished, reload or use 'Load selected model'.")
    except Exception as e:
        st.error(f"Failed to start training: {e}")

st.divider()
st.header("Prompt & Scrubbing")
prompt_text = st.text_area("Prompt text to send to LLM:", height=150)

# scrubbing controls
scrub_strictness = st.selectbox("Scrubbing strictness:", ["low", "medium", "high"], index=1)
if st.button("Scrub prompt"):
    ds = DynamicScrubber()
    scrubbed, placeholders = ds.scrub(prompt_text)
    st.session_state["scrubbed_prompt"] = scrubbed
    st.session_state["placeholders"] = placeholders

scrubbed = st.session_state.get("scrubbed_prompt", "")
placeholders = st.session_state.get("placeholders", {})

st.subheader("Scrubbed prompt")
st.write(scrubbed)

st.subheader("Placeholder map")
st.json(placeholders)

st.header("LLM interaction")
# OpenAI call (real) - requires API key in sidebar
if st.button("Call LLM (OpenAI)"):
    if not scrubbed:
        st.error("No scrubbed prompt available. Scrub the prompt first.")
    else:
        try:
            # prefer session state key if set
            if st.session_state.get("openai_api_key"):
                openai.api_key = st.session_state.get("openai_api_key")

            # instruct model to preserve placeholders exactly and keep similar word count
            system_msg = (
                "You are a helpful assistant. When producing the user-facing reply, DO NOT modify anonymized placeholders "
                "like <NAME_1>, <EMAIL_2>, or similar tokens. Preserve them exactly as they appear in the input. "
                "Also try to keep the word count of your reply close to the user's input (within +/- 20% if possible). "
                "Return only the assistant reply without extra commentary."
            )

            resp = openai.ChatCompletion.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": scrubbed},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            content = resp["choices"][0]["message"]["content"]
            st.session_state["llm_response_scrubbed"] = content
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")

# simulate LLM response (fallback)
if st.button("Call LLM (simulated)"):
    # Simulated response: mirror the scrubbed prompt, keep placeholders as-is,
    # and aim for similar word count by truncating or repeating as needed.
    words = scrubbed.split()
    target_len = max(1, int(len(words) * 0.95))
    mirrored = " ".join(words[:target_len])
    # ensure placeholders remain present explicitly
    if placeholders:
        # append first 1-2 placeholders if not already in mirrored
        for i, k in enumerate(placeholders.keys()):
            if i > 1:
                break
            if k not in mirrored:
                mirrored += " " + k
    simulated = mirrored
    st.session_state["llm_response_scrubbed"] = simulated

llm_scrubbed = st.session_state.get("llm_response_scrubbed", "")
st.subheader("LLM response (scrubbed)")
st.write(llm_scrubbed)

st.header("Descrub & Reveal")
if st.button("Descrub LLM response"):
    ds = DynamicScrubber()
    descrubbed = ds.descrub(llm_scrubbed, placeholders or {})
    st.session_state["llm_response_descrubbed"] = descrubbed
    # After descrubbing, automatically classify the revealed text and store results
    try:
        pred = None
        conf = None
        sel_path = Path("model_selection.joblib")
        if not sel_path.exists():
            # No selection; try to use any model already loaded into session state
            if st.session_state.get("loaded_model") is not None:
                pipeline = st.session_state["loaded_model"]
                pred, conf = predict_with_confidence(pipeline, descrubbed)
            else:
                st.info("No model selection found and no model loaded. Run 'Train & Auto-Select Model' or load a model in the sidebar.")
        else:
            sel = joblib.load(sel_path)
            best_model = sel.get("best_model_type")
            if best_model is None:
                st.warning("model_selection.joblib missing 'best_model_type'. Please retrain.")
            else:
                # prefer an explicitly loaded model (for classical models)
                if st.session_state.get("loaded_model") is not None and best_model != "deepnn":
                    pipeline = st.session_state["loaded_model"]
                    pred, conf = predict_with_confidence(pipeline, descrubbed, model_type=best_model)
                else:
                    if best_model == "deepnn":
                        pred, conf = predict_with_confidence(None, descrubbed, model_type="deepnn")
                    else:
                        model_file = MODEL_PATHS.get(best_model)
                        if not model_file or not Path(model_file).exists():
                            st.warning(f"Model file for '{best_model}' not found: {model_file}")
                        else:
                            model = joblib.load(model_file)
                            if isinstance(model, dict) and "pipeline" in model:
                                pipeline = model["pipeline"]
                            else:
                                pipeline = model
                            pred, conf = predict_with_confidence(pipeline, descrubbed, model_type=best_model)

        if pred is not None:
            st.session_state["llm_descrub_pred"] = pred
            st.session_state["llm_descrub_conf"] = conf
    except Exception as e:
        st.error(f"Classification after descrub failed: {e}")

st.subheader("LLM response (descrubbed)")
st.write(st.session_state.get("llm_response_descrubbed", ""))

# Show automatic classification result (if available)
if st.session_state.get("llm_descrub_pred"):
    st.markdown("**Automatic classification of descrubbed LLM response**")
    st.write(f"**Predicted class:** {st.session_state.get('llm_descrub_pred')}")
    st.subheader("Confidence / scores")
    st.json(st.session_state.get("llm_descrub_conf", {}))

st.divider()

if st.button("Classify"):
    sel_path = Path("model_selection.joblib")
    if not sel_path.exists():
        st.error("No model selection found. Train models first (use 'Train & Auto-Select Model').")
    else:
        try:
            sel = joblib.load(sel_path)
            best_model = sel.get("best_model_type")
            if best_model is None:
                st.error("model_selection.joblib is missing 'best_model_type'. Please retrain.")
            else:
                # prefer a model explicitly loaded into session state
                if st.session_state.get("loaded_model") is not None and best_model != "deepnn":
                    pipeline = st.session_state["loaded_model"]
                    pred, conf = predict_with_confidence(pipeline, prompt_text, model_type=best_model)
                else:
                    if best_model == "deepnn":
                        pred, conf = predict_with_confidence(None, prompt_text, model_type="deepnn")
                    else:
                        model_file = MODEL_PATHS.get(best_model)
                        if not model_file or not Path(model_file).exists():
                            st.error(f"Model file for '{best_model}' not found: {model_file}")
                            st.stop()
                        model = joblib.load(model_file)
                        if isinstance(model, dict) and "pipeline" in model:
                            pipeline = model["pipeline"]
                        else:
                            pipeline = model
                        pred, conf = predict_with_confidence(pipeline, prompt_text, model_type=best_model)

                st.write(f"**Predicted class:** {pred}")
                st.json(conf)
        except Exception as e:
            st.error(f"Error during classification: {e}")

    # After prompt / scrubbing / descrubbing workflows, show the main log viewer
    render_main_log_viewer()
