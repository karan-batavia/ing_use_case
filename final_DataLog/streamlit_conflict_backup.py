import streamlit as st
import joblib
import openai
from ml_predict import auto_select_model, predict_with_confidence, hybrid_predict, MODEL_PATHS, DynamicScrubber
from pathlib import Path

st.title("🔐 Hybrid Sensitivity Classifier")

data_dir = st.text_input("Training data directory:", "train_data")

# Sidebar: model loader
st.sidebar.header("Model management")
available = {k: v for k, v in MODEL_PATHS.items() if Path(v).exists()}
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

# OpenAI settings
st.sidebar.markdown("### OpenAI settings")
openai_key = st.sidebar.text_input("OpenAI API key", type="password")
if openai_key:
    # temporarily store in session state for the running app only
    st.session_state["openai_api_key"] = openai_key
    openai.api_key = openai_key
openai_model = st.sidebar.selectbox("OpenAI model", ["gpt-4", "gpt-4o", "gpt-3.5-turbo"], index=2)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

if st.button("Train & Auto-Select Model"):
    try:
        best = auto_select_model(data_dir)
        st.success(f"Best model selected: {best}")
    except Exception as e:
        st.error(f"Training failed: {e}")

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
