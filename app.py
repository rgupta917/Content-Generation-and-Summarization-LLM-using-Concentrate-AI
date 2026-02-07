import os
import time
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from concentrate_client import ConcentrateClient
from prompts import (
    compare_judge_prompt,
    generation_prompt,
    merge_prompt,
    summarization_prompt,
)
from text_utils import chunk_text, safe_filename

# ----------------- App Config -----------------
st.set_page_config(page_title="Concentrate Content Studio", layout="wide")
load_dotenv()

API_KEY = os.getenv("CONCENTRATE_API_KEY", "").strip()

st.title("Concentrate Content Studio")
st.caption(
    "Generate content and summarize long text using Concentrate (multi-model + compare + fallback + streaming)."
)


# ----------------- Helpers -----------------
def require_key() -> None:
    if not API_KEY:
        st.error("Missing CONCENTRATE_API_KEY. Create a .env and set your key.")
        st.stop()


@st.cache_data(ttl=300)
def cached_models(api_base: str) -> List[Dict]:
    c = ConcentrateClient(api_key="DUMMY", api_base=api_base)
    try:
        return c.list_models()
    except Exception:
        return []


@st.cache_data(ttl=300)
def cached_providers(api_base: str) -> List[Dict]:
    c = ConcentrateClient(api_key="DUMMY", api_base=api_base)
    try:
        return c.list_providers()
    except Exception:
        return []


def build_payload(model: str, user_text: str, max_tokens: int, temperature: float) -> Dict:
    return {
        "model": model,
        "input": user_text,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }


def call_with_fallback(
    client: ConcentrateClient,
    payload: Dict,
    fallback_models: List[str],
    retry_on: Tuple[int, ...] = (424, 429, 500),
) -> Tuple[str, List[Dict]]:
    """
    Attempt primary model, then fall back through fallback_models on retryable errors.
    Returns (final_text, attempts_log).
    """

    def extract_text(resp: Dict) -> str:
        out = resp.get("output")
        if isinstance(out, list):
            parts = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            t = c.get("text")
                            if isinstance(t, str) and t:
                                parts.append(t)
            if parts:
                return "".join(parts)

        ot = resp.get("output_text")
        if isinstance(ot, str) and ot:
            return ot

        return ""

    attempts: List[Dict] = []

    chain = [payload["model"]] + [m for m in fallback_models if m and m != payload["model"] and m != "auto"]

    for idx, m in enumerate(chain):
        p = dict(payload)
        p["model"] = m

        resp, stats = client.create_response(p, used_fallback=(idx > 0))
        attempts.append(
            {
                "model": stats.model,
                "status": stats.status_code,
                "latency_ms": stats.latency_ms,
                "used_fallback": stats.used_fallback,
                "error": stats.error,
            }
        )

        if stats.status_code == 200:
            text = extract_text(resp).strip()
            if text:
                return text, attempts
            attempts[-1]["error"] = "200 but could not parse response text"

        if stats.status_code not in retry_on:
            break

    return "", attempts

st.sidebar.header("Connection")
require_key()

api_base = st.sidebar.text_input("API Base", value="https://api.concentrate.ai")
timeout_s = st.sidebar.number_input("Timeout (seconds)", min_value=10, max_value=300, value=60, step=5)

client = ConcentrateClient(api_key=API_KEY, api_base=api_base, timeout_s=timeout_s)

with st.sidebar.expander("Health check", expanded=False):
    if st.button("Ping /v1/responses/health"):
        try:
            st.json(client.health())
        except Exception as e:
            st.error(f"Health check failed: {e}")

st.sidebar.header("Model Selection")

RECOMMENDED_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-1.5-pro",
    "auto",
]

show_all = st.sidebar.checkbox("Show all models (advanced)", value=False)

models = cached_models(api_base) if show_all else []
model_names = list(RECOMMENDED_MODELS)

if show_all and models:
    for m in models:
        mid = m.get("id") or m.get("model")
        if isinstance(mid, str) and mid and mid not in model_names and " " not in mid:
            model_names.append(mid)

# Deduplicate while preserving order
seen = set()
model_names = [m for m in model_names if not (m in seen or seen.add(m))]

selected_model = st.sidebar.selectbox("Model", options=model_names, index=0)

fallback_enabled = st.sidebar.checkbox("Enable fallback chain", value=True)
fallback_options = [m for m in model_names if m != "auto"]
fallback_models = st.sidebar.multiselect(
    "Fallback models (in order)",
    options=fallback_options,
    default=[m for m in ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"] if m in fallback_options],
    help="Used if primary fails (e.g., 424/429/500).",
)
effective_fallback_models = fallback_models if fallback_models else [m for m in fallback_options[:2]]

st.sidebar.header("Generation params")
max_tokens = st.sidebar.slider("Max output tokens", min_value=64, max_value=2000, value=600, step=32)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.4, step=0.1)

st.sidebar.header("Compare mode")
compare_mode = st.sidebar.checkbox("Compare across models", value=False)
compare_models = st.sidebar.multiselect(
    "Compare models (2–3 recommended)",
    options=[m for m in model_names if m != "auto"],
    default=[],
)
judge_best = st.sidebar.checkbox("Judge best output (LLM)", value=True)

streaming = st.sidebar.checkbox(
    "Streaming output",
    value=False,
    help="Streams tokens in UI for single-model runs (not compare).",
)

tab1, tab2 = st.tabs(["✍️ Generate", "🧾 Summarize"])

with tab1:
    st.subheader("Content Generation")

    colA, colB = st.columns(2)
    with colA:
        topic = st.text_area("Topic / input", height=150, placeholder="e.g., Write a product announcement for ...")
        audience = st.text_input("Audience", value="Engineers and PMs")
        tone = st.selectbox("Tone", ["Neutral", "Professional", "Friendly", "Persuasive", "Technical"], index=1)
    with colB:
        format_name = st.selectbox(
            "Format",
            ["Blog post", "Email", "Product announcement", "PRD section", "LinkedIn post"],
            index=1,
        )
        length_hint = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
        run = st.button("Generate", type="primary", use_container_width=True)

    if run:
        if not topic.strip():
            st.warning("Please enter a topic/input.")
        else:
            prompt = generation_prompt(
                topic=topic, audience=audience, tone=tone, format_name=format_name, length_hint=length_hint
            )
            goal = f"Generate {format_name} for {audience} in {tone} tone ({length_hint})."

            if compare_mode and len(compare_models) >= 2:
                st.info("Compare mode enabled: running multiple models...")
                outputs: Dict[str, str] = {}
                attempts_log_all: Dict[str, List[Dict]] = {}

                for m in compare_models[:3]:
                    payload = build_payload(model=m, user_text=prompt, max_tokens=max_tokens, temperature=temperature)
                    text, attempts = call_with_fallback(
                        client,
                        payload,
                        fallback_models=effective_fallback_models if fallback_enabled else [],
                        retry_on=(424, 429, 500),
                    )
                    outputs[m] = text or "[No output]"
                    attempts_log_all[m] = attempts

                st.write("### Outputs")
                for m, txt in outputs.items():
                    with st.expander(f"Model: {m}", expanded=True):
                        st.markdown(txt)

                st.write("### Attempts / Stats")
                st.json(attempts_log_all)

                if judge_best:
                    judge_payload = build_payload(
                        model=selected_model,
                        user_text=compare_judge_prompt(outputs, goal=goal),
                        max_tokens=500,
                        temperature=0.2,
                    )
                    judge_text, judge_attempts = call_with_fallback(
                        client,
                        judge_payload,
                        fallback_models=effective_fallback_models if fallback_enabled else [],
                        retry_on=(424, 429, 500),
                    )
                    st.write("### Best-output Judge")
                    st.markdown(judge_text or "_No judge output_")
                    st.caption("Judge attempts:")
                    st.json(judge_attempts)

            else:
                payload = build_payload(
                    model=selected_model,
                    user_text=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                st.write("### Output")
                placeholder = st.empty()
                meta_placeholder = st.empty()

                if streaming and not compare_mode:
                    text_accum = ""
                    start = time.time()
                    error_hit = None

                    for chunk in client.stream_response_text(payload):
                        if chunk and chunk.lstrip().startswith("[ERROR"):
                            error_hit = chunk
                            break
                        text_accum += chunk
                        placeholder.markdown(text_accum)

                    elapsed_ms = int((time.time() - start) * 1000)

                    if not text_accum.strip() and not error_hit:
                        st.warning("Streaming returned no text. Falling back to non-stream request...")

                        final_text, attempts = call_with_fallback(
                            client,
                            payload,
                            fallback_models=effective_fallback_models if fallback_enabled else [],
                            retry_on=(424, 429, 500),
                        )

                        placeholder.markdown(final_text or "_No output_")
                        meta_placeholder.caption("Attempts:")
                        meta_placeholder.json(attempts)
                        st.stop()

                    if error_hit:
                        st.error(error_hit)
                        st.info("Retrying via fallback chain (non-streaming)...")

                        final_text, attempts = call_with_fallback(
                            client,
                            payload,
                            fallback_models=effective_fallback_models if fallback_enabled else [],
                            retry_on=(424, 429, 500),
                        )

                        placeholder.markdown(final_text or "_No output_")
                        meta_placeholder.caption("Attempts:")
                        meta_placeholder.json(attempts)

                    else:
                        meta_placeholder.caption(f"Streaming complete in {elapsed_ms} ms. Model: {selected_model}")
                        final_text = text_accum.strip()

                        if final_text:
                            fname = safe_filename(format_name + "-" + topic[:40]) + ".md"
                            st.download_button(
                                "Download markdown",
                                data=final_text,
                                file_name=fname,
                                mime="text/markdown",
                            )

                else:
                    final_text, attempts = call_with_fallback(
                        client,
                        payload,
                        fallback_models=effective_fallback_models if fallback_enabled else [],
                        retry_on=(424, 429, 500),
                    )
                    placeholder.markdown(final_text or "_No output_")
                    meta_placeholder.caption("Attempts:")
                    meta_placeholder.json(attempts)

                    if final_text:
                        fname = safe_filename(format_name + "-" + topic[:40]) + ".md"
                        st.download_button(
                            "Download markdown",
                            data=final_text,
                            file_name=fname,
                            mime="text/markdown",
                        )

# ---- Summarize ----
with tab2:
    st.subheader("Summarization")

    style = st.selectbox(
        "Summary style",
        ["Executive", "Technical", "Action-oriented", "Customer-friendly"],
        index=0,
    )
    text = st.text_area("Paste text to summarize", height=220, placeholder="Paste a long document here...")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        do_summarize = st.button("Summarize", type="primary", use_container_width=True)
    with col2:
        max_chars = st.number_input("Chunk size (chars)", min_value=2000, max_value=20000, value=8000, step=1000)
    with col3:
        overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=300, step=50)

    if do_summarize:
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            chunks = chunk_text(text, max_chars=int(max_chars), overlap=int(overlap))
            st.caption(f"Chunked into {len(chunks)} chunk(s).")

            # 1) summarize each chunk
            chunk_summaries: List[str] = []
            chunk_attempts: List[Dict] = []

            for i, ch in enumerate(chunks, start=1):
                prompt = summarization_prompt(ch, style=style)
                payload = build_payload(
                    model=selected_model,
                    user_text=prompt,
                    max_tokens=min(max_tokens, 800),
                    temperature=0.2,
                )
                summary, attempts = call_with_fallback(
                    client,
                    payload,
                    fallback_models=effective_fallback_models if fallback_enabled else [],
                    retry_on=(424, 429, 500),
                )
                chunk_summaries.append(f"## Chunk {i}\n{summary}\n")
                chunk_attempts.append({"chunk": i, "attempts": attempts})

            # 2) merge summaries
            merged_prompt = merge_prompt("\n\n".join(chunk_summaries), style=style)
            merge_payload = build_payload(
                model=selected_model,
                user_text=merged_prompt,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            final_summary, merge_attempts = call_with_fallback(
                client,
                merge_payload,
                fallback_models=effective_fallback_models if fallback_enabled else [],
                retry_on=(424, 429, 500),
            )

            st.write("### Final Summary")
            st.markdown(final_summary or "_No output_")

            with st.expander("Attempts / Stats", expanded=False):
                st.json({"chunk_attempts": chunk_attempts, "merge_attempts": merge_attempts})

            if final_summary:
                fname = safe_filename("summary-" + style) + ".md"
                st.download_button(
                    "Download summary markdown",
                    data=final_summary,
                    file_name=fname,
                    mime="text/markdown",
                )
