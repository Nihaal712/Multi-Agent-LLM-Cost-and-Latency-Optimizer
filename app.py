"""
Streamlit UI for the LLM Cost & Latency Optimizer.

Layout:
- Left: Query, SLA, Budget, Run button
- Right: Answer
- Below: Completed status + results
"""

import asyncio
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

import config
from main import orchestrate

load_dotenv()


def _run_async(coro):
    """Safely run async coroutine from Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def main():
    st.set_page_config(
        page_title="LLM Cost & Latency Optimizer",
        page_icon="⚡",
        layout="wide",
    )

    st.title("LLM Cost & Latency Optimizer")
    st.caption("Deterministic cost/latency-aware RAG orchestration (temperature=0).")

    # =====================================================
    # TOP SECTION: INPUTS (LEFT) + ANSWER (RIGHT)
    # =====================================================
    left, right = st.columns([1, 2])

    with left:
        query = st.text_area(
            "Query",
            placeholder="Ask a question about AI, quantum, etc.",
            height=120,
        )

        sla = st.selectbox(
            "SLA",
            options=["fast", "balanced", "relaxed"],
            index=["fast", "balanced", "relaxed"].index(config.DEFAULTS["sla"]),
        )

        budget_text = st.text_input(
            "Budget (USD)",
            placeholder="e.g., 0.01",
        )

        run = st.button("Run Orchestration", type="primary")

    with right:
        st.subheader("Answer")
        answer_placeholder = st.empty()

    # =====================================================
    # RUN ORCHESTRATION
    # =====================================================
    if not run:
        return

    if not query.strip():
        st.warning("Please enter a query.")
        return

    # Parse budget
    budget: Optional[float] = None
    if budget_text.strip():
        try:
            budget = float(budget_text.strip())
        except ValueError:
            st.error("Budget must be a number (e.g., 0.01).")
            return

    with st.spinner("Running orchestration..."):
        try:
            state = _run_async(
                orchestrate(query=query, budget=budget, sla=sla)
            )
        except Exception as e:
            st.error(f"Orchestration failed: {e}")
            return

    # Fill answer
    answer_placeholder.write(state.get("answer", "No answer returned."))

    # =====================================================
    # STATUS
    # =====================================================
    st.markdown("### ✅ Completed")

    decisions = state.get("decisions", {})
    estimates = state.get("estimates", {})

    # =====================================================
    # EXECUTION DECISIONS
    # =====================================================
    st.subheader("Execution Decisions")
    d1, d2, d3 = st.columns(3)
    d1.metric("Model Tier", decisions.get("model_tier", "N/A"))
    d2.metric("Context Size (tokens)", decisions.get("context_size", "N/A"))
    d3.metric("Retrieval Depth", decisions.get("retrieval_depth", "N/A"))

    # =====================================================
    # METRICS
    # =====================================================
    st.subheader("Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens Used", state.get("tokens", 0))
    m2.metric("Estimated Cost (USD)", f"{estimates.get('est_cost', 0):.6f}")
    m3.metric("Estimated Latency (s)", f"{estimates.get('est_latency', 0):.2f}")

    # =====================================================
    # TRADE-OFFS & EXPLANATIONS
    # =====================================================
    st.subheader("Trade-offs & Explanations")

    complexity_expl = state.get("complexity", {}).get("output", {}).get("explanation")
    cost_expl = state.get("cost", {}).get("output", {}).get("explanation")
    latency_expl = state.get("latency", {}).get("output", {}).get("explanation")

    st.write(
        (complexity_expl or "N/A")
        + "\n\n"
        + (cost_expl or "N/A")
        + "\n\n"
        + (latency_expl or "N/A")
    )

    st.info(
        "Notes: Deterministic settings (temperature=0). "
        "Ensure GROQ_API_KEY is set. "
        "Sample corpus uses FAISS with CPU embeddings."
    )


if __name__ == "__main__":
    main()