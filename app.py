import streamlit as st
import time
import uuid
import re
import json
from datetime import datetime
import os
import glob as glob_mod
import pandas as pd
import plotly.express as px

import config
from rag_engine import chunk_document, get_chroma_collection, embed_and_store, retrieve_context, generate_answer
from observability import (
    init_db, log_query, log_retrieval, log_response, evaluate_hallucination,
    get_metrics_by_similarity_threshold, get_recent_queries, get_hallucination_rate,
    get_all_query_ids, get_query_details
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def is_text_garbage(text: str) -> bool:
    if not text.strip():
        return True
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return True
    pdf_markers = ["endobj", "endstream", "endxref", "startxref"]
    if any(m in text.lower() for m in pdf_markers):
        return True
    return False


def sanitize_text(text: str) -> str:
    text = text.replace("\ufffd", "?")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text


def load_sample_docs(collection) -> int:
    total = 0
    for fpath in sorted(glob_mod.glob(os.path.join(config.DATA_DIR, "*.md"))):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        text = sanitize_text(text)
        chunks = chunk_document(text)
        chunks = [c for c in chunks if not is_text_garbage(c)]
        doc_name = os.path.basename(fpath)
        n = embed_and_store(chunks, doc_name, collection)
        total += n
    return total


st.set_page_config(page_title="RAG Hallucination Baseline", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

init_db()

collection = get_chroma_collection()
if collection.count() == 0:
    with st.spinner("Auto-loading sample documents..."):
        total = load_sample_docs(collection)
        st.session_state.collection_ready = True
        st.session_state.num_chunks = total

st.title("RAG Hallucination Reduction \u2014 Baseline")

tab_chat, tab_metrics, tab_inspector = st.tabs([
    "\U0001f4ac Chat",
    "\U0001f4ca Metrics Dashboard",
    "\U0001f50d Query Inspector"
])

# ──────────────────────────────────────────────
# TAB 1: Chat
# ──────────────────────────────────────────────
with tab_chat:
    st.markdown("Fixed 500-char chunks, no overlap, top-3 retrieval, no threshold filtering.")

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                raw = uploaded_file.read()
                if uploaded_file.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(raw)
                else:
                    text = raw.decode("utf-8", errors="replace")
                text = sanitize_text(text)
                chunks = chunk_document(text)
                chunks = [c for c in chunks if not is_text_garbage(c)]
                if not chunks:
                    st.error("No usable text could be extracted from this file.")
                    st.stop()
                st.session_state.num_chunks = len(chunks)
                embed_and_store(chunks, uploaded_file.name, collection)
                st.session_state.collection_ready = True
                st.success(f"Stored {len(chunks)} chunks from {uploaded_file.name}")

        with st.expander("Sample Data"):
            if st.button("Reload sample documents"):
                with st.spinner("Reloading..."):
                    collection.delete(collection.get()["ids"])
                    total = load_sample_docs(collection)
                    st.session_state.collection_ready = True
                    st.session_state.num_chunks = total
                    st.rerun()

        if st.session_state.collection_ready:
            st.metric("Chunks in DB", st.session_state.num_chunks)

        st.divider()
        st.header("Session Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset DB"):
                collection.delete(collection.get()["ids"])
                st.session_state.messages = []
                st.session_state.collection_ready = False
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(sanitize_text(msg["content"]))
            if "details" in msg:
                with st.expander("Retrieval & Evaluation Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Retrieved Chunks")
                        for i, chunk in enumerate(msg["details"]["chunks"], 1):
                            score = chunk["similarity_score"]
                            if score > 0.7:
                                color = "green"
                            elif score > 0.5:
                                color = "orange"
                            else:
                                color = "red"
                            st.markdown(f"**Chunk {i}** \u2014 Score: **:{color}[{score:.4f}]**")
                            st.code(sanitize_text(chunk["text"][:200]) + "...", language="text")
                    with col2:
                        st.subheader("Hallucination Evaluation")
                        ev = msg["details"]["evaluation"]
                        flag = "\U0001f6a8 Hallucination Detected" if ev["detected"] else "\u2705 No Hallucination"
                        st.markdown(f"**Score:** {ev['score']:.2f}")
                        st.markdown(f"**Status:** {flag}")
                        st.markdown(f"**Reasoning:** {ev['reasoning']}")
                        st.markdown(f"**Chunks Used:** {ev['chunks_used']}")
                        st.markdown(f"**Latency:** {msg['details']['latency_ms']:.0f}ms")

    if prompt := st.chat_input("Ask a question about the documents..."):
        if not st.session_state.collection_ready:
            st.warning("Please upload documents or load sample data first.")
            st.stop()

        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        log_query(query_id, prompt, timestamp)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        t0 = time.time()
        chunks = retrieve_context(prompt, collection)
        log_retrieval(query_id, chunks)
        answer_obj = generate_answer(prompt, chunks)
        answer = answer_obj.answer
        latency_ms = (time.time() - t0) * 1000
        log_response(query_id, answer, latency_ms)

        evaluation = evaluate_hallucination(query_id, prompt, chunks, answer)

        details = {
            "chunks": chunks,
            "evaluation": evaluation,
            "latency_ms": latency_ms
        }

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "details": details
        })

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Retrieval & Evaluation Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Retrieved Chunks")
                    for i, chunk in enumerate(chunks, 1):
                        score = chunk["similarity_score"]
                        if score > 0.7:
                            color = "green"
                        elif score > 0.5:
                            color = "orange"
                        else:
                            color = "red"
                        st.markdown(f"**Chunk {i}** \u2014 Score: **:{color}[{score:.4f}]**")
                        st.code(sanitize_text(chunk["text"][:200]) + "...", language="text")
                with col2:
                    st.subheader("Hallucination Evaluation")
                    ev = evaluation
                    flag = "\U0001f6a8 Hallucination Detected" if ev["detected"] else "\u2705 No Hallucination"
                    st.markdown(f"**Score:** {ev['score']:.2f}")
                    st.markdown(f"**Status:** {flag}")
                    st.markdown(f"**Reasoning:** {ev['reasoning']}")
                    st.markdown(f"**Chunks Used:** {ev['chunks_used']}")
                    st.markdown(f"**Latency:** {latency_ms:.0f}ms")

# ──────────────────────────────────────────────
# TAB 2: Metrics Dashboard
# ──────────────────────────────────────────────
with tab_metrics:
    hallucination_rate = get_hallucination_rate()
    df_metrics = get_metrics_by_similarity_threshold()

    total_queries = len(df_metrics)
    avg_similarity = df_metrics["avg_similarity"].mean() if not df_metrics.empty and "avg_similarity" in df_metrics.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", total_queries)
    col2.metric("Hallucination Rate", f"{hallucination_rate*100:.1f}%")
    col3.metric("Avg Similarity Score", f"{avg_similarity:.3f}")
    col4.metric("Hallucination Events", int(df_metrics["hallucination_detected"].sum()) if not df_metrics.empty else 0)

    st.divider()

    mt1, mt2, mt3, mt4 = st.tabs([
        "Hallucination Rate Over Time",
        "Similarity vs Hallucination",
        "Similarity Score Distribution",
        "Recent Queries"
    ])

    with mt1:
        st.subheader("Hallucination Rate (Last 50 Queries)")
        df_recent = get_recent_queries(limit=50)
        if not df_recent.empty and "hallucination_detected" in df_recent.columns and "timestamp" in df_recent.columns:
            try:
                df_plot = df_recent.assign(timestamp=pd.to_datetime(df_recent["timestamp"]))
            except Exception:
                df_plot = df_recent.assign(timestamp=pd.to_datetime(df_recent["timestamp"], errors="coerce"))
            df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df_plot.empty:
                st.info("No valid timestamp data to plot.")
            else:
                df_plot = df_plot.assign(
                    rolling_rate=df_plot["hallucination_detected"].rolling(5, min_periods=1).mean()
                )
                fig = px.line(
                    df_plot,
                    x="timestamp",
                    y="rolling_rate",
                    title="Rolling Hallucination Rate (window=5)",
                    labels={"rolling_rate": "Hallucination Rate", "timestamp": "Time"}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query data yet. Run some queries to populate metrics.")

    with mt2:
        st.subheader("Similarity Score vs Hallucination Score")
        if not df_metrics.empty and "avg_similarity" in df_metrics.columns:
            fig = px.scatter(
                df_metrics,
                x="avg_similarity",
                y="hallucination_score",
                color="hallucination_detected",
                hover_data=["query_text", "evaluation_reasoning"],
                title="Each dot is a query",
                labels={
                    "avg_similarity": "Avg Retrieved Similarity",
                    "hallucination_score": "Hallucination Score (0-1)",
                    "hallucination_detected": "Detected"
                },
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    with mt3:
        st.subheader("Distribution of Chunk Similarity Scores")
        if not df_metrics.empty and "avg_similarity" in df_metrics.columns:
            fig = px.histogram(
                df_metrics,
                x="avg_similarity",
                nbins=20,
                title="How similar were retrieved chunks on average?",
                labels={"avg_similarity": "Average Similarity Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    with mt4:
        st.subheader("Recent Queries")
        if not df_recent.empty:
            display_cols = ["timestamp", "query_text", "hallucination_score", "hallucination_detected", "total_latency_ms"]
            df_display = df_recent[[c for c in display_cols if c in df_recent.columns]].copy()
            if "hallucination_detected" in df_display.columns:
                df_display = df_display.assign(
                    hallucination_detected=df_display["hallucination_detected"].map(
                        {1: "\U0001f6a8 Yes", 0: "\u2705 No"}
                    )
                )
            if "timestamp" in df_display.columns:
                try:
                    df_display = df_display.assign(
                        timestamp=pd.to_datetime(df_display["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                    )
                except Exception:
                    df_display = df_display.assign(timestamp=df_display["timestamp"].astype(str))
            if "total_latency_ms" in df_display.columns:
                df_display = df_display.assign(
                    **{"Latency (s)": (df_display["total_latency_ms"] / 1000).round(2)}
                )
                df_display = df_display.drop(columns=["total_latency_ms"])
            st.dataframe(df_display, use_container_width=True)

            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="Download All Metrics as CSV",
                data=csv,
                file_name="rag_baseline_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("No query data yet.")

# ──────────────────────────────────────────────
# TAB 3: Query Inspector
# ──────────────────────────────────────────────
with tab_inspector:
    query_ids = get_all_query_ids()

    if not query_ids:
        st.info("No queries found. Run some queries on the Chat tab first.")
    else:
        selected_id = st.selectbox("Select a past query ID", query_ids)

        if selected_id:
            details = get_query_details(selected_id)
            if not details:
                st.error("Query details not found.")
            else:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Query")
                    st.markdown(f"> {details['query_text']}")

                    st.subheader("Answer")
                    st.markdown(details["response_text"])

                    if details.get("total_latency_ms"):
                        st.caption(f"Latency: {details['total_latency_ms']:.0f}ms")

                with col2:
                    st.subheader("Hallucination Evaluation")
                    ev = details.get("evaluation")
                    if ev:
                        flag = "\U0001f6a8 Detected" if ev["hallucination_detected"] else "\u2705 Not Detected"
                        st.metric("Score", f"{ev['hallucination_score']:.2f}")
                        st.metric("Status", flag)
                        st.markdown("**Reasoning:**")
                        st.info(ev["evaluation_reasoning"])
                        if ev.get("chunks_actually_used"):
                            try:
                                used = json.loads(ev["chunks_actually_used"])
                            except (json.JSONDecodeError, TypeError):
                                used = ev["chunks_actually_used"]
                            st.markdown(f"**Chunks cited by LLM:** {used}")
                    else:
                        st.info("No evaluation yet.")

                st.divider()

                st.subheader("Side-by-Side: Chunks vs Answer")
                retrievals = details.get("retrievals", [])
                if retrievals:
                    cols = st.columns(len(retrievals))
                    for i, (col, chunk) in enumerate(zip(cols, retrievals), 1):
                        with col:
                            score = chunk["similarity_score"]
                            if score > 0.7:
                                color = "green"
                            elif score > 0.5:
                                color = "orange"
                            else:
                                color = "red"
                            st.markdown(f"**Chunk {i}** \u2014 :{color}[Score: {score:.4f}]")
                            st.text(chunk["chunk_text"])
                else:
                    st.info("No retrieval data.")

                st.divider()
                st.subheader("Raw Data")
                st.json({
                    "query_id": details["query_id"],
                    "timestamp": details["timestamp"],
                    "query_text": details["query_text"],
                    "response_text": (details["response_text"] or "")[:500],
                    "total_latency_ms": details.get("total_latency_ms"),
                    "num_chunks_retrieved": len(retrievals),
                    "evaluation": {
                        "score": ev["hallucination_score"] if ev else None,
                        "detected": bool(ev["hallucination_detected"]) if ev else None,
                        "reasoning": ev["evaluation_reasoning"] if ev else None
                    }
                })
