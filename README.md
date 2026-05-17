# RAG Hallucination Reduction: Case Study Baseline

**This is the BASELINE with known issues. Improvement experiments tracked in separate branches.**

## Problem Statement

Retrieval-Augmented Generation (RAG) systems can hallucinate — generating facts not present in the retrieved context. This baseline systematically measures hallucination rates using a deliberately naive implementation, providing a reference point for measuring improvement.

## Baseline Approach

- **Chunking**: Fixed 500-character chunks, no sentence boundary awareness, no overlap
- **Retrieval**: Top-3 chunks by cosine similarity, no threshold filtering
- **Generation**: LangChain default prompt, `llama-3.1-70b-versatile` via Groq
- **Evaluation**: LLM-as-judge using `llama-3.1-8b-instant`, scoring hallucination likelihood 0-1
- **Observability**: Every query, retrieval, and evaluation logged to SQLite

Expected baseline hallucination rate: **30-40%**

## How to Run

1. Clone this repository
2. Copy `.env.example` to `.env` and set your `GROQ_API_KEY`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```
5. Sample documents auto-load on first run. Navigate via the three tabs at the top:
   - **Chat** — ask questions, view chunks and hallucination evaluations
   - **Metrics Dashboard** — overall stats, charts, CSV export
   - **Query Inspector** — deep-dive into individual query details
6. Ask questions and observe hallucination patterns

## Next Steps

This baseline is designed for systematic improvement. Future work:
- Semantic chunking (sentence/paragraph boundaries)
- Dynamic chunk sizes with overlap
- Similarity threshold filtering (>0.7)
- Reranking (Cohere, Cross-Encoder)
- Query rewriting
- Hybrid search (sparse + dense)
- Improved prompts with citations

## License

MIT
# Termonitor
