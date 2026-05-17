import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd
from groq import Groq
import config

JUDGE_PROMPT = """Given these source chunks and this answer, rate hallucination likelihood (0-1). Does the answer include facts NOT present in the chunks? Return JSON: {{score: float, detected: bool, reasoning: str, chunks_used: [1,2,3]}}

SOURCE CHUNKS:
{chunks}

ANSWER:
{answer}

Return only valid JSON."""


def get_db():
    conn = sqlite3.connect(config.OBSERVABILITY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY,
            timestamp TEXT,
            query_text TEXT,
            response_text TEXT,
            total_latency_ms REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            chunk_id TEXT,
            chunk_text TEXT,
            similarity_score REAL,
            chunk_position INTEGER,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT,
            hallucination_score REAL,
            hallucination_detected INTEGER,
            chunks_actually_used TEXT,
            evaluation_reasoning TEXT,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        )
    """)
    conn.commit()
    conn.close()


def log_query(query_id: str, query_text: str, timestamp: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO queries (query_id, timestamp, query_text) VALUES (?, ?, ?)",
        (query_id, timestamp, query_text)
    )
    conn.commit()
    conn.close()


def log_retrieval(query_id: str, chunks_with_scores: List[Dict]):
    conn = get_db()
    for pos, chunk in enumerate(chunks_with_scores, 1):
        conn.execute(
            """INSERT INTO retrievals (query_id, chunk_id, chunk_text, similarity_score, chunk_position)
               VALUES (?, ?, ?, ?, ?)""",
            (query_id, chunk.get("id", f"chunk_{pos}"), chunk["text"],
             chunk["similarity_score"], pos)
        )
    conn.commit()
    conn.close()


def log_response(query_id: str, response_text: str, latency_ms: float):
    conn = get_db()
    conn.execute(
        "UPDATE queries SET response_text = ?, total_latency_ms = ? WHERE query_id = ?",
        (response_text, latency_ms, query_id)
    )
    conn.commit()
    conn.close()


def evaluate_hallucination(query_id: str, query: str, chunks: List[Dict], answer: str) -> Dict:
    chunks_text = "\n---\n".join([f"[Chunk {i+1}] {c['text']}" for i, c in enumerate(chunks)])
    prompt = JUDGE_PROMPT.format(chunks=chunks_text, answer=answer)
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256
    )
    result_text = response.choices[0].message.content
    try:
        result = json.loads(result_text.strip().strip("```").strip("json").strip())
    except json.JSONDecodeError:
        result = {"score": 0.5, "detected": True, "reasoning": "Failed to parse judge response", "chunks_used": []}

    conn = get_db()
    conn.execute(
        """INSERT INTO evaluations (query_id, hallucination_score, hallucination_detected, chunks_actually_used, evaluation_reasoning)
           VALUES (?, ?, ?, ?, ?)""",
        (query_id, result.get("score", 0.5), 1 if result.get("detected", False) else 0,
         json.dumps(result.get("chunks_used", [])), result.get("reasoning", ""))
    )
    conn.commit()
    conn.close()

    return {
        "score": result.get("score", 0.5),
        "detected": result.get("detected", False),
        "reasoning": result.get("reasoning", ""),
        "chunks_used": result.get("chunks_used", [])
    }


def get_hallucination_rate() -> float:
    conn = get_db()
    row = conn.execute(
        "SELECT AVG(hallucination_detected) as rate FROM evaluations"
    ).fetchone()
    conn.close()
    return row["rate"] if row and row["rate"] is not None else 0.0


def get_metrics_by_similarity_threshold() -> pd.DataFrame:
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT
            q.query_id,
            q.timestamp,
            q.query_text,
            q.response_text,
            q.total_latency_ms,
            r.min_similarity,
            r.max_similarity,
            r.avg_similarity,
            e.hallucination_score,
            e.hallucination_detected,
            e.evaluation_reasoning
        FROM queries q
        LEFT JOIN (
            SELECT
                query_id,
                MIN(similarity_score) as min_similarity,
                MAX(similarity_score) as max_similarity,
                AVG(similarity_score) as avg_similarity
            FROM retrievals
            GROUP BY query_id
        ) r ON q.query_id = r.query_id
        LEFT JOIN evaluations e ON q.query_id = e.query_id
        ORDER BY q.timestamp DESC
    """, conn)
    conn.close()
    return df


def get_recent_queries(limit: int = 50) -> pd.DataFrame:
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT
            q.query_id,
            q.timestamp,
            q.query_text,
            q.response_text,
            q.total_latency_ms,
            e.hallucination_score,
            e.hallucination_detected,
            e.evaluation_reasoning
        FROM queries q
        LEFT JOIN evaluations e ON q.query_id = e.query_id
        ORDER BY q.timestamp DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


def get_query_details(query_id: str) -> Dict:
    conn = get_db()
    query = conn.execute(
        "SELECT * FROM queries WHERE query_id = ?", (query_id,)
    ).fetchone()
    if not query:
        conn.close()
        return {}

    retrievals = conn.execute(
        "SELECT * FROM retrievals WHERE query_id = ? ORDER BY chunk_position",
        (query_id,)
    ).fetchall()

    evaluation = conn.execute(
        "SELECT * FROM evaluations WHERE query_id = ?", (query_id,)
    ).fetchone()

    conn.close()

    result = dict(query)
    result["retrievals"] = [dict(r) for r in retrievals]
    result["evaluation"] = dict(evaluation) if evaluation else None
    return result


def get_all_query_ids() -> List[str]:
    conn = get_db()
    rows = conn.execute(
        "SELECT query_id FROM queries ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [row["query_id"] for row in rows]
