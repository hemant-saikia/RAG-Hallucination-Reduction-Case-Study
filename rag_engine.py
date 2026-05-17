from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
import config

PROMPT_v1 = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based solely on the provided context.

Context:
{context}

Instructions:
- Answer ONLY using information from the context above
- Quote specific phrases from the context using "..." 
- If the context doesn't contain enough information, say "I don't have enough information to answer this confidently"
- Never invent facts not present in the context

Question: {question}
Answer:"""

def chunk_document(text: str, chunk_size: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


def get_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name="rag_docs",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_and_store(chunks: List[str], doc_name: str, collection):
    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_name": doc_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)


def retrieve_context(query: str, collection, k: int = None) -> List[Dict]:
    if k is None:
        k = config.TOP_K_CHUNKS
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    chunks_with_scores = []
    if results["documents"] and results["distances"]:
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            chunks_with_scores.append({
                "text": results["documents"][0][i],
                "similarity_score": round(similarity, 4),
                "id": results["ids"][0][i] if results["ids"] else f"chunk_{i}"
            })
    return chunks_with_scores


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content
