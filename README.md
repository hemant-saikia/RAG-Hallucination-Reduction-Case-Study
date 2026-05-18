# RAG Hallucination Reduction: A Data-Driven Case Study

## Problem
A naive RAG implementation with fixed 500-char chunks, top-3 cosine similarity retrieval (no threshold), and a LangChain-style default prompt produced hallucinations in **44.4%** of queries.

## Approach
Built comprehensive observability (SQLite-backed logging of every query, retrieval, response, and LLM-as-judge evaluation) to measure and reduce hallucinations through 4 iterative experiments.

## Results at a Glance

| Experiment | Change | Hallucination Rate | Improvement vs Baseline |
|---|---|---|---|
| **Baseline** | Fixed chunks, no threshold, default prompt | 44.4% | — |
| **Exp 1: Similarity Threshold** | Filter chunks < 0.6 similarity, fetch more candidates | 22.2% | **-50.0%** |
| **Exp 2: Citation Prompt** | Require LLM to cite source chunks, ground answers in context | 10.0% | **-77.5%** |
| **Exp 3: Semantic Chunking** | Sentence/paragraph-aware chunk boundaries instead of fixed 500-char | 20.0% | **-55.0%** |
| **Exp 4: Confidence Scoring** | Structured JSON output with self-reported confidence; low-confidence → "I do not know" | 30.0% | **-32.5%** |

**Best result: 77.5% reduction** in hallucination rate (44.4% → 10.0%) with citation-enforcing prompts.

## Experiment Progression

```
44.4% ──┐  Baseline
        │
22.2% ──┼── Exp 1: Similarity Threshold (-50.0%)
        │
20.0% ──┤     Exp 3: Semantic Chunking (-55.0%)
        │
30.0% ──┤     Exp 4: Confidence Scoring (-32.5%)
        │
10.0% ──┴── Exp 2: Citation Prompt (-77.5%) ★ Best
```

## Detailed Experiment Breakdown

### Baseline
- **Chunking**: Fixed 500-character chunks, no sentence boundary awareness, no overlap
- **Retrieval**: Top-3 by cosine similarity, no threshold filtering (avg similarity: 0.376)
- **Prompt**: LangChain default — no citation or grounding requirements
- **Evaluation**: LLM-as-judge (`llama-3.1-8b-instant`), scoring hallucination likelihood 0–1
- **Result**: 4 of 9 queries hallucinated. Low-quality chunks (similarity < 0.6) retrieved for every single query.

### Exp 1: Similarity Threshold Filtering
- **Change**: Retrieve more candidates, filter out chunks with similarity < 0.6, keep top-3 from filtered set
- **Result**: Hallucination dropped to 22.2%. Avg similarity improved from 0.376 → 0.448.
- **Key finding**: 5 of 18 queries returned zero chunks after filtering (similarity too low), which eliminated hallucination risk entirely for those queries but left the LLM to answer from its own parametric knowledge.

### Exp 2: Citation-Enforcing Prompt
- **Change**: Updated prompt template to require the LLM to ground answers in provided context and cite source quotes. Respond in structured JSON format.
- **Result**: Hallucination dropped to **10.0%** — the single most effective intervention. Only 1 of 10 queries hallucinated.
- **Key finding**: Forcing the model to anchor its response to specific text spans dramatically reduced fabricated details. The model also became more willing to say "I don't have enough information" when context was insufficient.

### Exp 3: Semantic Chunking
- **Change**: Replaced fixed 500-char splitting with sentence/paragraph boundary-aware chunking to preserve semantic coherence.
- **Result**: Hallucination at 20.0%. Avg similarity 0.386 (slightly better than baseline's 0.376).
- **Key finding**: Semantic boundaries improved context quality marginally but did not address the core problem — the LLM still freely added external knowledge. 2 of 10 queries hallucinated (LeNet-5/AlexNet comparison added details not in chunks; legal document analysis question repeated hallucinated phrases).

### Exp 4: Confidence Scoring with Pydantic Enforcement
- **Change**: Structured JSON output with self-reported confidence (0.0–1.0). Pydantic model validates response and forces answer to "I do not know" when confidence < 0.6.
- **Result**: Hallucination at 30.0%. The confidence gate successfully suppressed low-confidence answers, but the LLM's self-reported confidence was poorly calibrated — it remained confident on some hallucinated responses.
- **Key finding**: Self-reported confidence alone is insufficient. It works best combined with citation requirements (Exp 2) and similarity thresholding (Exp 1).

## Key Learnings

1. **Observability is foundational** — 100% of low-quality chunks (similarity < 0.6) were retrieved in the baseline. Without logging, this would have been invisible.
2. **Citation requirements are the highest-leverage intervention** — forcing the LLM to quote and ground answers in context reduced hallucinations by 77.5%.
3. **Similarity thresholding is a strong second** — filtering out irrelevant chunks cut hallucinations in half, but introduced a new failure mode: zero-chunk responses where the LLM falls back to parametric knowledge.
4. **Semantic chunking alone is not enough** — preserving sentence boundaries improved coherence but did not prevent the LLM from adding external facts.
5. **Self-reported confidence is poorly calibrated** — the model was confident on hallucinated answers 30% of the time. Confidence gating needs external verification (e.g., NLI-based entailment checks) to be effective.
6. **Combining interventions is the path forward** — the best production system would layer similarity thresholding + citation prompts + external confidence verification.

## Reproducibility

Each experiment is tracked on a separate git branch. Metrics exported to `exports_*/` directories contain per-query CSV data and JSON summaries.

```bash
# Run baseline
git checkout main
streamlit run app.py

# Run experiments
git checkout experiment-1-similarity-threshold
git checkout experiment-2-citation-prompt-1
git checkout experiment-3-semantic-chunking
git checkout experiment-4-confidence
```

## License

MIT
