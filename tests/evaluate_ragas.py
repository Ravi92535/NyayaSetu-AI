#!/usr/bin/env python3
"""
evaluate_ragas.py
──────────────────────────────────────────────────────────────────────────────
Ragas evaluation runner for NyayaSetu AI (rag-project).

Evaluates the RAG pipeline using the Ragas framework on:
  1. Faithfulness      : Is the answer grounded in the retrieved legal context?
  2. Answer Relevancy  : Does the answer address the user's question?
  3. Context Precision : Are the most relevant chunks ranked at the top?
  4. Context Recall    : Does the retrieved context contain the ground truth?

Usage:
  python tests/evaluate_ragas.py --samples 200 # full 200-sample benchmark
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from src.config import settings
from src.generation import RAGPipeline
from src.retrieval import VectorStoreManager, get_embedding_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation on NyayaSetu AI golden dataset."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of questions to evaluate (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("⚖️  NyayaSetu AI — Ragas Evaluation Benchmark")
    print("=" * 70)

    # ── 1. Load Golden Dataset ────────────────────────────────────────────────
    dataset_path = PROJECT_ROOT / "eval_dataset" / "dataset.json"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT.parent / "eval_dataset" / "dataset.json"
    if not dataset_path.exists():
        print(f"❌ Golden dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    all_samples = raw.get("dataset", [])
    samples = all_samples[: args.samples]
    print(f"📖 Loaded {len(samples)} samples from {dataset_path.name}")

    # ── 2. Connect to Weaviate (reuse pre-indexed LegalStatute collection) ────
    print("\n🔍 Connecting to Weaviate 'LegalStatute' collection...")
    vector_store = VectorStoreManager()
    count = vector_store.count()
    print(f"   ✅ {count} sections already indexed — skipping re-ingestion.")
    pipeline = RAGPipeline(vector_store=vector_store)

    # ── 3. Generate answers from the RAG pipeline ─────────────────────────────
    print(f"\n🚀 Generating RAG answers for {len(samples)} questions (Top-K = 6)...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, s in enumerate(samples, 1):
        question = s["input"]
        ground_truth = s.get("expected_output", "")

        rag_result = pipeline.query(question, top_k=6)
        retrieved_docs = vector_store.similarity_search(question, k=6)

        questions.append(question)
        answers.append(rag_result["answer"])
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(ground_truth)

        print(f"  [{i:>2}/{len(samples)}] ✓  {question[:65]}...")

    # ── 4. Build Ragas-compatible HuggingFace Dataset ────────────────────────
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ── 5. Init Ragas evaluator (Groq via OpenAI-compatible endpoint) ─────────
    print("\n🤖 Initializing Ragas evaluator LLM & embeddings...")
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.GROQ_API_KEY,
    )
    evaluator_llm = llm_factory(model=settings.LLM_MODEL, client=groq_client)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    print(f"   ✅ Evaluator LLM : {settings.LLM_MODEL} (via Groq)")
    print(f"   ✅ Embeddings    : {settings.EMBEDDING_MODEL}")

    # ── 6. Run Ragas evaluation ───────────────────────────────────────────────
    # faithfulness + answer_relevancy work reliably with Groq free-tier rate limits.
    # context_precision / context_recall need OpenAI-tier rate limits (3+ parallel calls).
    metrics = [faithfulness, answer_relevancy]
    print(f"\n⏳ Running Ragas ({len(metrics)} metrics × {len(samples)} samples)...")

    # max_workers=1 → sequential calls, avoids Groq free-tier rate-limit errors
    run_cfg = RunConfig(max_workers=1, max_retries=3, max_wait=30, timeout=120)

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_cfg,
        show_progress=True,
        raise_exceptions=False,
    )

    # ── 7. Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊  RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Samples evaluated : {len(samples)}")
    print(f"  LLM used          : {settings.LLM_MODEL}")
    print("-" * 70)

    # Compute mean for each metric from results.scores
    metric_scores = {}
    for m in metrics:
        name = getattr(m, "name", str(m))
        scores_list = [
            s[name]
            for s in results.scores
            if name in s
            and s[name] is not None
            and not (isinstance(s[name], float) and s[name] != s[name])
        ]
        mean_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        metric_scores[name] = mean_score
        bar_len = int(mean_score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  🎯 {name:<22}  {mean_score * 100:5.1f}%  [{bar}]")
    print("=" * 70)

    # ── 8. Save JSON report ───────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "eval_dataset" / "ragas_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": settings.LLM_MODEL,
        "samples_evaluated": len(samples),
        "metrics": {k: round(float(v), 4) for k, v in metric_scores.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Saved report → {out_path.relative_to(PROJECT_ROOT)}")

    # Also save detailed pandas dataframe if available
    try:
        df = results.to_pandas()
        csv_path = PROJECT_ROOT / "eval_dataset" / "ragas_eval_details.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved per-question breakdown → {csv_path.relative_to(PROJECT_ROOT)}")
    except Exception:
        pass

    vector_store.close()
    print("\n✅ Ragas evaluation complete!\n")


if __name__ == "__main__":
    main()
