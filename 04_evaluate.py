"""
Step 4: Run benchmark (36 grounded questions). Compare Flat RAG vs GraphRAG.
Results saved to results/comparison.csv
Usage: python 04_evaluate.py
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from graphrag.config import (
    CHUNKS_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    get_chat_client, get_embed_client,
)
from graphrag.pdf_loader import load_chunks
from graphrag.retriever import GraphRAGRetriever
from graphrag.flat_rag import FlatRAGRetriever
from graphrag.evaluator import Evaluator, BENCHMARK_CASES

RESULTS_PATH = "results/comparison.csv"
FAISS_INDEX_PATH = "data/faiss.index"
FAISS_META_PATH = "data/faiss_meta.pkl"


def main():
    parser = argparse.ArgumentParser(description="Run grounded Flat RAG vs GraphRAG benchmark.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between benchmark cases")
    args = parser.parse_args()

    if not os.path.exists(CHUNKS_PATH):
        print(f"'{CHUNKS_PATH}' not found. Run 01_ingest_pdfs.py first.")
        sys.exit(1)

    chat_client, chat_model = get_chat_client()
    embed_client, embed_model = get_embed_client()
    print(f"LLM model   : {chat_model}")
    print(f"Embed model : {embed_model}")

    chunks = load_chunks(CHUNKS_PATH)
    flat_rag = FlatRAGRetriever(embed_client, embed_model, chat_client, chat_model)

    if flat_rag.load_index(FAISS_INDEX_PATH, FAISS_META_PATH):
        print("Flat RAG index loaded from cache.")
    else:
        print("\nBuilding Flat RAG index ...")
        flat_rag.build_index(chunks)
        flat_rag.save_index(FAISS_INDEX_PATH, FAISS_META_PATH)

    print("Connecting to Neo4j ...")
    graphrag = GraphRAGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, chat_client, chat_model, chunks=chunks)

    cases = BENCHMARK_CASES[:args.limit] if args.limit else BENCHMARK_CASES
    print(f"\nRunning {len(cases)} benchmark cases ...")
    evaluator = Evaluator(flat_rag, graphrag)
    try:
        records = evaluator.run(cases=cases, delay=args.delay)
    finally:
        graphrag.close()

    Evaluator.save_csv(records, RESULTS_PATH)
    Evaluator.print_summary(records)

    print("\nDone. Review results/comparison.csv for detailed answers.")


if __name__ == "__main__":
    main()
