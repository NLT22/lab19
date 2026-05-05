"""
Step 4: Run 20-question benchmark. Compare Flat RAG vs GraphRAG.
Results saved to results/comparison.csv
Usage: python 04_evaluate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from graphrag.config import (
    CHUNKS_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    get_chat_client, get_embed_client,
)
from graphrag.pdf_loader import load_chunks
from graphrag.retriever import GraphRAGRetriever
from graphrag.flat_rag import FlatRAGRetriever
from graphrag.evaluator import Evaluator, BENCHMARK_QUESTIONS

RESULTS_PATH = "results/comparison.csv"


def main():
    if not os.path.exists(CHUNKS_PATH):
        print(f"'{CHUNKS_PATH}' not found. Run 01_ingest_pdfs.py first.")
        sys.exit(1)

    chat_client, chat_model = get_chat_client()
    embed_client, embed_model = get_embed_client()
    print(f"LLM model   : {chat_model}")
    print(f"Embed model : {embed_model}")

    print("\nBuilding Flat RAG index ...")
    chunks = load_chunks(CHUNKS_PATH)
    flat_rag = FlatRAGRetriever(embed_client, embed_model, chat_client, chat_model)
    flat_rag.build_index(chunks)

    print("Connecting to Neo4j ...")
    graphrag = GraphRAGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, chat_client, chat_model)

    print(f"\nRunning {len(BENCHMARK_QUESTIONS)} benchmark questions ...")
    evaluator = Evaluator(flat_rag, graphrag)
    try:
        records = evaluator.run(delay=1.5)
    finally:
        graphrag.close()

    Evaluator.save_csv(records, RESULTS_PATH)
    Evaluator.print_summary(records)

    print("\nDone. Review results/comparison.csv for detailed answers.")


if __name__ == "__main__":
    main()
