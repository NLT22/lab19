"""
Step 3: Interactive query REPL — compare GraphRAG vs Flat RAG answers in real time.
Usage: python 03_query.py
Commands: type a question, or 'flat', 'graph', 'both' to change mode, 'quit' to exit.
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

BANNER = """
╔══════════════════════════════════════════╗
║   GraphRAG vs Flat RAG — Query Console  ║
╚══════════════════════════════════════════╝
Mode commands: 'flat', 'graph', 'both'
Type 'quit' to exit.
"""

MODE_HELP = {
    "both": "Both Flat RAG and GraphRAG",
    "flat": "Flat RAG only",
    "graph": "GraphRAG only",
}


def print_result(label: str, result: dict):
    print(f"\n[{label}]")
    print(f"Answer  : {result['answer']}")
    print(f"Tokens  : {result['tokens']}  |  Latency: {result['latency']}s")


def main():
    chat_client, chat_model = get_chat_client()
    embed_client, embed_model = get_embed_client()

    print("Loading chunks for Flat RAG index ...")
    if not os.path.exists(CHUNKS_PATH):
        print(f"'{CHUNKS_PATH}' not found. Run 01_ingest_pdfs.py first.")
        sys.exit(1)
    chunks = load_chunks(CHUNKS_PATH)

    flat_rag = FlatRAGRetriever(embed_client, embed_model, chat_client, chat_model)
    flat_rag.build_index(chunks)

    graphrag = GraphRAGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, chat_client, chat_model)

    print(BANNER)
    mode = "both"
    print(f"Current mode: {MODE_HELP[mode]}")

    try:
        while True:
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if user_input.lower() in MODE_HELP:
                mode = user_input.lower()
                print(f"Mode set to: {MODE_HELP[mode]}")
                continue

            question = user_input
            if mode in ("flat", "both"):
                flat_result = flat_rag.answer(question)
                print_result("Flat RAG", flat_result)

            if mode in ("graph", "both"):
                graph_result = graphrag.answer(question)
                print_result("GraphRAG", graph_result)
    finally:
        graphrag.close()


if __name__ == "__main__":
    main()
