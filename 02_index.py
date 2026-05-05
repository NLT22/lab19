"""
Step 2: Extract entity/relation triples from chunks using LLM, push to Neo4j.
Usage: python 02_index.py [--clear]
  --clear  Wipe existing graph before indexing (use carefully)
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from tqdm import tqdm
from graphrag.config import (
    CHUNKS_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    get_chat_client,
)
from graphrag.pdf_loader import load_chunks
from graphrag.extractor import EntityRelationExtractor
from graphrag.graph_builder import Neo4jGraphBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear graph before indexing")
    args = parser.parse_args()

    if not os.path.exists(CHUNKS_PATH):
        print(f"chunks.json not found at '{CHUNKS_PATH}'. Run 01_ingest_pdfs.py first.")
        sys.exit(1)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    chat_client, chat_model = get_chat_client()
    print(f"LLM: {chat_model}")

    extractor = EntityRelationExtractor(chat_client, chat_model)

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    with Neo4jGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as builder:
        builder.setup_schema()

        if args.clear:
            confirm = input("Are you sure you want to clear the graph? (yes/no): ")
            if confirm.strip().lower() == "yes":
                builder.clear_graph()
                print("Graph cleared.")

        total_triples = 0
        for chunk in tqdm(chunks, desc="Indexing"):
            triples = extractor.extract(chunk["text"])
            if triples:
                builder.bulk_upsert(triples, source_doc=chunk["source"])
                total_triples += len(triples)

        stats = builder.get_stats()

    print(f"\nIndexing complete.")
    print(f"  Triples extracted : {total_triples}")
    print(f"  Nodes in graph    : {stats['nodes']}")
    print(f"  Relations in graph: {stats['relationships']}")
    print(f"\nOpen Neo4j Browser at http://localhost:7474 to visualize.")
    print("  Try: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")


if __name__ == "__main__":
    main()
