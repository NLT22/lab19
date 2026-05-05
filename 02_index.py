"""
Step 2: Extract entity/relation/literal facts from chunks using LLM, push to Neo4j.
Chạy 4 chunk song song (tận dụng LM Studio Max Concurrent Predictions = 4).

Usage:
  python 02_index.py           # index chunks
  python 02_index.py --clear   # xóa graph cũ rồi index lại
  python 02_index.py --workers 4   # số parallel workers (default: 4)
"""
import sys
import os
import asyncio
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from tqdm import tqdm
from graphrag.config import (
    CHUNKS_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    get_async_chat_client,
)
from graphrag.pdf_loader import load_chunks
from graphrag.extractor import AsyncEntityRelationExtractor
from graphrag.graph_builder import Neo4jGraphBuilder


async def process_chunks(chunks: list[dict], extractor: AsyncEntityRelationExtractor,
                         workers: int) -> list[tuple[list, dict]]:
    """Process all chunks concurrently, limited by semaphore."""
    sem = asyncio.Semaphore(workers)
    results = [None] * len(chunks)
    pbar = tqdm(total=len(chunks), desc="Indexing")

    async def handle(i: int, chunk: dict):
        async with sem:
            facts = await extractor.aextract(chunk["text"])
            results[i] = (facts, chunk)
            pbar.update(1)

    await asyncio.gather(*[handle(i, c) for i, c in enumerate(chunks)])
    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear graph before indexing")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (match LM Studio Max Concurrent Predictions)")
    args = parser.parse_args()

    if not os.path.exists(CHUNKS_PATH):
        print(f"chunks.json not found at '{CHUNKS_PATH}'. Run 01_ingest_pdfs.py first.")
        sys.exit(1)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    chat_client, chat_model = get_async_chat_client()
    print(f"LLM: {chat_model}  |  workers: {args.workers}")

    extractor = AsyncEntityRelationExtractor(chat_client, chat_model)

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    try:
        builder_instance = Neo4jGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        builder_instance.setup_schema()
    except Exception as e:
        if "ServiceUnavailable" in type(e).__name__ or "10061" in str(e):
            print("\n[ERROR] Cannot connect to Neo4j.")
            print("  → Make sure Neo4j Desktop is running and the database is STARTED.")
            print(f"  → Expected: {NEO4J_URI}")
        else:
            print(f"\n[ERROR] Neo4j connection failed: {e}")
        sys.exit(1)

    with builder_instance as builder:
        if args.clear:
            confirm = input("Are you sure you want to clear the graph? (yes/no): ")
            if confirm.strip().lower() == "yes":
                builder.clear_graph()
                print("Graph cleared.")

        print(f"\nExtracting structured facts from {len(chunks)} chunks ({args.workers} parallel workers) ...")
        results = asyncio.run(process_chunks(chunks, extractor, args.workers))

        print("Writing to Neo4j ...")
        total_facts = 0
        for facts, chunk in results:
            if facts:
                builder.bulk_upsert(
                    facts,
                    source_doc=chunk.get("source", ""),
                    page=chunk.get("page"),
                    chunk_idx=chunk.get("chunk_idx"),
                )
                total_facts += len(facts)

        stats = builder.get_stats()

    print(f"\nIndexing complete.")
    print(f"  Facts extracted   : {total_facts}")
    print(f"  Nodes in graph    : {stats['nodes']}")
    print(f"  Relations in graph: {stats['relationships']}")
    print(f"\nOpen Neo4j Browser at http://localhost:7474 to visualize.")
    print("  Try: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")


if __name__ == "__main__":
    main()
