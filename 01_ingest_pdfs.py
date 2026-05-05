"""
Step 1: Load PDF files from data/pdfs/, extract and chunk text, save to data/chunks.json
Usage: python 01_ingest_pdfs.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from graphrag.config import PDF_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from graphrag.pdf_loader import PDFLoader, save_chunks
from pathlib import Path


def main():
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'.")
        print("Please download Wikipedia PDFs about tech companies and place them in that folder.")
        print("  Example: Google.pdf, OpenAI.pdf, Microsoft.pdf, Apple.pdf, Tesla.pdf ...")
        return

    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    loader = PDFLoader(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"\nExtracting text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = loader.load_all_pdfs(PDF_DIR)

    if not chunks:
        print("No text could be extracted. Verify that the PDFs contain selectable text.")
        return

    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"\nExtraction complete:")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length   : {avg_len:.0f} chars/chunk")
    sources = {c["source"] for c in chunks}
    print(f"  Sources      : {', '.join(sorted(sources))}")

    save_chunks(chunks, CHUNKS_PATH)
    print(f"\nSaved → {CHUNKS_PATH}")
    print("\nPreview of first chunk:")
    print("-" * 60)
    print(chunks[0]["text"][:400], "...")


if __name__ == "__main__":
    main()
