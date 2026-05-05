"""
Step 1: Load PDF files from data/pdfs/, extract and chunk text, save to data/chunks.json

Usage:
  python 01_ingest_pdfs.py                   # dùng toàn bộ trang
  python 01_ingest_pdfs.py --max-pages 10    # chỉ đọc 10 trang đầu mỗi PDF (~300 chunks)
  python 01_ingest_pdfs.py --stats           # chỉ in thống kê, không ghi file
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter
from graphrag.config import PDF_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from graphrag.pdf_loader import PDFLoader, save_chunks
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Max pages to read per PDF (0 = all pages)")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats only, do not save chunks.json")
    args = parser.parse_args()

    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'.")
        print("Please download Wikipedia PDFs about tech companies and place them in that folder.")
        print("  Recommended: Google.pdf, OpenAI.pdf, Microsoft.pdf, Apple_Inc.pdf,")
        print("               Tesla_Inc.pdf, Meta_Platforms.pdf, Nvidia.pdf, Anthropic.pdf,")
        print("               Amazon_Web_Services.pdf, YouTube.pdf, LinkedIn.pdf")
        return

    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in sorted(pdf_files):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    max_pages = args.max_pages
    page_info = f"max {max_pages} pages/PDF" if max_pages else "all pages"
    print(f"\nExtracting text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, {page_info}) ...")

    loader = PDFLoader(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_pages=max_pages)
    chunks = loader.load_all_pdfs(PDF_DIR)

    if not chunks:
        print("No text could be extracted. Verify that the PDFs contain selectable text.")
        return

    # per-source stats
    counter = Counter(c["source"] for c in chunks)
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)

    print(f"\n{'Source':<45} {'Chunks':>7} {'Est. @ 3s/chunk':>16}")
    print("-" * 70)
    for src, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        mins = cnt * 3 / 60
        print(f"{src:<45} {cnt:>7} {mins:>13.1f} min")
    print("-" * 70)
    total = len(chunks)
    total_mins = total * 3 / 60
    print(f"{'TOTAL':<45} {total:>7} {total_mins:>13.1f} min ({total_mins/60:.1f} hr)")
    print(f"\nAvg chunk length: {avg_len:.0f} chars")

    if args.stats:
        print("\n[--stats mode] Not saving. Re-run without --stats to write chunks.json.")
        return

    save_chunks(chunks, CHUNKS_PATH)
    print(f"\nSaved → {CHUNKS_PATH}")
    print("\nPreview of first chunk:")
    print("-" * 60)
    print(chunks[0]["text"][:400], "...")


if __name__ == "__main__":
    main()
