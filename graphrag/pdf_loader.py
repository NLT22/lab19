import json
import os
from pathlib import Path

from pypdf import PdfReader


class PDFLoader:
    def __init__(self, chunk_size: int = 800, overlap: int = 100, max_pages: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_pages = max_pages  # 0 = no limit

    def load_pdf(self, path: str) -> list[dict]:
        """Extract text from each page of a PDF, return list of {source, page, text}."""
        reader = PdfReader(path)
        source = Path(path).stem
        pages = []
        page_list = reader.pages
        if self.max_pages > 0:
            page_list = page_list[:self.max_pages]
        for i, page in enumerate(page_list):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"source": source, "page": i + 1, "text": text})
        return pages

    def chunk_text(self, text: str, source: str, page: int) -> list[dict]:
        """Sliding window chunking of text."""
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "source": source,
                    "page": page,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                })
                chunk_idx += 1
            start += self.chunk_size - self.overlap
        return chunks

    def load_all_pdfs(self, pdf_dir: str) -> list[dict]:
        """Scan pdf_dir for *.pdf files, extract and chunk all text."""
        pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"[pdf_loader] No PDF files found in '{pdf_dir}'")
            return []

        all_chunks = []
        for pdf_path in pdf_files:
            print(f"  Loading: {pdf_path.name}")
            try:
                pages = self.load_pdf(str(pdf_path))
                for page_data in pages:
                    chunks = self.chunk_text(
                        page_data["text"], page_data["source"], page_data["page"]
                    )
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"  [warning] Failed to load {pdf_path.name}: {e}")

        return all_chunks


def save_chunks(chunks: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
