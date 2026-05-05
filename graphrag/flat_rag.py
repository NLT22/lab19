import os
import pickle
import time
import numpy as np
import faiss
from openai import OpenAI
from graphrag.config import chat_kwargs

ANSWER_PROMPT = """You are a knowledgeable assistant. Use the following retrieved passages to answer the question.
If the passages do not contain enough information, say so clearly.

Retrieved passages:
{context}

Question: {question}
Answer:"""


class FlatRAGRetriever:
    def __init__(self, embed_client: OpenAI, embed_model: str,
                 chat_client: OpenAI, chat_model: str):
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.index: faiss.IndexFlatIP | None = None
        self.docs: list[str] = []
        self.doc_meta: list[dict] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Get embeddings from LM Studio, returns L2-normalized numpy array."""
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def build_index(self, chunks: list[dict], batch_size: int = 32) -> None:
        """Embed all chunks and build FAISS index."""
        texts = [c["text"] for c in chunks]
        self.doc_meta = chunks
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            print(f"  [flat_rag] embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            vecs = self._embed(batch)
            all_vecs.append(vecs)

        all_vecs = np.vstack(all_vecs)
        dim = all_vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_vecs)
        self.docs = texts
        print(f"  [flat_rag] Index built: {len(texts)} vectors, dim={dim}")

    def save_index(self, index_path: str, meta_path: str) -> None:
        """Persist FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"docs": self.docs, "doc_meta": self.doc_meta}, f)
        print(f"  [flat_rag] Index saved → {index_path}")

    def load_index(self, index_path: str, meta_path: str) -> bool:
        """Load persisted index. Returns True if successful."""
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.doc_meta = data["doc_meta"]
        print(f"  [flat_rag] Index loaded from {index_path} ({len(self.docs)} vectors)")
        return True

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k most relevant chunks for query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_vec = self._embed([query])
        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.doc_meta[idx]
            results.append({**meta, "score": float(score)})
        return results

    def answer(self, question: str, k: int = 5) -> dict:
        """Run full Flat RAG pipeline: embed → search → LLM answer."""
        t0 = time.time()
        hits = self.search(question, k=k)
        context_parts = []
        for i, h in enumerate(hits, 1):
            source = h.get("source", "unknown")
            context_parts.append(f"[{i}] (source: {source})\n{h['text']}")
        context = "\n\n".join(context_parts) if context_parts else "(No relevant passages found.)"

        try:
            resp = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                    context=context, question=question
                )}],
                max_completion_tokens=512,
                **chat_kwargs(temperature=0.2),
            )
            answer_text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
        except Exception as e:
            answer_text = f"[Error: {e}]"
            tokens = 0

        sources = sorted({h.get("source", "") for h in hits if h.get("source")})
        return {
            "answer": answer_text,
            "retrieved_chunks": len(hits),
            "sources": sources,
            "tokens": tokens,
            "latency": round(time.time() - t0, 2),
        }
