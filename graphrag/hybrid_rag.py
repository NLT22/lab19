import time

from openai import OpenAI

from graphrag.config import chat_kwargs

HYBRID_ANSWER_PROMPT = """You are a corpus-grounded assistant. Use BOTH evidence channels:

1. Graph context: structured facts and graph-guided evidence.
2. Vector context: semantically retrieved source passages.

Prefer exact numbers, dates, and names from the source passages. Use the graph context
to connect entities and relationships across passages. If neither channel contains
enough information, say so clearly.

Graph context:
{graph_context}

Vector context:
{vector_context}

Question: {question}
Answer:"""


class HybridRAGRetriever:
    """Combine GraphRAG structured retrieval with Flat RAG vector retrieval."""

    def __init__(
        self,
        graph_rag,
        flat_rag,
        chat_client: OpenAI,
        chat_model: str,
        vector_k: int = 5,
    ):
        self.graph_rag = graph_rag
        self.flat_rag = flat_rag
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.vector_k = vector_k

    def _vector_context(self, question: str) -> tuple[str, list[str]]:
        hits = self.flat_rag.search(question, k=self.vector_k)
        parts = []
        sources = []
        for i, hit in enumerate(hits, 1):
            source = hit.get("source", "unknown")
            page = hit.get("page", "?")
            chunk_idx = hit.get("chunk_idx", "?")
            score = hit.get("score", 0.0)
            sources.append(f"{source} p{page} c{chunk_idx}")
            parts.append(
                f"[{i}] source={source} page={page} chunk={chunk_idx} score={score:.3f}\n"
                f"{hit.get('text', '')}"
            )
        if not parts:
            return "(No vector passages found.)", []
        return "\n\n".join(parts), sources

    def answer(self, question: str) -> dict:
        t0 = time.time()
        graph = self.graph_rag.retrieve_context(question)
        vector_context, vector_sources = self._vector_context(question)

        if graph.get("boundary_answer"):
            # For corpus-boundary questions, trust the graph's primary-source check.
            answer_text = graph["boundary_answer"]
            tokens = 0
        else:
            try:
                resp = self.chat_client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{
                        "role": "user",
                        "content": HYBRID_ANSWER_PROMPT.format(
                            graph_context=graph["context"],
                            vector_context=vector_context,
                            question=question,
                        ),
                    }],
                    max_completion_tokens=512,
                    **chat_kwargs(temperature=0.2),
                )
                answer_text = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else 0
            except Exception as e:
                answer_text = f"[Error: {e}]"
                tokens = 0

        subgraph = graph.get("subgraph", [])
        top_triples = "; ".join(
            f"{r.get('start')}--[{r.get('rel_type')}]-->{r.get('end')}"
            for r in subgraph[:5]
        )

        return {
            "answer": answer_text,
            "entities_found": graph.get("entities", []),
            "subgraph_triples": len(subgraph),
            "top_triples": top_triples,
            "retrieved_chunks": self.vector_k,
            "sources": vector_sources,
            "tokens": tokens,
            "latency": round(time.time() - t0, 2),
        }
