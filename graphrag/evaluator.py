import csv
import os
import time
from dataclasses import dataclass, field, asdict

BENCHMARK_QUESTIONS = [
    # === OpenAI (sources: OpenAI.pdf, Anthropic.pdf) ===
    "Who founded OpenAI and in what year?",
    "What is the investment relationship between Microsoft and OpenAI?",
    "Who has provided funding to Anthropic?",
    "What is the relationship between OpenAI and Anthropic in terms of founding team?",

    # === Microsoft (sources: Microsoft.pdf, LinkedIn.pdf) ===
    "What major AI or cloud acquisition did Microsoft complete in recent years?",
    "Who are the co-founders of LinkedIn, and how is LinkedIn related to Microsoft?",

    # === Apple & Nvidia (sources: Apple_Inc.pdf, Nvidia.pdf) ===
    "What are the main hardware products Apple is known for?",
    "What role does Nvidia play in the AI and machine learning ecosystem?",
    "What chip or semiconductor technology does Nvidia supply to AI companies?",

    # === Tesla & SpaceX / Elon Musk (sources: Tesla,_Inc.pdf) ===
    "Who is the CEO of Tesla and what other companies is that person associated with?",
    "When was Tesla founded and who were its original founders?",

    # === Meta & YouTube (sources: Meta_Platforms.pdf, YouTube.pdf) ===
    "What social media platforms does Meta own or operate?",
    "How did YouTube come to be owned by Google, and what is its business model?",

    # === Amazon AWS (sources: Amazon_Web_Services.pdf) ===
    "What cloud computing services does Amazon Web Services offer?",
    "Which companies are the main competitors of Amazon AWS?",

    # === Multi-hop: cross-source reasoning ===
    "Which AI company was founded by people who previously worked at OpenAI?",
    "What is the connection between Microsoft, OpenAI, and Azure cloud services?",
    "How does Nvidia's GPU technology relate to the growth of companies like OpenAI and Anthropic?",
    "What companies in the corpus are most closely connected to Elon Musk?",
    "Compare the founding stories of OpenAI and Anthropic — what do they have in common?",
]


@dataclass
class EvalRecord:
    question: str
    flat_rag_answer: str = ""
    graphrag_answer: str = ""
    flat_rag_tokens: int = 0
    graphrag_tokens: int = 0
    flat_rag_time: float = 0.0
    graphrag_time: float = 0.0
    flat_rag_chunks: int = 0
    graphrag_triples: int = 0


class Evaluator:
    def __init__(self, flat_rag, graphrag_retriever):
        self.flat_rag = flat_rag
        self.graphrag = graphrag_retriever

    def run(self, questions: list[str] | None = None, delay: float = 1.0) -> list[EvalRecord]:
        if questions is None:
            questions = BENCHMARK_QUESTIONS

        records = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {q}")

            print("  → Flat RAG ...")
            flat_result = self.flat_rag.answer(q)

            print("  → GraphRAG ...")
            graph_result = self.graphrag.answer(q)

            rec = EvalRecord(
                question=q,
                flat_rag_answer=flat_result["answer"],
                graphrag_answer=graph_result["answer"],
                flat_rag_tokens=flat_result["tokens"],
                graphrag_tokens=graph_result["tokens"],
                flat_rag_time=flat_result["latency"],
                graphrag_time=graph_result["latency"],
                flat_rag_chunks=flat_result.get("retrieved_chunks", 0),
                graphrag_triples=graph_result.get("subgraph_triples", 0),
            )
            records.append(rec)
            time.sleep(delay)

        return records

    @staticmethod
    def save_csv(records: list[EvalRecord], path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))
        print(f"\nResults saved → {path}")

    @staticmethod
    def print_summary(records: list[EvalRecord]) -> None:
        total = len(records)
        avg_flat_tokens = sum(r.flat_rag_tokens for r in records) / total
        avg_graph_tokens = sum(r.graphrag_tokens for r in records) / total
        avg_flat_time = sum(r.flat_rag_time for r in records) / total
        avg_graph_time = sum(r.graphrag_time for r in records) / total

        print("\n" + "=" * 60)
        print(f"EVALUATION SUMMARY ({total} questions)")
        print("=" * 60)
        print(f"{'Metric':<30} {'Flat RAG':>12} {'GraphRAG':>12}")
        print("-" * 60)
        print(f"{'Avg tokens/question':<30} {avg_flat_tokens:>12.1f} {avg_graph_tokens:>12.1f}")
        print(f"{'Avg latency (s)':<30} {avg_flat_time:>12.2f} {avg_graph_time:>12.2f}")
        print("=" * 60)
