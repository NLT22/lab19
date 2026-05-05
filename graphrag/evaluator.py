import csv
import os
import time
from dataclasses import dataclass, field, asdict

BENCHMARK_QUESTIONS = [
    "Who founded Google and when was it established?",
    "What is the relationship between OpenAI and Microsoft?",
    "What major products and services does Apple offer?",
    "Who is the CEO of Tesla?",
    "Which companies compete with Amazon AWS in cloud computing?",
    "What programming language was created by Google?",
    "Who has invested in OpenAI?",
    "What significant AI-related acquisition did Microsoft make?",
    "Where is Meta's headquarters located?",
    "What is the relationship between YouTube and Google?",
    "What companies did Elon Musk found or co-found?",
    "Which major tech companies are headquartered in Seattle?",
    "What is the supply relationship between Apple and Samsung?",
    "Who are the co-founders of LinkedIn?",
    "What social media platform did Meta (Facebook) acquire?",
    "What cloud services does Google provide?",
    "Who invented the World Wide Web?",
    "What company developed the Android operating system?",
    "Which companies have invested in Anthropic?",
    "What role does Nvidia play in the AI and machine learning ecosystem?",
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
