import csv
import os
import re
import time
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Benchmark cases — grounded in corpus, with gold answers and automated scoring
# ---------------------------------------------------------------------------
BENCHMARK_CASES: list[dict] = [
    # === OpenAI ===
    {"id": "E01", "type": "ownership",   "difficulty": "medium", "question": "According to the corpus, how is OpenAI ownership split in 2026?", "gold_answer": "OpenAI ownership is listed as employees and investors 47%, Microsoft 27%, and OpenAI Foundation 26%.", "must_include": ["employees and investors 47%", "Microsoft 27%", "OpenAI Foundation 26%"], "must_not_include": []},
    {"id": "E02", "type": "valuation",   "difficulty": "easy",   "question": "What valuation did OpenAI reach in its October 2025 share sale?", "gold_answer": "OpenAI conducted a $6.6 billion share sale in October 2025 that valued the company at $500 billion.", "must_include": ["$6.6 billion", "October 2025", "$500 billion"], "must_not_include": []},
    {"id": "E03", "type": "financial",   "difficulty": "easy",   "question": "What revenue and estimated net income does the corpus list for OpenAI in 2025?", "gold_answer": "OpenAI is listed with US$13.1 billion revenue in 2025 and an estimated US$9 billion net loss.", "must_include": ["13.1 billion", "2025", "net loss", "9 billion"], "must_not_include": []},
    {"id": "E04", "type": "governance",  "difficulty": "medium", "question": "What happened to Sam Altman in November 2023 according to the OpenAI article?", "gold_answer": "OpenAI's board removed Sam Altman as CEO in November 2023, then reinstated him five days later after a board reconstruction.", "must_include": ["removed", "CEO", "November 2023", "reinstated", "five days"], "must_not_include": []},
    {"id": "E05", "type": "founding",    "difficulty": "medium", "question": "Who pledged OpenAI's initial $1 billion in capital?", "gold_answer": "The corpus says $1 billion was pledged by Elon Musk, Sam Altman, Greg Brockman, Reid Hoffman, Jessica Livingston, Peter Thiel, Amazon Web Services, and Infosys.", "must_include": ["Elon Musk", "Sam Altman", "Greg Brockman", "Reid Hoffman", "AWS", "Infosys"], "must_not_include": []},
    {"id": "E06", "type": "timeline",    "difficulty": "medium", "question": "What OpenAI products or systems are listed around ChatGPT, Sora, Whisper, Codex, and Deep Research?", "gold_answer": "The corpus lists ChatGPT, Deep Research, ChatGPT Search, ChatGPT Atlas, OpenAI Codex, Sora, Whisper, and an API giving access to OpenAI models.", "must_include": ["ChatGPT", "Deep Research", "ChatGPT Atlas", "OpenAI Codex", "Sora", "Whisper", "API"], "must_not_include": []},
    {"id": "E07", "type": "partnership", "difficulty": "hard",   "question": "How is Disney connected to OpenAI in the corpus?", "gold_answer": "Disney said in December 2025 it would invest $1 billion in OpenAI and signed a three-year licensing deal allowing users to generate videos using Sora with Disney, Marvel, Star Wars, and Pixar characters; Disney later exited the deal in late March 2026 because Sora was discontinued.", "must_include": ["Disney", "$1 billion", "three-year licensing deal", "Sora", "exited"], "must_not_include": []},
    {"id": "E08", "type": "partnership", "difficulty": "hard",   "question": "What proposed Amazon-OpenAI partnership is mentioned in early 2026?", "gold_answer": "Amazon entered advanced discussions to invest up to $50 billion in OpenAI, with OpenAI models potentially integrated into Alexa and internal projects.", "must_include": ["Amazon", "50 billion", "Alexa"], "must_not_include": []},
    {"id": "E09", "type": "technical",   "difficulty": "medium", "question": "Why did OpenAI collaborate with Broadcom in 2024?", "gold_answer": "OpenAI began collaborating with Broadcom to design a custom AI chip for training and inference, targeted for mass production in 2026 by TSMC, to reduce dependence on Nvidia GPUs.", "must_include": ["Broadcom", "custom AI chip", "training and inference", "2026", "Nvidia"], "must_not_include": []},

    # === Anthropic ===
    {"id": "E10", "type": "valuation",   "difficulty": "easy",   "question": "What valuation is Anthropic estimated to have as of February 2026?", "gold_answer": "Anthropic is described as privately held and valued at an estimated $380 billion as of February 2026.", "must_include": ["380 billion", "February 2026"], "must_not_include": []},
    {"id": "E11", "type": "product",     "difficulty": "easy",   "question": "What did Anthropic announce in May 2025 besides Claude 4?", "gold_answer": "In May 2025 Anthropic announced Claude 4 and introduced new API capabilities including the Model Context Protocol connector.", "must_include": ["Claude 4", "API", "Model Context Protocol", "MCP"], "must_not_include": []},
    {"id": "E12", "type": "policy",      "difficulty": "medium", "question": "What sales restriction did Anthropic announce in September 2025?", "gold_answer": "Anthropic said it would stop selling products to groups majority-owned by Chinese, Russian, Iranian, or North Korean entities because of national security concerns.", "must_include": ["Chinese", "Russian", "Iranian", "North Korean", "national security"], "must_not_include": []},
    {"id": "E13", "type": "security",    "difficulty": "medium", "question": "How did hackers reportedly misuse Claude in November 2025?", "gold_answer": "Anthropic said Chinese government-sponsored hackers used Claude to perform automated cyberattacks against around 30 global organizations, bypassing safeguards by pretending the activity was defensive testing.", "must_include": ["Chinese government", "cyberattacks", "30", "defensive testing"], "must_not_include": []},
    {"id": "E14", "type": "acquisition", "difficulty": "easy",   "question": "Why did Anthropic acquire Bun?", "gold_answer": "Anthropic acquired Bun in December 2025 to improve the speed and stability of Claude Code.", "must_include": ["Bun", "December 2025", "Claude Code"], "must_not_include": []},
    {"id": "E15", "type": "partnership", "difficulty": "medium", "question": "What was Anthropic's Snowflake partnership?", "gold_answer": "Anthropic signed a multi-year $200 million partnership with Snowflake to make Claude models available through Snowflake's platform.", "must_include": ["Snowflake", "200 million", "Claude"], "must_not_include": []},
    {"id": "E16", "type": "government",  "difficulty": "hard",   "question": "How is Anthropic connected to Palantir and U.S. defense agencies?", "gold_answer": "In November 2024 Anthropic partnered with Palantir and AWS to provide Claude to U.S. intelligence and defense agencies; by February 2026 the corpus says Claude was the only AI model used in classified missions.", "must_include": ["Palantir", "AWS", "intelligence", "classified"], "must_not_include": []},

    # === Microsoft ===
    {"id": "E17", "type": "services",    "difficulty": "medium", "question": "List Microsoft's major services from the infobox.", "gold_answer": "The corpus lists services including Edge, Azure, Bing, LinkedIn, Viva Engage, Microsoft 365, OneDrive, Dynamics 365, Outlook, GitHub, Microsoft Store, Windows Update, Game Pass, and Xbox network.", "must_include": ["Azure", "Bing", "LinkedIn", "GitHub", "Game Pass"], "must_not_include": []},
    {"id": "E18", "type": "financial",   "difficulty": "easy",   "question": "What revenue, operating income, and net income does Microsoft report for 2025?", "gold_answer": "Microsoft reports US$281.7 billion revenue, US$128.5 billion operating income, and US$101.8 billion net income for 2025.", "must_include": ["281.7 billion", "128.5 billion", "101.8 billion"], "must_not_include": []},
    {"id": "E19", "type": "acquisition", "difficulty": "medium", "question": "What major gaming acquisition did Microsoft complete in 2023?", "gold_answer": "Microsoft acquired Activision Blizzard in an all-cash deal worth $68.7 billion, completed in 2023.", "must_include": ["Activision Blizzard", "68.7 billion", "2023"], "must_not_include": []},
    {"id": "E20", "type": "ai_org",      "difficulty": "medium", "question": "What happened between Microsoft and Inflection AI in March 2024?", "gold_answer": "Inflection AI cofounders Mustafa Suleyman and Karen Simonyan left to start Microsoft AI, Microsoft acqui-hired nearly the 70-person workforce, and paid Inflection $650 million to license its technology.", "must_include": ["Mustafa Suleyman", "Karen Simonyan", "Microsoft AI", "70-person", "650 million"], "must_not_include": []},
    {"id": "E21", "type": "chip",        "difficulty": "medium", "question": "What custom chips did Microsoft announce in November 2023?", "gold_answer": "Microsoft announced Maia, a chip designed to run large language models, and Cobalt CPU, designed to power general cloud services on Azure.", "must_include": ["Maia", "large language models", "Cobalt CPU", "Azure"], "must_not_include": []},

    # === Nvidia ===
    {"id": "E22", "type": "financial",      "difficulty": "easy",   "question": "What revenue and net income does Nvidia report for FY26?", "gold_answer": "Nvidia reports US$215.9 billion revenue and US$120.1 billion net income for FY26.", "must_include": ["215.9 billion", "120.1 billion", "FY26"], "must_not_include": []},
    {"id": "E23", "type": "products",       "difficulty": "medium", "question": "What Nvidia product lines are described for gaming, professional, and cloud gaming uses?", "gold_answer": "The corpus mentions GeForce GPUs for gaming and creative workloads, professional GPUs for edge/scientific/industrial applications, Shield devices, GeForce Now cloud gaming, and Tegra mobile processors.", "must_include": ["GeForce", "Shield", "GeForce Now", "Tegra"], "must_not_include": []},
    {"id": "E24", "type": "market",         "difficulty": "easy",   "question": "When did Nvidia first surpass $4 trillion and $5 trillion in market capitalization?", "gold_answer": "The corpus says Nvidia became the first company to surpass US$4 trillion and US$5 trillion in market capitalization in 2025 amid AI data center demand.", "must_include": ["2025", "4 trillion", "5 trillion", "AI data center"], "must_not_include": []},
    {"id": "E25", "type": "product_demand", "difficulty": "medium", "question": "What does the corpus say about Nvidia H100 demand and price in 2023-2024?", "gold_answer": "Nvidia H100 GPUs were in very high demand; in January 2024 analysts estimated H100s sold for $25,000 to $30,000 each, while individual units cost over $40,000 on eBay.", "must_include": ["H100", "25,000", "30,000", "40,000"], "must_not_include": []},
    {"id": "E26", "type": "partnership",    "difficulty": "medium", "question": "How was Nvidia connected to Microsoft's Xbox in the early 2000s?", "gold_answer": "Nvidia won the contract to develop graphics hardware for Microsoft's Xbox game console, including a $200 million advance.", "must_include": ["Xbox", "graphics hardware", "200 million"], "must_not_include": []},
    {"id": "E27", "type": "antitrust",      "difficulty": "medium", "question": "Which agencies investigated Nvidia, Microsoft, and OpenAI in 2024 and how were responsibilities split?", "gold_answer": "The FTC and DOJ began antitrust investigations. The FTC led investigations into Microsoft and OpenAI, while the DOJ handled Nvidia.", "must_include": ["FTC", "DOJ", "Microsoft", "OpenAI", "Nvidia"], "must_not_include": []},

    # === Meta ===
    {"id": "E28", "type": "financial",      "difficulty": "easy",   "question": "What revenue and employee count does Meta report for 2025/2026?", "gold_answer": "Meta reports US$201 billion revenue in 2025 and 77,986 employees as of March 2026.", "must_include": ["201 billion", "77,986"], "must_not_include": []},
    {"id": "E29", "type": "business_model", "difficulty": "medium", "question": "How dependent was Meta on advertising revenue as of 2023?", "gold_answer": "The corpus says advertising accounted for 97.8 percent of Meta's total revenue as of 2023.", "must_include": ["advertising", "97.8", "2023"], "must_not_include": []},
    {"id": "E30", "type": "ai_investment",  "difficulty": "medium", "question": "What AI startup investment did Meta decide to make in June 2025?", "gold_answer": "Meta decided to make a multibillion-dollar investment into AI startup Scale AI, with financing potentially exceeding $10 billion.", "must_include": ["Scale AI", "multibillion", "10 billion"], "must_not_include": []},
    {"id": "E31", "type": "partnership",    "difficulty": "easy",   "question": "What long-term AI-related partnership did Meta announce in February 2026?", "gold_answer": "Meta announced a long-term partnership with Nvidia in February 2026.", "must_include": ["Meta", "Nvidia", "long-term", "February 2026"], "must_not_include": []},
    {"id": "E32", "type": "wearables",      "difficulty": "medium", "question": "What smart-glasses products are mentioned for Meta?", "gold_answer": "Meta and Luxottica released Ray-Ban Stories in March 2022, and Meta unveiled two new Ray-Ban smart glasses for prescription lenses on March 31, 2026.", "must_include": ["Ray-Ban", "Luxottica", "2022"], "must_not_include": []},

    # === Cross-company / unanswerable ===
    {"id": "E33", "type": "cross_company",  "difficulty": "hard",   "question": "Which companies in the corpus are connected to both Microsoft and Nvidia?", "gold_answer": "OpenAI is connected to Microsoft through investment/Azure and to Nvidia through chips. Anthropic is connected to both Microsoft Azure and Nvidia AI systems. Meta has partnerships with both.", "must_include": ["OpenAI", "Anthropic", "Meta", "Microsoft", "Nvidia"], "must_not_include": []},
    {"id": "E34", "type": "comparison",     "difficulty": "hard",   "question": "Compare how OpenAI and Anthropic are connected to Microsoft cloud infrastructure.", "gold_answer": "OpenAI receives Azure cloud computing resources from Microsoft after Microsoft invested over $13 billion. Anthropic said it would buy $30 billion of Microsoft Azure compute capacity running on Nvidia AI systems.", "must_include": ["OpenAI", "13 billion", "Azure", "Anthropic", "30 billion"], "must_not_include": []},
    {"id": "E35", "type": "comparison",     "difficulty": "hard",   "question": "Which corpus companies are described as directly developing or releasing named AI models?", "gold_answer": "OpenAI developed GPT, DALL-E, Sora and released ChatGPT; Anthropic developed Claude models; Meta announced Llama 2; Nvidia announced Alpamayo-R1 and Nemotron 3 models.", "must_include": ["OpenAI", "Anthropic", "Meta", "Nvidia", "GPT", "Claude", "Llama"], "must_not_include": []},
    {"id": "E36", "type": "unanswerable",   "difficulty": "medium", "question": "According to the current corpus, what is Apple's 2025 revenue?", "gold_answer": "The current corpus does not contain an Apple article or enough information to answer Apple's 2025 revenue.", "must_include": ["does not", "Apple"], "must_not_include": []},
]

# Flat list of question strings (backward-compat for 03_query.py)
BENCHMARK_QUESTIONS = [c["question"] for c in BENCHMARK_CASES]


_ALIASES: dict[str, list[str]] = {
    "aws": ["aws", "amazon web services"],
    "does not": ["does not", "doesn't", "do not", "not contain", "no information", "not enough information", "cannot provide"],
    "employees and investors 47%": [
        "employees and investors 47%",
        "employees and other investors 47%",
        "employees and investors own 47%",
        "employees and other investors own 47%",
        "remaining 47% is owned by employees",
    ],
    "microsoft 27%": ["microsoft 27%", "microsoft holds 27%", "microsoft (27%)"],
    "openai foundation 26%": ["openai foundation 26%", "openai foundation holds 26%", "openai foundation (26%)"],
    "net loss": ["net loss", "loss", "negative net income"],
    "three-year licensing deal": ["three-year licensing deal", "three year licensing deal", "3-year licensing deal"],
    "large language models": ["large language models", "llms", "llm"],
    "70-person": ["70-person", "70 person", "70-person workforce", "70 person workforce"],
}

_REFUSAL_MARKERS = [
    "does not contain",
    "do not contain",
    "not contain information",
    "not enough information",
    "cannot provide",
    "cannot answer",
    "no information",
    "missing",
]


def _normalize_for_score(text: str) -> str:
    text = text.lower()
    text = text.replace("us$", "$")
    text = text.replace("percent", "%")
    text = text.replace("per cent", "%")
    text = text.replace("-", " ")
    text = re.sub(r"[$,()]", "", text)
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    stop = {"and", "or", "the", "a", "an", "of", "in", "as", "to", "with", "by", "for", "own", "owns", "holds"}
    return [t for t in _normalize_for_score(text).split() if t not in stop]


def _term_matches(answer_norm: str, term: str) -> bool:
    variants = _ALIASES.get(term.lower(), [term])
    for variant in variants:
        variant_norm = _normalize_for_score(variant)
        if variant_norm and variant_norm in answer_norm:
            return True
        vtoks = _tokens(variant)
        # Multi-token checks allow harmless wording changes, e.g. "employees and other investors own 47%".
        if len(vtoks) >= 2 and all(tok in answer_norm.split() for tok in vtoks):
            return True
    return False


def _is_refusal(answer_norm: str) -> bool:
    return any(marker in answer_norm for marker in _REFUSAL_MARKERS)


def _expects_refusal(must_include: list[str]) -> bool:
    text = " ".join(must_include).lower()
    return "does not" in text or "not enough" in text or "cannot" in text


def _score_answer(answer: str, must_include: list[str], must_not_include: list[str]) -> tuple[int, int]:
    """Return (hits, max_possible), with light normalization and alias matching."""
    answer_norm = _normalize_for_score(answer)
    if _is_refusal(answer_norm) and not _expects_refusal(must_include):
        return 0, len(must_include)

    hits = sum(1 for term in must_include if _term_matches(answer_norm, term))
    penalty = sum(1 for term in must_not_include if _term_matches(answer_norm, term))
    return max(0, hits - penalty), len(must_include)


@dataclass
class EvalRecord:
    question_id: str
    question_type: str
    difficulty: str
    question: str
    gold_answer: str
    flat_rag_answer: str = ""
    graphrag_answer: str = ""
    flat_rag_score: int = 0
    graphrag_score: int = 0
    max_score: int = 0
    flat_rag_tokens: int = 0
    graphrag_tokens: int = 0
    flat_rag_time: float = 0.0
    graphrag_time: float = 0.0
    flat_rag_chunks: int = 0
    graphrag_triples: int = 0
    flat_rag_sources: str = ""
    graphrag_entities: str = ""
    graphrag_top_triples: str = ""


class Evaluator:
    def __init__(self, flat_rag, graphrag_retriever):
        self.flat_rag = flat_rag
        self.graphrag = graphrag_retriever

    def run(self, cases: list[dict] | None = None, delay: float = 1.0) -> list[EvalRecord]:
        if cases is None:
            cases = BENCHMARK_CASES

        records = []
        for i, case in enumerate(cases, 1):
            q = case["question"]
            print(f"\n[{i}/{len(cases)}] [{case['id']}][{case['difficulty']}] {q[:70]}")

            print("  -> Flat RAG ...")
            flat_result = self.flat_rag.answer(q)

            print("  -> GraphRAG ...")
            graph_result = self.graphrag.answer(q)

            mi = case.get("must_include", [])
            mni = case.get("must_not_include", [])
            flat_score, max_score = _score_answer(flat_result["answer"], mi, mni)
            graph_score, _ = _score_answer(graph_result["answer"], mi, mni)

            print(f"     flat={flat_score}/{max_score}  graph={graph_score}/{max_score}")

            rec = EvalRecord(
                question_id=case["id"],
                question_type=case["type"],
                difficulty=case["difficulty"],
                question=q,
                gold_answer=case.get("gold_answer", ""),
                flat_rag_answer=flat_result["answer"],
                graphrag_answer=graph_result["answer"],
                flat_rag_score=flat_score,
                graphrag_score=graph_score,
                max_score=max_score,
                flat_rag_tokens=flat_result["tokens"],
                graphrag_tokens=graph_result["tokens"],
                flat_rag_time=flat_result["latency"],
                graphrag_time=graph_result["latency"],
                flat_rag_chunks=flat_result.get("retrieved_chunks", 0),
                graphrag_triples=graph_result.get("subgraph_triples", 0),
                flat_rag_sources="|".join(flat_result.get("sources", [])),
                graphrag_entities="|".join(graph_result.get("entities_found", [])),
                graphrag_top_triples=graph_result.get("top_triples", ""),
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
        print(f"\nResults saved: {path}")

    @staticmethod
    def print_summary(records: list[EvalRecord]) -> None:
        total = len(records)
        total_max = sum(r.max_score for r in records)
        flat_total = sum(r.flat_rag_score for r in records)
        graph_total = sum(r.graphrag_score for r in records)
        avg_flat_tok = sum(r.flat_rag_tokens for r in records) / total
        avg_graph_tok = sum(r.graphrag_tokens for r in records) / total
        avg_flat_t = sum(r.flat_rag_time for r in records) / total
        avg_graph_t = sum(r.graphrag_time for r in records) / total

        flat_pct = flat_total / total_max * 100 if total_max else 0
        graph_pct = graph_total / total_max * 100 if total_max else 0

        print("\n" + "=" * 65)
        print(f"EVALUATION SUMMARY  ({total} questions, {total_max} total points)")
        print("=" * 65)
        print(f"{'Metric':<32} {'Flat RAG':>14} {'GraphRAG':>14}")
        print("-" * 65)
        print(f"{'Score (must_include hits)':<32} {flat_total:>8}/{total_max} {graph_total:>8}/{total_max}")
        print(f"{'Accuracy %':<32} {flat_pct:>13.1f}% {graph_pct:>13.1f}%")
        print(f"{'Avg tokens/question':<32} {avg_flat_tok:>14.1f} {avg_graph_tok:>14.1f}")
        print(f"{'Avg latency (s)':<32} {avg_flat_t:>14.2f} {avg_graph_t:>14.2f}")
        print("-" * 65)

        # Break down by difficulty
        for diff in ["easy", "medium", "hard"]:
            sub = [r for r in records if r.difficulty == diff]
            if not sub:
                continue
            sub_max = sum(r.max_score for r in sub)
            fs = sum(r.flat_rag_score for r in sub)
            gs = sum(r.graphrag_score for r in sub)
            fp = fs / sub_max * 100 if sub_max else 0
            gp = gs / sub_max * 100 if sub_max else 0
            print(f"  {diff:<8} ({len(sub):2d} Qs, {sub_max:3d} pts)   flat={fs}/{sub_max} ({fp:.0f}%)   graph={gs}/{sub_max} ({gp:.0f}%)")

        # Win breakdown
        flat_wins  = sum(1 for r in records if r.flat_rag_score > r.graphrag_score)
        graph_wins = sum(1 for r in records if r.graphrag_score > r.flat_rag_score)
        ties       = total - flat_wins - graph_wins
        print(f"\n  Flat wins: {flat_wins}  |  Graph wins: {graph_wins}  |  Ties: {ties}")
        print("=" * 65)
