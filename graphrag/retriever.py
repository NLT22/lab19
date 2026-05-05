import json
import re
import time
from openai import OpenAI
from neo4j import GraphDatabase

ENTITY_EXTRACT_PROMPT = """Extract the main named entities (companies, people, products, places) from this question.
Return ONLY a JSON array of strings, e.g. ["Google", "Sundar Pichai"]
Question: {question}"""

ANSWER_PROMPT = """You are a knowledgeable assistant. Use the following knowledge graph context to answer the question.
If the context does not contain enough information, say so clearly.

Context from knowledge graph:
{context}

Question: {question}
Answer:"""


def _parse_entity_list(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _textualize(subgraph: list[dict]) -> str:
    """Convert list of {start, rel_type, end, source} dicts to plain text."""
    if not subgraph:
        return "(No relevant information found in the knowledge graph.)"
    lines = []
    seen = set()
    for row in subgraph:
        line = f"{row['start']} --[{row['rel_type']}]--> {row['end']}"
        if row.get("source"):
            line += f" (source: {row['source']})"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


class GraphRAGRetriever:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 chat_client: OpenAI, chat_model: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.client = chat_client
        self.model = chat_model

    def close(self):
        self.driver.close()

    def extract_query_entities(self, question: str) -> list[str]:
        """Use LLM to extract entity names from a question."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(question=question)}],
                temperature=0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
            return _parse_entity_list(raw)
        except Exception as e:
            print(f"  [retriever] entity extraction failed: {e}")
            return []

    def get_subgraph(self, entity_names: list[str], hops: int = 2) -> list[dict]:
        """Cypher 2-hop traversal from matched entity nodes."""
        if not entity_names:
            return []
        # case-insensitive partial match
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (start:Entity)
                WHERE any(name IN $names WHERE toLower(start.name) CONTAINS toLower(name))
                MATCH path = (start)-[:RELATION*1..""" + str(hops) + """]->(neighbor)
                WITH start, relationships(path) AS rels, neighbor
                UNWIND rels AS r
                RETURN startNode(r).name AS start, r.type AS rel_type,
                       endNode(r).name AS end, r.source_doc AS source
                LIMIT 200
                """,
                names=entity_names,
            )
            return [dict(r) for r in result]

    def answer(self, question: str) -> dict:
        """Run full GraphRAG pipeline: extract entities → subgraph → LLM answer."""
        t0 = time.time()
        entities = self.extract_query_entities(question)
        subgraph = self.get_subgraph(entities)
        context = _textualize(subgraph)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                    context=context, question=question
                )}],
                temperature=0.2,
                max_tokens=512,
            )
            answer_text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
        except Exception as e:
            answer_text = f"[Error: {e}]"
            tokens = 0

        return {
            "answer": answer_text,
            "entities_found": entities,
            "subgraph_triples": len(subgraph),
            "tokens": tokens,
            "latency": round(time.time() - t0, 2),
        }
