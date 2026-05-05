import json
import re
import time
from collections import defaultdict

from neo4j import GraphDatabase
from openai import OpenAI

from graphrag.config import chat_kwargs

ENTITY_EXTRACT_PROMPT = """Extract the main named entities from this question.
Include companies, people, products, places, services, technologies, and named models.
Return ONLY a JSON array of strings, e.g. ["OpenAI", "Microsoft Azure"].
Question: {question}"""

ANSWER_PROMPT = """You are a corpus-grounded assistant. Answer only from the supplied GraphRAG context.
The context contains structured graph facts and, when available, original evidence snippets.
If the context does not contain enough information, say so clearly and name the missing entity or fact.
Prefer exact numbers, dates, and names from evidence over paraphrase.

GraphRAG context:
{context}

Question: {question}
Answer:"""


REL_PRIORITY: dict[str, int] = {
    "CEO_OF": 6,
    "FOUNDED_BY": 6,
    "HAS_MISSION": 6,
    "HAS_REVENUE": 6,
    "HAS_NET_INCOME": 6,
    "HAS_EMPLOYEES": 6,
    "HAS_OWNERSHIP_SHARE": 6,
    "HAS_MARKET_SHARE": 6,
    "FOUNDED_IN": 5,
    "INVESTED_IN": 5,
    "VALUED_AT": 5,
    "RAISED": 5,
    "ACQUIRED": 4,
    "RELEASED": 4,
    "PROVIDES": 4,
    "DEVELOPED_BY": 4,
    "WORKS_FOR": 4,
    "SUBSIDIARY_OF": 3,
    "PARTNERED_WITH": 3,
    "OWNS": 3,
    "USES": 3,
    "LOCATED_IN": 2,
    "COMPETITOR_OF": 0,
    "COMPETES_WITH": 0,
}

ALLOWED_RELATIONS = set(REL_PRIORITY)
ATTRIBUTE_RELATIONS = {
    "HAS_REVENUE",
    "HAS_NET_INCOME",
    "HAS_EMPLOYEES",
    "HAS_OWNERSHIP_SHARE",
    "HAS_MARKET_SHARE",
    "VALUED_AT",
}

QUESTION_REL_HINTS: dict[str, list[str]] = {
    "revenue": ["HAS_REVENUE"],
    "net income": ["HAS_NET_INCOME"],
    "employee": ["HAS_EMPLOYEES"],
    "ownership": ["HAS_OWNERSHIP_SHARE"],
    "owned": ["HAS_OWNERSHIP_SHARE", "OWNS"],
    "market share": ["HAS_MARKET_SHARE"],
    "founded": ["FOUNDED_BY", "FOUNDED_IN"],
    "founder": ["FOUNDED_BY"],
    "ceo": ["CEO_OF"],
    "chairman": ["CEO_OF"],
    "invest": ["INVESTED_IN", "RAISED"],
    "pledged": ["RAISED", "INVESTED_IN"],
    "capital": ["RAISED", "INVESTED_IN"],
    "funding": ["RAISED", "INVESTED_IN"],
    "share sale": ["VALUED_AT", "RAISED"],
    "valuation": ["VALUED_AT"],
    "valued": ["VALUED_AT"],
    "sales restriction": ["PROVIDES"],
    "restriction": ["PROVIDES"],
    "acquisition": ["ACQUIRED"],
    "acquire": ["ACQUIRED"],
    "product": ["RELEASED", "OWNS"],
    "model": ["RELEASED", "DEVELOPED_BY"],
    "custom chip": ["RELEASED", "PROVIDES", "USES"],
    "chip": ["RELEASED", "PROVIDES", "USES"],
    "h100": ["RELEASED", "HAS_MARKET_SHARE"],
    "price": ["VALUED_AT", "HAS_REVENUE"],
    "hackers": ["USES"],
    "misuse": ["USES"],
    "service": ["PROVIDES"],
    "azure": ["PROVIDES", "USES", "INVESTED_IN"],
    "gpu": ["PROVIDES", "USES", "RELEASED", "HAS_MARKET_SHARE"],
    "partnership": ["PARTNERED_WITH", "INVESTED_IN", "USES"],
    "connected": ["PARTNERED_WITH", "INVESTED_IN", "USES", "PROVIDES", "RELEASED"],
}

LEXICAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "by",
    "from", "as", "at", "what", "which", "who", "how", "when", "where", "why",
    "according", "corpus", "current", "article", "report", "reports", "mentioned",
    "does", "did", "was", "were", "is", "are", "be", "have", "has", "had",
    "company", "companies", "list", "major", "between",
}

NOISY_SEED_TERMS = {
    "api", "apis", "service", "services", "product", "products", "model", "models",
    "company", "companies", "corpus companies", "ai", "ai models", "chips", "chip",
    "gpu", "gpus", "partnership", "partnerships", "revenue", "employees", "ownership",
    "valuation", "financial", "article", "corpus", "current corpus",
}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _parse_entity_list(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def _relation_hints(question: str) -> list[str]:
    q = question.lower()
    rels: list[str] = []
    for phrase, candidates in QUESTION_REL_HINTS.items():
        if phrase in q:
            rels.extend(candidates)
    seen: set[str] = set()
    return [r for r in rels if not (r in seen or seen.add(r))]


def _is_noisy_seed(entity: str) -> bool:
    value = entity.strip().lower()
    if not value:
        return True
    if value in NOISY_SEED_TERMS or value in MONTHS:
        return True
    if re.fullmatch(r"\d{4}", value):
        return True
    if re.fullmatch(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}", value):
        return True
    if re.fullmatch(r"\$?\d+(\.\d+)?\s*(billion|million|trillion)?", value):
        return True
    return False


def _source_aliases(source: str) -> set[str]:
    compact = re.sub(r"[^a-z0-9]+", " ", source.replace("_", " ").lower()).strip()
    aliases = {compact} if compact else set()
    if compact:
        aliases.add(compact.split()[0])
    aliases |= {
        "meta platforms": {"meta", "meta platforms", "meta platforms inc"},
        "microsoft": {"microsoft", "microsoft corporation"},
        "nvidia": {"nvidia", "nvidia corporation", "nvidia corp"},
        "openai": {"openai", "openai inc", "openai global", "openai global llc"},
        "anthropic": {"anthropic"},
    }.get(compact, set())
    return aliases


def _chunk_key(source: str, page, chunk_idx) -> tuple[str, str, str]:
    return (str(source), str(page), str(chunk_idx))


def _parse_source_refs(value: str) -> list[tuple[str, str, str]]:
    refs = []
    for ref in str(value or "").split("|"):
        match = re.match(r"(.+?)\s+p(\d+)\s+c(\d+)$", ref.strip())
        if match:
            refs.append((match.group(1), match.group(2), match.group(3)))
    return refs


def _row_sources(row: dict) -> set[str]:
    sources = set()
    for source, _, _ in _parse_source_refs(row.get("source", "")):
        sources.add(source)
    source_doc = row.get("source_doc")
    if source_doc:
        sources.add(str(source_doc))
    return sources


def _lexical_terms(question: str, entities: list[str]) -> list[str]:
    raw = " ".join([question, *entities]).lower()
    terms = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", raw)
    keep = []
    seen = set()
    for term in terms:
        if len(term) < 2 or term in LEXICAL_STOPWORDS:
            continue
        if term not in seen:
            seen.add(term)
            keep.append(term)
    # Preserve important multi-word phrases that single-token scoring misses.
    for phrase in [
        "share sale", "custom chip", "custom chips", "large language models",
        "scale ai", "artificial intelligence", "net income", "cloud computing",
        "source available", "defensive testing", "ray ban", "game pass",
    ]:
        if phrase in raw:
            keep.append(phrase)
    return keep


def _rank_subgraph(
    rows_1hop: list[dict],
    rows_2hop: list[dict],
    entity_names: list[str],
    top_1hop: int = 45,
    top_2hop: int = 20,
) -> list[dict]:
    names_lower = [n.lower() for n in entity_names]

    def score(row: dict) -> int:
        start = str(row.get("start", "")).lower()
        end = str(row.get("end", "")).lower()
        entity_hits = sum(1 for name in names_lower if name in start or name in end)
        rel_prio = REL_PRIORITY.get(str(row.get("rel_type", "")), 1)
        confidence = float(row.get("confidence") or 0.0)
        has_evidence = 1 if row.get("evidence") else 0
        return entity_hits * 20 + rel_prio * 3 + int(confidence * 3) + has_evidence

    ranked_1 = sorted(rows_1hop, key=score, reverse=True)[:top_1hop]
    ranked_2 = sorted(rows_2hop, key=score, reverse=True)[:top_2hop]

    seen: set[tuple] = set()
    combined: list[dict] = []
    for row in ranked_1 + ranked_2:
        key = (row.get("start"), row.get("rel_type"), row.get("end"))
        if key not in seen:
            seen.add(key)
            combined.append(row)
    return combined


def _textualize(subgraph: list[dict], entity_names: list[str] | None = None) -> str:
    if not subgraph:
        return "(No relevant information found in the knowledge graph.)"

    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for row in subgraph:
        key = (row.get("start"), row.get("rel_type"), row.get("end"))
        if key in seen:
            continue
        seen.add(key)
        groups[str(row.get("start", ""))].append(row)

    names_lower = [n.lower() for n in (entity_names or [])]

    def priority(name: str) -> int:
        return -sum(1 for n in names_lower if n in name.lower())

    lines: list[str] = []
    for start in sorted(groups.keys(), key=priority):
        lines.append(f"{start}:")
        for row in groups[start]:
            rel = row.get("rel_type", "")
            end = row.get("end", "")
            end_kind = row.get("end_kind") or "Other"
            src = row.get("source") or row.get("source_doc") or ""
            evidence = row.get("evidence") or ""
            src_str = f" [{src}]" if src else ""
            lines.append(f"  --[{rel}]--> {end} ({end_kind}){src_str}")
            if evidence:
                lines.append(f"     evidence: {evidence.split('|')[0]}")
    return "\n".join(lines)


class GraphRAGRetriever:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        chat_client: OpenAI,
        chat_model: str,
        chunks: list[dict] | None = None,
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.client = chat_client
        self.model = chat_model
        self.chunks = chunks or []
        self.chunk_lookup = {
            _chunk_key(c.get("source", ""), c.get("page", ""), c.get("chunk_idx", "")): c.get("text", "")
            for c in self.chunks
        }
        self.primary_sources = sorted({str(c.get("source", "")) for c in self.chunks if c.get("source")})
        self.primary_aliases: set[str] = set()
        for source in self.primary_sources:
            self.primary_aliases |= _source_aliases(source)

    def close(self):
        self.driver.close()

    def _filter_seed_entities(self, entities: list[str]) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            entity = str(entity).strip()
            key = entity.lower()
            if not entity or key in seen or _is_noisy_seed(entity):
                continue
            seen.add(key)
            filtered.append(entity)
        return filtered

    def _is_primary_entity(self, entity: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", entity.lower()).strip()
        return normalized in self.primary_aliases

    def _attribute_boundary_refusal(self, question: str, entities: list[str]) -> str | None:
        if not self.primary_sources:
            return None
        rel_hints = set(_relation_hints(question))
        if not (rel_hints & ATTRIBUTE_RELATIONS):
            return None
        meaningful = [e for e in entities if not _is_noisy_seed(e)]
        if meaningful and not any(self._is_primary_entity(e) for e in meaningful):
            names = ", ".join(meaningful)
            sources = ", ".join(self.primary_sources)
            return (
                f"The current corpus does not contain a primary article for {names}; "
                f"available primary corpus sources are: {sources}. I do not have enough "
                f"corpus-grounded information to answer that attribute question."
            )
        return None

    @staticmethod
    def _filter_allowed_rows(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r.get("rel_type") in ALLOWED_RELATIONS]

    def extract_query_entities(self, question: str) -> list[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(question=question)}],
                max_completion_tokens=256,
                **chat_kwargs(temperature=0),
            )
            raw = resp.choices[0].message.content or ""
            return _parse_entity_list(raw)
        except Exception as e:
            print(f"  [retriever] entity extraction failed: {e}")
            return []

    def infer_entities_from_graph(self, question: str, limit: int = 8) -> list[str]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE size(n.name) >= 3 AND toLower($question) CONTAINS toLower(n.name)
                RETURN n.name AS name
                ORDER BY size(n.name) DESC
                LIMIT $limit
                """,
                question=question,
                limit=limit,
            )
            return [r["name"] for r in result]

    def get_subgraph(self, entity_names: list[str], question: str = "", hops: int = 2) -> tuple[list[dict], list[dict]]:
        rel_hints = _relation_hints(question)
        with self.driver.session() as session:
            if not entity_names:
                if not rel_hints:
                    return [], []
                result = session.run(
                    """
                    MATCH (start:Entity)-[r:RELATION]->(neighbor:Entity)
                    WHERE r.type IN $rel_hints
                      AND r.type IN $allowed_rels
                    RETURN start.name AS start, start.kind AS start_kind,
                           r.type AS rel_type, neighbor.name AS end,
                           neighbor.kind AS end_kind, r.sources AS source,
                           r.source_doc AS source_doc, r.evidence AS evidence,
                           r.confidence AS confidence
                    LIMIT 120
                    """,
                    rel_hints=rel_hints,
                    allowed_rels=list(ALLOWED_RELATIONS),
                )
                return self._filter_allowed_rows([dict(r) for r in result]), []

            result = session.run(
                """
                MATCH (start:Entity)
                WHERE any(name IN $names WHERE toLower(start.name) CONTAINS toLower(name))
                MATCH (start)-[r:RELATION]-(neighbor)
                WHERE r.type IN $allowed_rels
                  AND (size($rel_hints) = 0
                       OR r.type IN $rel_hints
                       OR any(name IN $names WHERE toLower(neighbor.name) CONTAINS toLower(name)))
                RETURN startNode(r).name AS start, startNode(r).kind AS start_kind,
                       r.type AS rel_type, endNode(r).name AS end,
                       endNode(r).kind AS end_kind, r.sources AS source,
                       r.source_doc AS source_doc, r.evidence AS evidence,
                       r.confidence AS confidence
                LIMIT 150
                """,
                names=entity_names,
                rel_hints=rel_hints,
                allowed_rels=list(ALLOWED_RELATIONS),
            )
            rows_1hop = self._filter_allowed_rows([dict(r) for r in result])

            rows_2hop: list[dict] = []
            if hops >= 2:
                result2 = session.run(
                    """
                    MATCH (start:Entity)
                    WHERE any(name IN $names WHERE toLower(start.name) CONTAINS toLower(name))
                    MATCH (start)-[:RELATION]-(mid)-[r:RELATION]-(neighbor)
                    WHERE NOT any(name IN $names WHERE toLower(neighbor.name) CONTAINS toLower(name))
                      AND r.type IN $allowed_rels
                      AND (size($rel_hints) = 0 OR r.type IN $rel_hints)
                    RETURN startNode(r).name AS start, startNode(r).kind AS start_kind,
                           r.type AS rel_type, endNode(r).name AS end,
                           endNode(r).kind AS end_kind, r.sources AS source,
                           r.source_doc AS source_doc, r.evidence AS evidence,
                           r.confidence AS confidence
                    LIMIT 80
                    """,
                    names=entity_names,
                    rel_hints=rel_hints,
                    allowed_rels=list(ALLOWED_RELATIONS),
                )
                rows_2hop = self._filter_allowed_rows([dict(r) for r in result2])

        return rows_1hop, rows_2hop

    def retrieve_context(self, question: str) -> dict:
        """Retrieve graph facts plus evidence snippets without calling the answer LLM."""
        raw_entities = self.extract_query_entities(question)
        entities = self._filter_seed_entities(raw_entities)
        for name in self.infer_entities_from_graph(question):
            if not _is_noisy_seed(name) and name not in entities:
                entities.append(name)
        entities = self._filter_seed_entities(entities)

        boundary_answer = self._attribute_boundary_refusal(question, entities)
        if boundary_answer:
            return {
                "context": boundary_answer,
                "entities": entities,
                "subgraph": [],
                "boundary_answer": boundary_answer,
            }

        rows_1hop, rows_2hop = self.get_subgraph(entities, question=question)
        subgraph = _rank_subgraph(rows_1hop, rows_2hop, entities)
        context = _textualize(subgraph, entities)

        evidence_chunks = self._evidence_chunks(subgraph, limit=5)
        lexical_chunks = self._lexical_evidence_chunks(question, entities, subgraph, limit=4)
        for chunk in lexical_chunks:
            if chunk not in evidence_chunks:
                evidence_chunks.append(chunk)
        if evidence_chunks:
            context += "\n\nOriginal source chunks:\n" + "\n\n".join(evidence_chunks)

        return {
            "context": context,
            "entities": entities,
            "subgraph": subgraph,
            "boundary_answer": "",
        }

    def answer(self, question: str) -> dict:
        t0 = time.time()
        retrieval = self.retrieve_context(question)
        context = retrieval["context"]
        subgraph = retrieval["subgraph"]

        if retrieval.get("boundary_answer"):
            return {
                "answer": retrieval["boundary_answer"],
                "entities_found": retrieval["entities"],
                "subgraph_triples": 0,
                "top_triples": "",
                "tokens": 0,
                "latency": round(time.time() - t0, 2),
            }

        top_triples = "; ".join(
            f"{r.get('start')}--[{r.get('rel_type')}]-->{r.get('end')}"
            for r in subgraph[:5]
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
                max_completion_tokens=512,
                **chat_kwargs(temperature=0.2),
            )
            answer_text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
        except Exception as e:
            answer_text = f"[Error: {e}]"
            tokens = 0

        return {
            "answer": answer_text,
            "entities_found": retrieval["entities"],
            "subgraph_triples": len(subgraph),
            "top_triples": top_triples,
            "tokens": tokens,
            "latency": round(time.time() - t0, 2),
        }

    def _evidence_chunks(self, subgraph: list[dict], limit: int = 5) -> list[str]:
        chunks: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for row in subgraph:
            for key in _parse_source_refs(row.get("source", "")):
                if key in seen:
                    continue
                seen.add(key)
                text = self.chunk_lookup.get(key)
                if text:
                    source, page, chunk_idx = key
                    chunks.append(f"[{source} p{page} c{chunk_idx}]\n{text}")
                if len(chunks) >= limit:
                    return chunks
        return chunks

    def _lexical_evidence_chunks(
        self,
        question: str,
        entities: list[str],
        subgraph: list[dict],
        limit: int = 4,
    ) -> list[str]:
        if not self.chunks:
            return []

        terms = _lexical_terms(question, entities)
        if not terms:
            return []

        graph_sources = set()
        for row in subgraph:
            graph_sources |= _row_sources(row)

        primary_entities = [e for e in entities if self._is_primary_entity(e)]
        primary_source_aliases = {
            src: _source_aliases(src)
            for src in self.primary_sources
        }

        def score_chunk(chunk: dict) -> int:
            text = str(chunk.get("text", ""))
            text_norm = text.lower()
            source = str(chunk.get("source", ""))
            score = 0

            for term in terms:
                if " " in term:
                    if term in text_norm:
                        score += 5
                elif term in text_norm:
                    score += 2

            for entity in entities:
                if _is_noisy_seed(entity):
                    continue
                if entity.lower() in text_norm:
                    score += 4

            if source in graph_sources:
                score += 2

            for entity in primary_entities:
                entity_norm = re.sub(r"[^a-z0-9]+", " ", entity.lower()).strip()
                if entity_norm in primary_source_aliases.get(source, set()):
                    score += 6

            # Infobox pages are dense with attributes and useful for benchmark facts.
            if chunk.get("page") in (1, 2):
                score += 1
            return score

        ranked = sorted(self.chunks, key=score_chunk, reverse=True)
        results = []
        seen_keys: set[tuple[str, str, str]] = set()
        for chunk in ranked:
            score = score_chunk(chunk)
            if score <= 0:
                break
            key = _chunk_key(chunk.get("source", ""), chunk.get("page", ""), chunk.get("chunk_idx", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            source, page, chunk_idx = key
            results.append(f"[lexical {source} p{page} c{chunk_idx}]\n{chunk.get('text', '')}")
            if len(results) >= limit:
                break
        return results
