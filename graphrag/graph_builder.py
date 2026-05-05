from neo4j import GraphDatabase


def _fact_from_item(item, source_doc: str = "", page: int | None = None,
                    chunk_idx: int | None = None) -> dict:
    """Accept new fact dicts and old (subject, relation, object) tuples."""
    if isinstance(item, dict):
        fact = dict(item)
    else:
        subject, relation, obj = item
        fact = {
            "subject": subject,
            "subject_type": "Other",
            "relation": relation,
            "object": obj,
            "object_type": "Other",
            "evidence": "",
            "confidence": 0.8,
        }

    fact.setdefault("subject_type", "Other")
    fact.setdefault("object_type", "Other")
    fact.setdefault("evidence", "")
    fact.setdefault("confidence", 0.8)
    fact["source_doc"] = source_doc
    fact["source_ref"] = _source_ref(source_doc, page, chunk_idx)
    fact["page"] = page
    fact["chunk_idx"] = chunk_idx
    return fact


def _source_ref(source_doc: str = "", page: int | None = None,
                chunk_idx: int | None = None) -> str:
    if not source_doc:
        return ""
    parts = [source_doc]
    if page is not None:
        parts.append(f"p{page}")
    if chunk_idx is not None:
        parts.append(f"c{chunk_idx}")
    return " ".join(parts)


class Neo4jGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def setup_schema(self):
        """Create uniqueness constraint on Entity.name."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def clear_graph(self):
        """Delete all nodes and relationships (use with caution)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # Pipe-separated strings keep this compatible with older Neo4j installs.
    _MERGE_FACT_CQL = """
        MERGE (a:Entity {name: $subj})
        SET a.kind = coalesce(a.kind, $subj_type)
        MERGE (b:Entity {name: $obj})
        SET b.kind = coalesce(b.kind, $obj_type)
        MERGE (a)-[r:RELATION {type: $rel}]->(b)
        ON CREATE SET
            r.source_doc = $source_doc,
            r.sources = $source_ref,
            r.evidence = $evidence,
            r.confidence = $confidence,
            r.pages = $page_ref,
            r.chunks = $chunk_ref
        ON MATCH SET
            r.source_doc = coalesce(r.source_doc, $source_doc),
            r.confidence = CASE
                WHEN r.confidence IS NULL OR $confidence > r.confidence THEN $confidence
                ELSE r.confidence
            END,
            r.sources = CASE
                WHEN $source_ref = '' THEN r.sources
                WHEN r.sources IS NULL THEN $source_ref
                WHEN $source_ref IN split(r.sources, '|') THEN r.sources
                ELSE r.sources + '|' + $source_ref
            END,
            r.evidence = CASE
                WHEN $evidence = '' THEN r.evidence
                WHEN r.evidence IS NULL THEN $evidence
                WHEN $evidence IN split(r.evidence, '|') THEN r.evidence
                ELSE r.evidence + '|' + $evidence
            END,
            r.pages = CASE
                WHEN $page_ref = '' THEN r.pages
                WHEN r.pages IS NULL THEN $page_ref
                WHEN $page_ref IN split(r.pages, '|') THEN r.pages
                ELSE r.pages + '|' + $page_ref
            END,
            r.chunks = CASE
                WHEN $chunk_ref = '' THEN r.chunks
                WHEN r.chunks IS NULL THEN $chunk_ref
                WHEN $chunk_ref IN split(r.chunks, '|') THEN r.chunks
                ELSE r.chunks + '|' + $chunk_ref
            END
    """

    def upsert_fact(self, fact: dict, source_doc: str = "", page: int | None = None,
                    chunk_idx: int | None = None):
        """Merge one structured fact, accumulating sources/evidence."""
        fact = _fact_from_item(fact, source_doc, page, chunk_idx)
        with self.driver.session() as session:
            session.run(self._MERGE_FACT_CQL, **self._params(fact))

    def upsert_triple(self, subject: str, relation: str, obj: str, source_doc: str = ""):
        """Backward-compatible wrapper for old tuple triples."""
        self.upsert_fact((subject, relation, obj), source_doc=source_doc)

    def bulk_upsert(self, triples: list, source_doc: str = "",
                    page: int | None = None, chunk_idx: int | None = None):
        """Insert multiple facts/triples in a single session, accumulating evidence."""
        with self.driver.session() as session:
            for item in triples:
                fact = _fact_from_item(item, source_doc, page, chunk_idx)
                session.run(self._MERGE_FACT_CQL, **self._params(fact))

    @staticmethod
    def _params(fact: dict) -> dict:
        page = fact.get("page")
        chunk_idx = fact.get("chunk_idx")
        return {
            "subj": str(fact.get("subject", "")),
            "subj_type": str(fact.get("subject_type", "Other")),
            "rel": str(fact.get("relation", "")),
            "obj": str(fact.get("object", "")),
            "obj_type": str(fact.get("object_type", "Other")),
            "source_doc": str(fact.get("source_doc", "")),
            "source_ref": str(fact.get("source_ref", "")),
            "evidence": str(fact.get("evidence", "")),
            "confidence": float(fact.get("confidence", 0.8) or 0.8),
            "page_ref": "" if page is None else str(page),
            "chunk_ref": "" if chunk_idx is None else str(chunk_idx),
        }

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}

    def get_all_entity_names(self) -> list[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n.name AS name")
            return [r["name"] for r in result]
