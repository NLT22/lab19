from neo4j import GraphDatabase


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

    def upsert_triple(self, subject: str, relation: str, obj: str, source_doc: str = ""):
        """Merge nodes and relationship for a single triple."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (a:Entity {name: $subj})
                MERGE (b:Entity {name: $obj})
                MERGE (a)-[r:RELATION {type: $rel}]->(b)
                SET r.source_doc = $source_doc
                """,
                subj=subject,
                obj=obj,
                rel=relation,
                source_doc=source_doc,
            )

    def bulk_upsert(self, triples: list[tuple[str, str, str]], source_doc: str = ""):
        """Insert multiple triples in a single session."""
        with self.driver.session() as session:
            for subject, relation, obj in triples:
                session.run(
                    """
                    MERGE (a:Entity {name: $subj})
                    MERGE (b:Entity {name: $obj})
                    MERGE (a)-[r:RELATION {type: $rel}]->(b)
                    SET r.source_doc = $source_doc
                    """,
                    subj=subject,
                    obj=obj,
                    rel=relation,
                    source_doc=source_doc,
                )

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}

    def get_all_entity_names(self) -> list[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n.name AS name")
            return [r["name"] for r in result]
