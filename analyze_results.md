# Analysis of GraphRAG Benchmark Results

This report summarizes the 36-question grounded benchmark in `results/comparison_3way.csv`, comparing three retrieval methods:

- **Flat RAG**: vector search over PDF chunks.
- **GraphRAG**: Neo4j graph facts plus graph-guided evidence snippets.
- **Hybrid RAG**: GraphRAG context combined with vector-retrieved passages.

## Overall Results

| Method | Score | Accuracy | Avg tokens/question | Avg latency |
|---|---:|---:|---:|---:|
| Flat RAG | 105/143 | 73.4% | 1,093.5 | 2.24s |
| GraphRAG | 120/143 | 83.9% | 3,210.0 | 2.73s |
| Hybrid RAG | 125/143 | 87.4% | 4,258.7 | 4.85s |

Hybrid RAG achieved the best total score. The gain comes from combining graph-structured relations with raw textual evidence, which helps on questions that need both precise numbers and multi-hop connections.

## Results by Difficulty

| Difficulty | Flat RAG | GraphRAG | Hybrid RAG |
|---|---:|---:|---:|
| Easy (10 questions, 32 pts) | 19/32 (59%) | 28/32 (88%) | 26/32 (81%) |
| Medium (20 questions, 82 pts) | 69/82 (84%) | 70/82 (85%) | 76/82 (93%) |
| Hard (6 questions, 29 pts) | 17/29 (59%) | 22/29 (76%) | 23/29 (79%) |

GraphRAG performs especially well on easy factual questions after literal attributes such as revenue, net income, employee count, market share, and ownership share were extracted into the graph. Hybrid RAG performs best on medium and hard questions because it can combine graph relationships with vector evidence.

## Best-Method Wins

| Outcome | Count |
|---|---:|
| Flat RAG best alone | 1 |
| GraphRAG best alone | 2 |
| Hybrid RAG best alone | 4 |
| Tie for best | 29 |

Most questions are ties because the scoring checks required terms, and multiple methods often included the same gold facts. The important signal is that Hybrid has the highest total score while GraphRAG also clearly improves over Flat RAG.

## Method-Level Interpretation

### Flat RAG

Flat RAG is fast and token-efficient. It works well when the answer appears directly inside a retrieved chunk, such as financial values or infobox lists. Its main weakness is recall: if vector search misses the exact chunk, the answer is incomplete or unavailable.

Flat RAG struggled more on cross-company and relation-heavy questions, for example finding companies connected to both Microsoft and Nvidia.

### GraphRAG

GraphRAG improved substantially after the graph schema was expanded beyond entity-to-entity triples. It now captures literal facts such as:

- revenue and net income;
- employee counts;
- ownership shares;
- market share;
- valuations;
- investments and partnerships;
- source references and evidence snippets.

GraphRAG is better than Flat RAG at corpus-boundary and relationship questions. For example, it handles questions such as Apple's revenue by recognizing that Apple is not a primary corpus source, instead of relying on incidental mentions.

Its main limitation is extraction quality. If a fact is not extracted into the graph or is extracted under a noisy relation, GraphRAG may still need raw text evidence to answer reliably.

### Hybrid RAG

Hybrid RAG is the strongest method overall. It uses:

1. graph facts for structure and entity relationships;
2. vector chunks for exact textual evidence.

This combination helps answer questions that need both precise numbers and relationship reasoning. The trade-off is cost: Hybrid RAG uses the most tokens and has the highest latency.

## Key Takeaways

1. **Flat RAG is a strong baseline** for Wikipedia-style documents because many answers are present verbatim in chunks.
2. **GraphRAG becomes useful only after extracting literal attributes**, not just named-entity triples.
3. **Hybrid RAG gives the best quality** because graph structure improves retrieval focus while vector passages preserve exact evidence.
4. **Token cost is the main downside** of Hybrid RAG. It achieved the best score, but used about 3.9x the tokens of Flat RAG.
5. **GraphRAG is most valuable on relationship-heavy and corpus-boundary questions**, while Flat RAG remains efficient for direct lookup.

## Files Produced

- `results/comparison.csv`: latest two-method benchmark output.
- `results/comparison_3way.csv`: full Flat RAG vs GraphRAG vs Hybrid RAG benchmark.
- `results/visualisation_Limit_1000.png`: graph visualization artifact.

## Final Conclusion

The final benchmark shows that GraphRAG is no longer just a weaker version of Flat RAG. With literal facts, evidence tracking, relation filtering, and corpus-boundary handling, GraphRAG improves over Flat RAG by about 10.5 percentage points.

Hybrid RAG performs best overall, reaching 87.4% accuracy. This confirms the practical lesson of the lab: for encyclopedia-style corpora, the best system is not pure graph retrieval or pure vector retrieval, but a hybrid pipeline that uses the graph for structure and vector search for exact supporting text.
