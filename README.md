# Lab 19 - GraphRAG với Tech Company Corpus

Dự án xây dựng pipeline **GraphRAG** cho corpus Wikipedia về các công ty công nghệ, sau đó so sánh với **Flat RAG** và **Hybrid RAG** trên bộ benchmark 36 câu hỏi.

Mục tiêu chính của lab:

- ingest PDF thành text chunks;
- dùng LLM trích xuất entity, relation và literal facts;
- lưu knowledge graph vào Neo4j;
- xây dựng vector index bằng FAISS;
- truy vấn bằng 3 phương pháp: Flat RAG, GraphRAG, Hybrid RAG;
- đánh giá chất lượng bằng benchmark có gold answer.

---

## Kết Quả Hiện Tại

Benchmark đầy đủ nằm ở `results/comparison_3way.csv`, báo cáo phân tích nằm ở `results/analyze_results.md`.

| Method | Score | Accuracy | Avg tokens/question | Avg latency |
|---|---:|---:|---:|---:|
| Flat RAG | 105/143 | 73.4% | 1,093.5 | 2.24s |
| GraphRAG | 120/143 | 83.9% | 3,210.0 | 2.73s |
| Hybrid RAG | 125/143 | 87.4% | 4,258.7 | 4.85s |

Kết luận ngắn:

- **Flat RAG** mạnh ở câu hỏi có đáp án xuất hiện nguyên văn trong chunk.
- **GraphRAG** tốt hơn sau khi graph lưu thêm literal facts như revenue, employee count, market share, ownership share và evidence snippets.
- **Hybrid RAG** đạt điểm cao nhất vì kết hợp graph facts với raw text chunks, đổi lại tốn token và latency hơn.

---

## Kiến Trúc Tổng Quan

```text
PDF files
   |
   v
01_ingest_pdfs.py
   |-- data/chunks.json
   v
02_index.py
   |-- Neo4j knowledge graph
   |-- data/faiss.index
   |-- data/faiss_meta.pkl
   v
03_query.py
   |-- Flat RAG
   |-- GraphRAG
   |-- Hybrid RAG
   v
04_evaluate.py
   |-- results/comparison.csv
   |-- results/comparison_3way.csv
```

### Ba phương pháp truy vấn

| Method | Retrieval | Điểm mạnh |
|---|---|---|
| Flat RAG | FAISS vector search trên text chunks | Nhanh, tốt với facts nằm nguyên trong văn bản |
| GraphRAG | Neo4j graph traversal + evidence snippets | Tốt với entity relationship, multi-hop, corpus-boundary |
| Hybrid RAG | Graph context + vector chunks | Chất lượng cao nhất, giữ được cả structure và exact evidence |

---

## Cài Đặt

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Sao chép `.env.example` thành `.env`, sau đó điền cấu hình cần thiết:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

LM_STUDIO_URL=http://127.0.0.1:1234
LM_STUDIO_CHAT_MODEL=openai/gpt-oss-20b
LM_STUDIO_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Nếu không có `OPENAI_API_KEY`, project sẽ dùng LM Studio local.

---

## Dữ Liệu

Corpus hiện tại nằm trong `data/pdfs/`:

```text
data/pdfs/
├── Anthropic.pdf
├── Meta_Platforms.pdf
├── Microsoft.pdf
├── Nvidia.pdf
└── OpenAI.pdf
```

Các artifact được tạo sau khi ingest/index:

- `data/chunks.json`: text chunks từ PDF;
- `data/faiss.index`: FAISS vector index;
- `data/faiss_meta.pkl`: metadata cho vector chunks;
- Neo4j graph: nodes, relationships, literal facts và source evidence.

---

## Chạy Pipeline

### 1. Ingest PDF

```bash
python 01_ingest_pdfs.py
```

Script đọc PDF trong `data/pdfs/`, chia thành chunks và lưu vào `data/chunks.json`.

### 2. Index graph và vector

```bash
python 02_index.py
```

Nếu muốn xoá graph cũ rồi index lại:

```bash
python 02_index.py --clear
```

Kiểm tra graph trong Neo4j Browser:

```cypher
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100
```

### 3. Query tương tác

```bash
python 03_query.py
```

Các mode hỗ trợ:

```text
flat    - chỉ chạy Flat RAG
graph   - chỉ chạy GraphRAG
hybrid  - chỉ chạy Hybrid RAG
both    - so sánh Flat RAG và GraphRAG
all     - chạy cả Flat RAG, GraphRAG và Hybrid RAG
quit    - thoát
```

Mặc định console chạy mode `all`.

### 4. Benchmark

Chạy toàn bộ 36 câu:

```bash
python 04_evaluate.py --delay 0 --output results/comparison_3way.csv
```

Chạy thử N câu đầu:

```bash
python 04_evaluate.py --limit 5 --delay 0 --output results/comparison_3way_sample.csv
```

Trong đó `--limit 5` nghĩa là chỉ chạy 5 câu đầu của benchmark để test nhanh, không chạy đủ 36 câu.

---

## Cấu Trúc Code

```text
Lab19/
├── 01_ingest_pdfs.py          # PDF -> chunks
├── 02_index.py                # chunks -> Neo4j graph + FAISS index
├── 03_query.py                # interactive query console
├── 04_evaluate.py             # benchmark evaluator
├── graphrag/
│   ├── config.py              # LLM/embed client factory
│   ├── pdf_loader.py          # PDF loading + chunking
│   ├── extractor.py           # LLM extraction prompt + parsing
│   ├── graph_builder.py       # Neo4j write logic
│   ├── retriever.py           # GraphRAG retrieval + answer
│   ├── flat_rag.py            # FAISS retrieval + answer
│   ├── hybrid_rag.py          # GraphRAG + Flat RAG hybrid retrieval
│   └── evaluator.py           # benchmark cases + scoring
├── data/
│   ├── chunks.json
│   ├── faiss.index
│   ├── faiss_meta.pkl
│   └── pdfs/
└── results/
    ├── analyze_results.md
    ├── comparison.csv
    ├── comparison_3way.csv
    └── visualisation_Limit_1000.png
```

---

## Neo4j Schema

Graph lưu cả entity relationships và literal facts.

```text
(:Entity {name})
  -[:RELATION {type, source_doc, evidence}]->
(:Entity {name})
```

Ví dụ:

```text
(OpenAI) --[FOUNDED_BY]--> (Sam Altman)
(OpenAI) --[FOUNDED_IN]--> (2015)
(Microsoft) --[INVESTED_IN]--> (OpenAI)
(Nvidia) --[HAS_REVENUE]--> ($130.5 billion)
```

Ngoài triple thông thường, extractor cũng cố gắng giữ lại các literal attributes quan trọng để GraphRAG trả lời được câu hỏi dạng "bao nhiêu", "năm nào", "tỷ lệ bao nhiêu".

---

## Results

Các file kết quả chính:

- `results/comparison.csv`: benchmark hai phương pháp cũ.
- `results/comparison_3way.csv`: benchmark Flat RAG vs GraphRAG vs Hybrid RAG.
- `results/analyze_results.md`: phân tích kết quả và nhận xét.
- `results/visualisation_Limit_1000.png`: ảnh visualization graph.

---

## Deliverables

- Mã nguồn pipeline GraphRAG, Flat RAG và Hybrid RAG.
- Knowledge graph trên Neo4j.
- FAISS index cho Flat/Hybrid RAG.
- Benchmark CSV trong `results/`.
- Báo cáo phân tích trong `results/analyze_results.md`.
