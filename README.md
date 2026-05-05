# Lab 19 — GraphRAG với Tech Company Corpus

Xây dựng hệ thống **GraphRAG** (Graph-based Retrieval Augmented Generation) từ các bài viết Wikipedia về công ty công nghệ, so sánh với Flat RAG truyền thống.

---

## Kiến trúc tổng quan

```
PDF files (Wikipedia)
       │
       ▼
 01_ingest_pdfs.py      ← pypdf, sliding-window chunking
       │ chunks.json
       ▼
 02_index.py            ← LLM extract triples → Neo4j Desktop
       │ Knowledge Graph
       ▼
 03_query.py            ← Interactive REPL: GraphRAG vs Flat RAG
 04_evaluate.py         ← 20-question benchmark → results/comparison.csv
```

### LLM & Embedding

| Điều kiện | Chat LLM | Embedding |
|---|---|---|
| Có `OPENAI_API_KEY` | OpenAI `gpt-4o-mini` | OpenAI `text-embedding-3-small` |
| Không có key | LM Studio `openai/gpt-oss-20b` | LM Studio `text-embedding-nomic-embed-text-v1.5` |

---

## Cài đặt

```bash
# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Cài thêm các thư viện cần thiết
pip install -r requirements.txt
```

---

## Cấu hình

Sao chép và điền thông tin vào `.env`:

```env
# Để trống nếu muốn dùng LM Studio hoàn toàn offline
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# LM Studio (cần chạy server tại localhost:1234)
LM_STUDIO_URL=http://127.0.0.1:1234
LM_STUDIO_CHAT_MODEL=openai/gpt-oss-20b
LM_STUDIO_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5

# Neo4j Desktop
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Yêu cầu LM Studio (nếu chạy offline)

Trong LM Studio, cần load sẵn 2 model:
- **Chat:** `openai/gpt-oss-20b`
- **Embedding:** `text-embedding-nomic-embed-text-v1.5`

---

## Dữ liệu

Tải PDF từ Wikipedia (Export → Download as PDF) về các công ty:

```
data/pdfs/
├── Google.pdf
├── OpenAI.pdf
├── Microsoft.pdf
├── Apple_Inc.pdf
├── Tesla_Inc.pdf
├── Meta_Platforms.pdf
├── Amazon.pdf
├── Nvidia.pdf
└── ...
```

---

## Chạy pipeline

### Bước 1 — Ingest PDF

```bash
python 01_ingest_pdfs.py
```

Xuất ra `data/chunks.json` với các đoạn text đã chunk (800 ký tự, overlap 100).

### Bước 2 — Index vào Neo4j

```bash
# Start Neo4j Desktop trước, sau đó:
python 02_index.py

# Xóa graph cũ và index lại:
python 02_index.py --clear
```

Kiểm tra đồ thị tại **Neo4j Browser**: `http://localhost:7474`
```cypher
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100
```

### Bước 3 — Query tương tác

```bash
python 03_query.py
```

```
> Who founded Google?
[Flat RAG]  Google was founded by Larry Page and Sergey Brin...
[GraphRAG]  Google --[FOUNDED_BY]--> Larry Page, Sergey Brin...

# Đổi mode:
> flat      ← chỉ dùng Flat RAG
> graph     ← chỉ dùng GraphRAG
> both      ← so sánh song song (mặc định)
> quit
```

### Bước 4 — Benchmark 20 câu hỏi

```bash
python 04_evaluate.py
```

Kết quả lưu tại `results/comparison.csv` gồm: câu hỏi, câu trả lời của cả hai hệ thống, số token, độ trễ.

---

## Cấu trúc code

```
Lab19/
├── .env                        # Credentials (không commit)
├── requirements.txt
├── graphrag/
│   ├── config.py               # LLM/embed client factory
│   ├── pdf_loader.py           # PDF → chunks
│   ├── extractor.py            # LLM → (subject, relation, object) triples
│   ├── graph_builder.py        # Neo4j MERGE nodes + relationships
│   ├── retriever.py            # 2-hop Cypher traversal → LLM answer
│   ├── flat_rag.py             # FAISS + embeddings → LLM answer
│   └── evaluator.py            # 20-question benchmark
├── 01_ingest_pdfs.py
├── 02_index.py
├── 03_query.py
└── 04_evaluate.py
```

---

## Neo4j Schema

```
(:Entity {name: String})
  -[:RELATION {type: String, source_doc: String}]->
(:Entity {name: String})
```

Ví dụ triples được trích xuất:
```
(OpenAI) --[FOUNDED_BY]--> (Sam Altman)
(OpenAI) --[FOUNDED_BY]--> (Elon Musk)
(OpenAI) --[FOUNDED_IN]--> (2015)
(Microsoft) --[INVESTED_IN]--> (OpenAI)
```

---

## Deliverables (nộp bài)

- [ ] Mã nguồn (thư mục `graphrag/` + 4 script)
- [ ] Ảnh chụp màn hình Neo4j Browser (Knowledge Graph)
- [ ] `results/comparison.csv` (20 câu hỏi benchmark)
- [ ] Phân tích chi phí token và thời gian
