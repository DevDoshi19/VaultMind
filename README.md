# VaultMind 

## What is VaultMind
Production-grade RAG system using Dev Doshi's resume PDF as the knowledge base. Built end-to-end covering the full MLOps stack. GitHub: github.com/DevDoshi19/VaultMind

## Current Stack
- **LLM:** gpt-4o-mini (temperature=0)
- **Embeddings:** text-embedding-3-small
- **Pipeline:** LangGraph 7-node graph
- **Vector DB:** ChromaDB (L2 distance, threshold 1.6)
- **Keyword search:** BM25Okapi + RRF merge
- **Backend:** FastAPI + uvicorn + SlowAPI (rate limiting disabled for now)
- **Frontend:** Streamlit calling backend over HTTP via httpx
- **Containerization:** Docker Compose (two containers)
- **CI/CD:** GitHub Actions → ghcr.io
- **Tracing:** LangSmith
- **Evaluation:** RAGAS (isolated, not in main stack)

---

## Current File Structure
```
VaultMind/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          ← pydantic-settings + .env
│   │   ├── state.py           ← RAGState TypedDict
│   │   ├── nodes.py           ← all 7 pipeline nodes
│   │   ├── graph.py           ← LangGraph assembly
│   │   └── ingest.py          ← PDF → chunk → embed → ChromaDB + BM25
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            ← FastAPI app + lifespan + CORS
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py      ← GET /health (shallow + deep probe)
│   │   │   └── query.py       ← POST /api/query
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   └── rate_limit.py  ← SlowAPI (disabled in main.py for now)
│   │   └── auth/
│   │       ├── __init__.py
│   │       └── oauth.py       ← placeholder for Phase 16
│   ├── data/
│   │   ├── resume.pdf         ← gitignored
│   │   └── chunks.json        ← gitignored
│   ├── chroma_db/             ← gitignored
│   ├── assets/
│   │   └── logo.png
│   ├── cli.py                 ← terminal dev tool (renamed from main.py)
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py       ← calls backend over HTTP, no LangGraph import
│   ├── assets/
│   │   └── logo.png
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt       ← only streamlit + httpx
├── evaluation/
│   ├── __init__.py
│   ├── test_dataset.py        ← 14 ground truth Q&A pairs
│   ├── ragas_eval.py          ← RAGAS evaluation script
│   └── requirements.txt       ← ragas==0.1.21 pinned, separate venv
├── .github/
│   └── workflows/
│       └── ci.yml             ← 3 jobs: backend-checks, frontend-checks, build-and-push
├── docker-compose.yml
├── .env                       ← gitignored
└── .gitignore
```

---

## Pipeline Architecture (7 nodes)
```
input_guardrail_node
    → pattern match + LLM safety check
    → if blocked: route to END (skip everything)

classify_query_node
    → is question about Dev Doshi? yes/no
    → if no: route to generate (early exit)

retrieve_node
    → hybrid search: ChromaDB semantic + BM25 keyword + RRF merge
    → retrieval_k=3, similarity_threshold=1.6

context_guard_node
    → token budget check (tiktoken, max 1000 tokens)
    → drops chunks that exceed budget

generate_node
    → GPT-4o-mini with tenacity retry (3 attempts, exponential backoff)
    → if not relevant or empty: early exit message

output_guardrail_node
    → PII redaction (regex) + LLM safety check

confidence_node
    → self-RAG scoring (0.0 to 1.0)
    → appends warning if score < 0.7
```

## RAGState fields
```python
question: str
question_is_relevant: bool
retrieved_docs: list[str]
retrieval_status: str
context_token_count: int
answer: str
confidence_score: float | None
prompt_tokens: int
completion_tokens: int
total_tokens: int
estimated_cost: float
input_blocked: bool
output_flagged: bool
```

---

## API Contract
**POST /api/query**
```json
Request:  { "question": "string (1-500 chars)" }
Response: {
  "answer": "string",
  "confidence_score": "float | null",
  "input_blocked": "bool",
  "output_flagged": "bool",
  "total_tokens": "int",
  "estimated_cost": "float",
  "retrieval_status": "string",
  "retrieved_chunks": "int"
}
```

**GET /health** → shallow probe
**GET /health?deep=true** → checks graph compiled, returns node list

---

## Docker Setup
```bash
# run everything
docker compose up --build

# backend: http://localhost:8000
# frontend: http://localhost:8501

# containers talk via Docker network
# frontend calls: http://backend:8000/api/query
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
```

Volumes mounted:
- `./backend/chroma_db:/app/chroma_db`
- `./backend/data:/app/data`

---

## CI/CD
Three jobs on every push/PR to main:
- **backend-checks** — import verification for pipeline + API layer
- **frontend-checks** — import verification for streamlit + httpx
- **build-and-push** — builds Docker images, smoke tests `/health`, pushes to ghcr.io on main merge

Images live at:
- `ghcr.io/devdoshi19/vaultmind-backend:latest`
- `ghcr.io/devdoshi19/vaultmind-frontend:latest`

---

## RAGAS Scores (14 test cases)
```
faithfulness:      0.993
answer_relevancy:  0.906
context_precision: 0.738
context_recall:    0.821
overall:           0.865

```

---

## Known Issues / Deferred
- SlowAPI rate limiting commented out in `main.py` — middleware import fixed but disabled
- Blocked input still continues past guardrail into classifier (conditional edge issue in graph.py)
- Duplicate chunks in ChromaDB from PDF ingestion
- `auth/oauth.py` is a placeholder — Phase 16

---

## Key Technical Decisions
- **ChromaDB returns L2 distance** — lower = more relevant, threshold 1.6
- **BM25 requires chunks.json** — saved during ingest, can't use ChromaDB vectors
- **`lifespan()` pattern** — graph compiled once at startup, stored on `app.state.graph`
- **`run_in_executor`** — LangGraph invoke is synchronous, must run in thread pool inside async FastAPI
- **RAGAS isolated** — `evaluation/requirements.txt` only, never in backend image
- **`getattr(request.app.state, "graph", None)`** — defensive access in case lifespan failed

---

## What's Next — Remaining Phases

### Phase 14 — Kubernetes
Deploy both services as Kubernetes pods using minikube locally.

Files to write:
- `k8s/backend-deployment.yaml` — 2 replicas, resource limits, env from secret
- `k8s/frontend-deployment.yaml` — 1 replica
- `k8s/backend-service.yaml` — ClusterIP, exposes port 8000 inside cluster
- `k8s/frontend-service.yaml` — NodePort, exposes 8501 to outside
- `k8s/secret.yaml` — OPENAI_API_KEY as K8s secret (base64 encoded)

Key concepts to understand before starting:
- Pod vs Deployment vs Service
- Why ClusterIP for backend (internal only) and NodePort for frontend (external)
- How K8s pulls from ghcr.io (imagePullSecrets)
- Liveness vs readiness probes (already have /health and /health?deep=true)

Run with: `minikube start` then `kubectl apply -f k8s/`

### Phase 15 — Redis Caching + Pinecone
Two upgrades, can be done separately:

**Redis:**
- Cache `POST /api/query` responses by question hash
- Cache key: `md5(question.lower().strip())`
- TTL: 24 hours
- Changes only `query.py` — check cache before invoking graph, write to cache after
- Add `redis` service to docker-compose.yml

**Pinecone:**
- Replace ChromaDB (local only, can't share between K8s replicas)
- Changes: `ingest.py` push to Pinecone, `nodes.py` retrieve_node queries Pinecone
- API contract unchanged

### Phase 16 — Auth + PostgreSQL + Per-user history
Three things in dependency order:

**16a — PostgreSQL + SQLAlchemy:**
- Users table: id, email, hashed_password, created_at
- Conversations table: id, user_id, created_at
- Messages table: id, conversation_id, role, content, metadata, created_at
- Alembic for migrations
- Add `postgres` service to docker-compose.yml

**16b — OAuth2 + JWT:**
- Fill in `auth/oauth.py` placeholder
- `POST /auth/register` and `POST /auth/login` endpoints
- JWT tokens, `/api/query` requires valid token
- `user_id` extracted from token, replaces IP-based rate limiting

**16c — Per-user history in frontend:**
- History sidebar in Streamlit
- `GET /api/conversations` endpoint
- `GET /api/conversations/{id}/messages` endpoint

### Phase 17 — Grafana + Prometheus Monitoring
- FastAPI exports metrics via `prometheus-fastapi-instrumentator`
- Prometheus scrapes metrics every 15s
- Grafana dashboard showing:
  - Queries per hour
  - Average confidence score over time
  - Cost per day
  - Blocked query rate
  - P95 response latency
- Add `prometheus` and `grafana` services to docker-compose.yml

---

## How to Run Locally
```bash
# terminal 1 — backend
cd VaultMind/backend
uvicorn api.main:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend
cd VaultMind/frontend
streamlit run streamlit_app.py

# or with Docker
cd VaultMind
docker compose up
```

## Environment Variables needed in .env
```
OPENAI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=vaultmind
```