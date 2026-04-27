# 🧠 Case Insights AI Dashboard

AI-powered support case categorization with DistilBERT fine-tuning, active learning, and interactive analytics.

---

## 🚀 Running with Docker (Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### One-command start

```bash
# Build and start
docker compose up --build

# Or run in background
docker compose up --build -d
```

Open **http://localhost:8000** in your browser.

> **First run:** The build downloads the `all-MiniLM-L6-v2` model (~90MB) and installs all dependencies. This takes 5–10 minutes. Subsequent starts are instant.

### Stop the container
```bash
docker compose down
```

### What persists between restarts
| Data | Location |
|---|---|
| BERT fine-tuned model | `bert_model` Docker volume |
| Human feedback corrections | `./data/feedback.json` |
| LinearSVC model cache | `./data/model_cache/` |

---

## 💻 Running Locally (Development)

### Prerequisites
- Python 3.12
- pip or conda

```bash
cd dashboard_app/backend
pip install -r requirements.txt
python3 main.py
```

Open **http://localhost:8000**

---

## 🏗️ Project Structure

```
Issue_Dashboard/
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Service definition + volumes
├── .dockerignore               # Excludes data/models from build context
├── data/
│   ├── feedback.json           # Persisted human corrections
│   └── model_cache/            # LinearSVC cached models
└── dashboard_app/
    ├── backend/
    │   ├── main.py             # FastAPI app + all endpoints
    │   ├── predictor.py        # 4-tier ML pipeline (Keyword→SVC→BERT→Semantic)
    │   ├── bert_trainer.py     # DistilBERT fine-tuning engine
    │   ├── requirements.txt    # Pinned Python dependencies
    │   └── bert_model/         # Saved fine-tuned BERT model (after training)
    └── frontend/
        ├── index.html          # Dashboard UI
        ├── app.js              # All frontend logic
        └── styles.css          # Glassmorphism theme
```

---

## 🧠 AI Architecture

| Tier | Method | Speed | Accuracy |
|---|---|---|---|
| 1 | Keyword Booster | ~0.001ms | Rule-based |
| 2 | Exact Match | ~0.01ms | 100% on matched |
| 3 | LinearSVC (TF-IDF) | ~0.1ms | ~82% |
| 4 | Semantic Embeddings | ~5ms | ~78% fallback |
| Optional | DistilBERT (fine-tuned) | ~15ms | ~92–95% |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (optional, avoids rate limits) |
| `PORT` | 8000 | API port |
