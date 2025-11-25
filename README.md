## Anomaly Detection Platform

A full-stack fraud analytics workspace that bundles a PyTorch autoencoder, an XGBoost classifier, a FastAPI inference service, and a Streamlit command center. Use it to ingest the Kaggle credit card fraud dataset, train an ensemble detector, expose real-time analytics APIs, and explore insights via an interactive dashboard.

### Highlights
- **Hybrid modeling pipeline** – trains an autoencoder on normal traffic and feeds its reconstruction error into an XGBoost classifier for high recall on rare fraud events.
- **Async FastAPI backend** – serves transaction stats, visualizations, and live predictions backed by PostgreSQL.
- **Executive-grade Streamlit UI** – pre-built dashboards, playground predictor, and Plotly visuals pointing to the API service.
- **Container-ready** – each service ships with its own `Dockerfile` for Hugging Face Spaces or self-managed deployments.

## Repository Layout
```
.
├── train/         # Offline ingestion + model training code
├── server/        # FastAPI app, SQLAlchemy models, serialized weights
├── streamlit/     # Streamlit dashboard that consumes the API
└── README.md
```

## Prerequisites
- Python 3.10+
- PostgreSQL 14+ (cloud-hosted works; defaults to the Render connection string found in the env files)
- Recommended: `pyenv`/`venv` and Docker 24+ if you plan to containerize.

## Quick Start
1. **Clone & enter repo**
   ```bash
   git clone <repo-url>
   cd anomaly_detection
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Set common environment variables**
   ```bash
   export DATABASE_URL="postgresql://<user>:<pass>@<host>:<port>/<db>"
   ```
4. Follow the component-specific steps below.

## Data Ingestion & Model Training (`train/`)
1. **Install deps**
   ```bash
   pip install -r train/requirements.txt
   ```
2. **Download the Kaggle credit card fraud CSV** into `train/data/creditcard.csv`.
3. **Ingest raw transactions** into PostgreSQL:
   ```bash
   cd train
   python src/ingest_csv.py
   ```
4. **Train the ensemble & export artifacts**:
   ```bash
   python src/train.py
   ```
   Outputs:
   - `models/scaler.pkl`
   - `models/autoencoder.pth`
   - `models/xgboost.json`

The autoencoder only sees normal transactions, while the downstream XGBoost model receives the reconstruction error as an additional feature, improving class separation amid heavy imbalance.

## Analytics API (`server/`)
1. **Install deps**
   ```bash
   pip install -r server/requirements.txt
   ```
2. **Ensure `DATABASE_URL` plus model artifacts exist under `server/models/`.**
3. **Run locally**
   ```bash
   cd server
   uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
   ```
4. **Core endpoints**

| Method | Path                        | Description |
|--------|-----------------------------|-------------|
| GET    | `/`                         | Service heartbeat |
| GET    | `/health`                   | Model + DB readiness |
| GET    | `/api/stats`                | Volume & fraud rate summary |
| GET    | `/api/transactions`         | Paginated transactions with class filter |
| GET    | `/api/anomalies`            | Confirmed fraud only |
| GET    | `/api/distributions/amount` | Histograms split by class |
| GET    | `/api/scatter`              | Feature-pair scatter plot data |
| GET    | `/api/model/performance`    | Live metrics, ROC, feature importance |
| POST   | `/api/predict`              | Score a single transaction & persist result |

The service lazily loads the scaler, autoencoder, and gradient-boosting model at startup, so keep latency low by mounting the `models/` folder alongside the app.

## Streamlit Dashboard (`streamlit/`)
1. **Install deps**
   ```bash
   pip install -r streamlit/requirements.txt
   ```
2. **Configure API target**
   - Update `API_BASE_URL` in `streamlit/src/streamlit_app.py` when pointing to a non-default backend.
3. **Run locally**
   ```bash
   cd streamlit
   streamlit run src/streamlit_app.py --server.port 8501
   ```
4. **Features**
   - Executive KPIs + auto-refresh transaction table.
   - Fraud predictor that samples from bundled `normal.csv` / `fraud.csv` and calls `/api/predict`.
   - Distribution, feature histogram, and scatter visuals sourced from backend analytics endpoints.

## Docker & Deployment
- **FastAPI server**
  ```bash
  cd server
  docker build -t anomaly-server .
  docker run -p 8000:8000 -e DATABASE_URL=... anomaly-server
  ```
- **Streamlit app**
  ```bash
  cd streamlit
  docker build -t anomaly-streamlit .
  docker run -p 8501:8501 -e API_BASE_URL=http://host:8000 anomaly-streamlit
  ```

Both images are compatible with Hugging Face Spaces (see the metadata headers in each `README.md`).

## Development Notes
- Keep the database schema in sync via SQLAlchemy models inside `server/src/schemas.py`.
- The Streamlit app caches (`st.cache_data`) API responses for 60 seconds to avoid hammering the backend.
- When retraining, copy the freshly generated artifacts from `train/models/` into `server/models/` before redeploying.
- Protect credentials: never commit plain `DATABASE_URL` strings; use `.env` or secret managers in production.
