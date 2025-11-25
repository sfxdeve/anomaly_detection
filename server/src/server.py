import pickle
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Local Imports
from src.db import init_db, get_db
from src.models.autoencoder import Autoencoder
from src.models.xgboost_model import XGBoostModel
from src.schemas import (
    TransactionTable,
    TransactionCreate,
    TransactionResponse,
    TransactionStats,
    DistributionData,
    ScatterData,
    ScatterPoint,
    ModelPerformance,
    ModelMetrics,
    ConfusionMatrix,
    RocCurve,
    FeatureImportance
)

# ============================================================================
# Configuration & Logging
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Global model storage
models: Dict[str, Any] = {
    "scaler": None,
    "autoencoder": None,
    "xgboost": None
}

# ============================================================================
# Helper Functions
# ============================================================================

def _run_inference_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centralized logic to preprocess data, run autoencoder, and prepare for XGBoost.
    Returns the DataFrame ready for prediction.
    """
    # 1. Scale Data
    scaler = models["scaler"]
    X_scaled = scaler.transform(df[VALID_FEATURES])
    X_scaled_df = pd.DataFrame(X_scaled, columns=VALID_FEATURES)
    
    # 2. Autoencoder Reconstruction Error
    autoencoder = models["autoencoder"]
    # Ensure consistent type for pytorch
    recon_error = autoencoder.get_reconstruction_error(X_scaled_df.values)
    X_scaled_df['ReconstructionError'] = recon_error
    
    return X_scaled_df

# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle manager."""
    logger.info("Starting Fraud Detection Analytics Server...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Load models
    try:
        logger.info("Loading models...")
        
        with open("models/scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
            
        models["autoencoder"] = Autoencoder(input_dim=29)
        models["autoencoder"].load_state_dict(torch.load("models/autoencoder.pth"))
        models["autoencoder"].eval()
        
        models["xgboost"] = XGBoostModel()
        models["xgboost"].load("models/xgboost.json")
        
        logger.info("Models loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
    except Exception as e:
        logger.error(f"Critical error loading models: {e}")
        logger.warning("Analytics endpoints requiring inference will fail.")

    yield
    
    logger.info("Shutting down server...")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Fraud Detection Analytics API",
    description="Analytics API for fraud detection system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Endpoints: System
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection Analytics API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": all(models.values())}

# ============================================================================
# API Endpoints: Analytics
# ============================================================================

@app.get("/api/stats", response_model=TransactionStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get overall transaction statistics."""
    total_result = await db.execute(select(func.count(TransactionTable.id)))
    total_transactions = total_result.scalar() or 0
    
    fraud_result = await db.execute(
        select(func.count(TransactionTable.id)).where(TransactionTable.Class == "1")
    )
    fraud_count = fraud_result.scalar() or 0
    
    normal_count = total_transactions - fraud_count
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0.0
    
    return TransactionStats(
        total_transactions=total_transactions,
        fraud_count=fraud_count,
        normal_count=normal_count,
        fraud_rate=fraud_rate
    )

@app.get("/api/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    class_filter: Optional[str] = Query(None, regex="^(0|1)$"),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated transactions with optional filtering."""
    query = select(TransactionTable).order_by(desc(TransactionTable.id))
    
    if class_filter:
        query = query.where(TransactionTable.Class == class_filter)
    
    result = await db.execute(query.offset(skip).limit(limit))
    return result.scalars().all()

@app.get("/api/recent", response_model=List[TransactionResponse])
async def get_recent_transactions(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get most recent transactions."""
    query = select(TransactionTable).order_by(desc(TransactionTable.id)).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

@app.get("/api/anomalies", response_model=List[TransactionResponse])
async def get_anomalies(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get confirmed fraud transactions only."""
    query = select(TransactionTable).where(
        TransactionTable.Class == "1"
    ).order_by(desc(TransactionTable.id)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

# ============================================================================
# API Endpoints: Visualizations
# ============================================================================

@app.get("/api/distributions/amount", response_model=DistributionData)
async def get_amount_distribution(
    sample_size: int = Query(1000, ge=100, le=10000),
    db: AsyncSession = Depends(get_db)
):
    """Get amount distribution for fraud vs normal transactions."""
    def _get_query(class_label):
        return select(TransactionTable.Amount).where(
            TransactionTable.Class == class_label
        ).limit(sample_size)

    fraud_res = await db.execute(_get_query("1"))
    normal_res = await db.execute(_get_query("0"))
    
    return DistributionData(
        fraud_values=[row[0] for row in fraud_res.all()],
        normal_values=[row[0] for row in normal_res.all()]
    )

@app.get("/api/distributions/feature/{feature_name}", response_model=DistributionData)
async def get_feature_distribution(
    feature_name: str,
    sample_size: int = Query(1000, ge=100, le=10000),
    db: AsyncSession = Depends(get_db)
):
    """Get distribution for a specific V-feature."""
    if feature_name not in VALID_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid feature. Must be one of {VALID_FEATURES}")
    
    feature_col = getattr(TransactionTable, feature_name)
    
    def _get_query(class_label):
        return select(feature_col).where(
            TransactionTable.Class == class_label
        ).limit(sample_size)
    
    fraud_res = await db.execute(_get_query("1"))
    normal_res = await db.execute(_get_query("0"))
    
    return DistributionData(
        fraud_values=[row[0] for row in fraud_res.all()],
        normal_values=[row[0] for row in normal_res.all()]
    )

@app.get("/api/scatter", response_model=ScatterData)
async def get_scatter_data(
    x_feature: str,
    y_feature: str,
    limit: int = Query(1000, ge=100, le=5000),
    db: AsyncSession = Depends(get_db)
):
    """Get scatter plot data comparing two features."""
    if x_feature not in VALID_FEATURES or y_feature not in VALID_FEATURES:
        raise HTTPException(status_code=400, detail="Invalid feature selected")
        
    x_col = getattr(TransactionTable, x_feature)
    y_col = getattr(TransactionTable, y_feature)
    
    # Balanced sampling: 50% fraud (if available), 50% normal
    fraud_limit = limit // 2
    normal_limit = limit - fraud_limit

    async def fetch_points(is_fraud_str, limit_count):
        q = select(x_col, y_col).where(TransactionTable.Class == is_fraud_str).limit(limit_count)
        res = await db.execute(q)
        return [
            ScatterPoint(x=row[0], y=row[1], is_fraud=(is_fraud_str == "1"))
            for row in res.all()
        ]

    fraud_points = await fetch_points("1", fraud_limit)
    # Adjust normal limit if we found fewer fraud cases than requested
    normal_limit += (fraud_limit - len(fraud_points))
    normal_points = await fetch_points("0", normal_limit)
    
    return ScatterData(points=fraud_points + normal_points)

# ============================================================================
# API Endpoints: Machine Learning
# ============================================================================

@app.get("/api/model/performance", response_model=ModelPerformance)
async def get_model_performance(
    sample_size: int = Query(2000, ge=100, le=5000),
    db: AsyncSession = Depends(get_db)
):
    """Calculate model performance metrics on a sample of database data."""
    if not all(models.values()):
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # Fetch balanced sample
    fraud_limit = sample_size // 2
    normal_limit = sample_size - fraud_limit
    
    fraud_res = await db.execute(select(TransactionTable).where(TransactionTable.Class == "1").limit(fraud_limit))
    normal_res = await db.execute(select(TransactionTable).where(TransactionTable.Class == "0").limit(normal_limit))
    
    fraud_txs = fraud_res.scalars().all()
    normal_txs = normal_res.scalars().all()
    all_txs = fraud_txs + normal_txs
    
    if not all_txs:
        raise HTTPException(status_code=404, detail="No data available for evaluation")
        
    # Convert SQLAlchemy objects to DataFrame efficiently
    data_dicts = []
    y_true = []
    for tx in all_txs:
        # Extract only valid features to avoid sqlalchemy state or ID columns
        row = {col: getattr(tx, col) for col in VALID_FEATURES}
        data_dicts.append(row)
        y_true.append(int(tx.Class))
        
    df = pd.DataFrame(data_dicts)
    
    # Run Pipeline
    X_prepared = _run_inference_pipeline(df)
    
    # Prediction
    xgb_model = models["xgboost"]
    y_pred = xgb_model.predict(X_prepared)
    y_prob = xgb_model.predict_proba(X_prepared)
    
    # Calculate Metrics
    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0))
    )
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Feature Importance (Top 10)
    importance_dict = xgb_model.model.get_booster().get_score(importance_type='gain')
    total_imp = sum(importance_dict.values()) or 1.0
    
    feat_imp = [
        FeatureImportance(feature=k, importance=v/total_imp)
        for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return ModelPerformance(
        metrics=metrics,
        confusion_matrix=ConfusionMatrix(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        roc_curve=RocCurve(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=float(roc_auc)),
        feature_importance=feat_imp
    )

@app.post("/api/predict", response_model=TransactionResponse)
async def predict_transaction(
    transaction: TransactionCreate,
    db: AsyncSession = Depends(get_db)
):
    """Predict fraud for a single incoming transaction and save to DB."""
    if not all(models.values()):
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # Convert input to DataFrame
    input_data = transaction.model_dump()
    df = pd.DataFrame([input_data])
    
    # Run Pipeline
    X_prepared = _run_inference_pipeline(df)
    
    # Predict
    y_pred = models["xgboost"].predict(X_prepared)[0]
    
    # Save to Database
    db_transaction = TransactionTable(
        **input_data,
        Class=str(int(y_pred))
    )
    
    db.add(db_transaction)
    await db.commit()
    await db.refresh(db_transaction)
    
    return db_transaction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)