from sqlalchemy import Column, Integer, Float, String
from pydantic import BaseModel
from typing import Optional
from src.db import Base

# --- SQLAlchemy Models ---

class TransactionTable(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Features
    V1 = Column(Float)
    V2 = Column(Float)
    V3 = Column(Float)
    V4 = Column(Float)
    V5 = Column(Float)
    V6 = Column(Float)
    V7 = Column(Float)
    V8 = Column(Float)
    V9 = Column(Float)
    V10 = Column(Float)
    V11 = Column(Float)
    V12 = Column(Float)
    V13 = Column(Float)
    V14 = Column(Float)
    V15 = Column(Float)
    V16 = Column(Float)
    V17 = Column(Float)
    V18 = Column(Float)
    V19 = Column(Float)
    V20 = Column(Float)
    V21 = Column(Float)
    V22 = Column(Float)
    V23 = Column(Float)
    V24 = Column(Float)
    V25 = Column(Float)
    V26 = Column(Float)
    V27 = Column(Float)
    V28 = Column(Float)
    Amount = Column(Float)

    # Prediction
    Class = Column(String)

# --- Pydantic Models ---

class TransactionBase(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    Class: Optional[str] = None

    class Config:
        from_attributes = True

# --- Analytics Models ---

class TransactionStats(BaseModel):
    """Overall transaction statistics"""
    total_transactions: int
    fraud_count: int
    normal_count: int
    fraud_rate: float  # Percentage

class DistributionData(BaseModel):
    """Distribution data for amount or features"""
    fraud_values: list[float]
    normal_values: list[float]

class ScatterPoint(BaseModel):
    x: float
    y: float
    is_fraud: bool

class ScatterData(BaseModel):
    points: list[ScatterPoint]

# --- Model Analytics Models ---

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int

class RocCurve(BaseModel):
    fpr: list[float]
    tpr: list[float]
    auc: float

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class ModelPerformance(BaseModel):
    metrics: ModelMetrics
    confusion_matrix: ConfusionMatrix
    roc_curve: RocCurve
    feature_importance: list[FeatureImportance]

