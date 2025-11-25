import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_data, split_data
from models.autoencoder import Autoencoder
from models.xgboost_model import XGBoostModel

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://anomaly_detection_db_39ci_user:92WZAAbtyaw2Ni09cYVfdNq6aYD2mxVL@dpg-d4idghemcj7s739u7j1g-a.singapore-postgres.render.com/anomaly_detection_db_39ci")

def load_data_from_db():
    """Load transaction data from PostgreSQL database."""
    print("Loading data from database...")
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, engine)
    
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int)
    
    print(f"Loaded {len(df)} transactions (Fraud: {df['Class'].sum()}, Normal: {(df['Class']==0).sum()})")
    return df

def train_autoencoder(X_train, y_train, model_save_dir, input_dim):
    """
    Train Autoencoder on normal transactions only.
    
    Returns:
        Trained Autoencoder model
    """
    print("\n=== Training Autoencoder ===")
    X_train_normal = X_train[y_train == 0]
    print(f"Training on {len(X_train_normal)} normal transactions")
    
    autoencoder = Autoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    X_train_normal_tensor = torch.FloatTensor(X_train_normal.values)
    train_loader = torch.utils.data.DataLoader(X_train_normal_tensor, batch_size=64, shuffle=True)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = criterion(output, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    torch.save(autoencoder.state_dict(), os.path.join(model_save_dir, "autoencoder.pth"))
    print("Autoencoder saved.")
    return autoencoder

def train_xgboost_ensemble(X_train, X_test, y_train, y_test, autoencoder, model_save_dir):
    """
    Train XGBoost with reconstruction error as ensemble feature.
    
    Returns:
        Trained XGBoost model
    """
    print("\n=== Training XGBoost Ensemble ===")
    
    # Calculate class imbalance weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
    print(f"Class distribution - Normal: {num_neg}, Fraud: {num_pos}")
    print(f"Scale weight: {scale_pos_weight:.2f}")
    
    # Generate reconstruction errors
    print("Generating reconstruction errors...")
    recon_error_train = autoencoder.get_reconstruction_error(X_train.values)
    recon_error_test = autoencoder.get_reconstruction_error(X_test.values)
    
    # Augment features
    X_train_aug = X_train.copy()
    X_train_aug['ReconstructionError'] = recon_error_train
    
    X_test_aug = X_test.copy()
    X_test_aug['ReconstructionError'] = recon_error_test
    
    print(f"Feature count: {X_train_aug.shape[1]} (29 original + 1 reconstruction error)")
    
    # Train XGBoost
    xgb_model = XGBoostModel(scale_pos_weight=scale_pos_weight)
    xgb_model.train(X_train_aug, y_train)
    xgb_model.save(os.path.join(model_save_dir, "xgboost.json"))
    print("XGBoost saved.")
    
    # Evaluate
    print("\n=== Model Evaluation ===")
    y_pred = xgb_model.predict(X_test_aug)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return xgb_model

def train_pipeline(model_save_dir: str = "models/"):
    """Main training pipeline."""
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load data
    df = load_data_from_db()
    if df.empty:
        print("ERROR: Database is empty. Run 'python src/ingest_csv.py' first.")
        return
    
    # Prepare features
    drop_cols = [c for c in ['Class', 'id'] if c in df.columns]
    X = df.drop(drop_cols, axis=1)
    y = df['Class']
    
    # Preprocess and save scaler
    print("\nPreprocessing data...")
    X_processed, scaler = preprocess_data(X)
    with open(os.path.join(model_save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_processed, y)
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train models
    autoencoder = train_autoencoder(X_train, y_train, model_save_dir, X_train.shape[1])
    xgb_model = train_xgboost_ensemble(X_train, X_test, y_train, y_test, autoencoder, model_save_dir)
    
    print("\n✓ Training complete. Models saved to:", model_save_dir)

if __name__ == "__main__":
    train_pipeline("models/")
