import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path: str):
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    
    data = df.copy()
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    data = pd.DataFrame(data_scaled, columns=data.columns)
    
    return data, scaler

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
