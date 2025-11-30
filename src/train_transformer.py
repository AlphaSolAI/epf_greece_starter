import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# --- PARAMS ---
SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS_PRE = 50   # Εποχές εκπαίδευσης στην Ιταλία
EPOCHS_FINE = 150 # Εποχές εκπαίδευσης στην Ελλάδα
LR_PRE = 0.0005
LR_FINE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
GREECE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
ITALY_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "italy.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "transformer.pth"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler_transformer.pkl"

# --- MODEL ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, src):
        x = self.input_linear(src)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

class EPFDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length
    def __len__(self): return len(self.X) - self.seq_length
    def __getitem__(self, i): return self.X[i:i+self.seq_length], self.y[i+self.seq_length]

def add_features(df):
    """Δημιουργεί lags και calendar features"""
    df = df.copy()
    if 'y' not in df.columns: return df
    
    # Calendar
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Lags (1, 2, 7, 14)
    for lag in [1, 2, 7, 14]:
        df[f'y_lag{lag}'] = df['y'].shift(lag)
            
    # Rolling (7, 30)
    for win in [7, 30]:
        df[f'y_rollmean_{win}'] = df['y'].rolling(win).mean()
        df[f'y_rollstd_{win}'] = df['y'].rolling(win).std()
        
    return df.dropna()

def train_phase(model, loader, epochs, lr, name):
    criterion = nn.HuberLoss() # Πιο ανθεκτικό στα spikes
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print(f"   Starting {name} ({epochs} epochs)...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"     [{name}] Ep {epoch+1}/{epochs} Loss: {epoch_loss/len(loader):.5f}")

def main():
    print(f"🚀 Transformer Transfer Learning on {DEVICE}...")
    
    # 1. Load Greece Data (Target)
    if not GREECE_PATH.exists(): return
    df_gr = pd.read_parquet(GREECE_PATH)
    # Βεβαιωνόμαστε ότι έχει τα features
    df_gr = add_features(df_gr)
    
    features = [c for c in df_gr.columns if c != 'y']
    print(f"   Features: {len(features)}")
    
    # Scalers (Fit on Greece)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    test_days = 60
    X_gr_vals = df_gr[features].values
    y_gr_vals = df_gr['y'].values.reshape(-1, 1)
    
    # Fit scalers on Train set ONLY
    scaler_X.fit(X_gr_vals[:-test_days])
    scaler_y.fit(y_gr_vals[:-test_days])
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)
    
    # Init Model
    model = TimeSeriesTransformer(input_dim=len(features)).to(DEVICE)

    # --- PHASE 1: PRE-TRAINING (ITALY) ---
    if ITALY_PATH.exists():
        print("\n🇮🇹 Φάση 1: Pre-training με Ιταλία...")
        df_it = pd.read_parquet(ITALY_PATH)
        # Δημιουργία ίδιων features για την Ιταλία
        df_it = add_features(df_it)
        
        # Ευθυγράμμιση στηλών (αν λείπει κάποια την γεμίζουμε με 0, π.χ. holidays αν είχαμε)
        for c in features:
            if c not in df_it.columns: df_it[c] = 0
        
        X_it = df_it[features].values
        y_it = df_it['y'].values.reshape(-1, 1)
        
        # Scale Italy data using Greece scalers
        X_it_sc = scaler_X.transform(X_it)
        y_it_sc = scaler_y.transform(y_it)
        
        ds_it = EPFDataset(X_it_sc, y_it_sc, SEQ_LENGTH)
        ld_it = DataLoader(ds_it, batch_size=BATCH_SIZE, shuffle=True)
        
        train_phase(model, ld_it, EPOCHS_PRE, LR_PRE, "ITALY Pre-train")
    else:
        print("⚠️ Italy data not found. Skipping pre-training.")

    # --- PHASE 2: FINE-TUNING (GREECE) ---
    print("\n🇬🇷 Φάση 2: Fine-tuning με Ελλάδα...")
    
    # Prepare Greece Train Data
    X_gr_train = X_gr_vals[:-test_days]
    y_gr_train = y_gr_vals[:-test_days]
    
    X_gr_sc = scaler_X.transform(X_gr_train)
    y_gr_sc = scaler_y.transform(y_gr_train)
    
    ds_gr = EPFDataset(X_gr_sc, y_gr_sc, SEQ_LENGTH)
    ld_gr = DataLoader(ds_gr, batch_size=BATCH_SIZE, shuffle=True)
    
    train_phase(model, ld_gr, EPOCHS_FINE, LR_FINE, "GREECE Fine-tune")
    
    # Save
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Transformer Saved.")

if __name__ == "__main__":
    main()