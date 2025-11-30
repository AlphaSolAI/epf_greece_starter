import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import sys
import io
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Ρυθμίσεις ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# --- Paths ---
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# --- Χρώματα ---
COLORS = {
    'Actual': 'black',
    'SVR': '#E63946',      # Κόκκινο
    'XGBoost': '#1D3557',  # Σκούρο Μπλε
    'MLP': '#457B9D',      # Ανοιχτό Μπλε
    'LGBM': '#2A9D8F',     # Τιρκουάζ
    'RF': '#F4A261',       # Πορτοκαλί
    'Transformer': '#9C27B0', # Μωβ
    'SARIMA': '#2CA02C',   # Πράσινο
    'Naive': '#7F7F7F'     # Γκρι
}

# --- Transformer Class (FIXED: dim_feedforward=256) ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))
        
        # ΕΔΩ ΗΤΑΝ ΤΟ ΛΑΘΟΣ -> dim_feedforward=256
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

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return mae, rmse, mape

def predict_transformer(model_path, scaler_path, df_full, test_indices, seq_length=30):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scalers = joblib.load(scaler_path)
        # X χωρίς το target y
        X_raw = df_full.drop(columns=['y']).values
        X_scaled = scalers['X'].transform(X_raw)
        
        model = TimeSeriesTransformer(input_dim=X_scaled.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        preds = []
        for i in range(len(test_indices)):
            idx = np.where(df_full.index == test_indices[i])[0][0]
            if idx < seq_length: 
                preds.append(np.nan)
                continue
            
            seq = torch.tensor(X_scaled[idx-seq_length : idx], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                val = model(seq).item()
                preds.append(scalers['y'].inverse_transform([[val]])[0][0])
        return np.array(preds)
    except Exception as e:
        print(f"❌ Transformer Prediction Error: {e}")
        return None

def main():
    print("📊 Final Evaluation (Corrected Model)...")
    
    if not DATA_PATH.exists():
        print("❌ Λείπει το daily.parquet")
        return

    df = pd.read_parquet(DATA_PATH)
    TEST_DAYS = 60
    train = df.iloc[:-TEST_DAYS]
    test = df.iloc[-TEST_DAYS:]
    y_test = test['y'].values
    
    preds = {'Actual': y_test}
    
    # --- 1. BASELINES ---
    print("   -> Calculating Baselines...")
    preds['Naive'] = df['y'].shift(1).iloc[-TEST_DAYS:].values
    
    try:
        # SARIMA Rolling
        history = list(train['y'])
        sarima_model = SARIMAX(history, order=(3,0,2), seasonal_order=(1,1,1,7), 
                               enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima_model.fit(disp=False)
        sarima_res = []
        for t in range(len(test)):
            sarima_res.append(sarima_fit.forecast()[0])
            sarima_fit = sarima_fit.append([test.iloc[t]['y']], refit=False)
        preds['SARIMA'] = np.array(sarima_res)
    except: pass

    # --- 2. ML MODELS ---
    print("   -> Loading ML Models...")
    ml_map = {'XGBoost':'xgb_day.pkl', 'SVR':'svr_day.pkl', 'MLP':'mlp_day.pkl', 'RF':'rf_day.pkl', 'LGBM':'lgbm_day.pkl'}
    for name, f in ml_map.items():
        path = MODELS_DIR / f
        if path.exists():
            try: preds[name] = joblib.load(path).predict(test.drop(columns=['y']))
            except: pass
    
    # --- 3. TRANSFORMER ---
    print("   -> Predicting Transformer...")
    if (MODELS_DIR/"transformer.pth").exists():
        tr_res = predict_transformer(MODELS_DIR/"transformer.pth", MODELS_DIR/"scaler_transformer.pkl", df, test.index)
        if tr_res is not None:
            preds['Transformer'] = tr_res

    # --- 4. METRICS ---
    metrics = []
    for name, val in preds.items():
        if name == 'Actual': continue
        mask = ~np.isnan(val)
        if np.sum(mask) > 0:
            mae, rmse, mape = calculate_metrics(y_test[mask], val[mask])
            metrics.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'sMAPE': mape})
            
    res_df = pd.DataFrame(metrics).sort_values('sMAPE').reset_index(drop=True)
    print("\n" + "="*60)
    print(f"🏆 WINNER: {res_df.iloc[0]['Model']} ({res_df.iloc[0]['sMAPE']:.2f}%)")
    print("="*60)
    print(res_df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    res_df.to_csv(REPORT_DIR / "final_metrics.csv", index=False)

    # --- 5. PLOTTING ---
    print("\n🎨 Generating Split Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: ML
    ax1.plot(test.index, y_test, label='Actual', color='black', linewidth=2.5, alpha=0.8)
    for m in ['SVR', 'XGBoost', 'MLP', 'LGBM', 'RF', 'Transformer']:
        if m in preds:
            ls = '--' if m == res_df.iloc[0]['Model'] else ':'
            lw = 2.5 if m == res_df.iloc[0]['Model'] else 2
            ax1.plot(test.index, preds[m], label=m, color=COLORS.get(m, 'blue'), linestyle=ls, linewidth=lw)
    ax1.set_title("Machine Learning & Deep Learning Models", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Baselines
    ax2.plot(test.index, y_test, label='Actual', color='black', linewidth=2.5, alpha=0.8)
    for m in ['Naive', 'SARIMA']:
        if m in preds:
            ax2.plot(test.index, preds[m], label=m, color=COLORS.get(m, 'gray'), linestyle='--', linewidth=2)
    ax2.set_title("Baseline Models", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "2_Test_Set_Split_View.png", dpi=300)
    print(f"✅ Done. Saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()