import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

class DashboardExporter:
    """
    Γέφυρα σύνδεσης μεταξύ Python Models και React Dashboard.
    Υπολογίζει αυτόματα metrics και εξάγει JSON στο σωστό format.
    """
    
    def __init__(self, export_path='./dashboard/public/results.json'):
        """
        :param export_path: Το path όπου θα σωθεί το json. 
                            Αν το react app είναι στο φάκελο 'dashboard', 
                            το βάζουμε στο 'dashboard/public/results.json' 
                            για να το βλέπει αυτόματα.
        """
        self.export_path = export_path

    def calculate_metrics(self, y_true, y_pred):
        """Υπολογίζει MAE, RMSE, MAPE"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero for MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape": round(float(mape), 2)
        }

    def save_results(self, 
                     target_variable, 
                     technique, 
                     timestamps, 
                     y_true, 
                     predictions_dict):
        """
        Κύρια συνάρτηση εξαγωγής.
        
        :param target_variable: "Price" ή "Load"
        :param technique: Το όνομα της τεχνικής (π.χ. "Recursive Closed Loop")
        :param timestamps: Λίστα ή Series με τα dates/hours
        :param y_true: Οι πραγματικές τιμές
        :param predictions_dict: Dictionary μορφής {'LGBM': pred_array, 'Baseline': pred_array}
        """
        
        # 1. Prepare Metrics
        metrics_list = []
        
        # Baseline keywords to identify baselines automatically
        baseline_keywords = ['naive', 'seasonal', 'moving', 'exponential']
        
        for model_name, y_pred in predictions_dict.items():
            mets = self.calculate_metrics(y_true, y_pred)
            is_baseline = any(bk in model_name.lower() for bk in baseline_keywords)
            
            metrics_list.append({
                "modelName": model_name,
                "mae": mets["mae"],
                "rmse": mets["rmse"],
                "mape": mets["mape"],
                "isBaseline": is_baseline
            })
            
        # Sort metrics by MAE (best on top)
        metrics_list.sort(key=lambda x: x["mae"])

        # 2. Prepare Time Series Data Points
        data_points = []
        
        # Convert to standard lists for JSON serialization
        timestamps_list = pd.to_datetime(timestamps).astype(str).tolist()
        y_true_list = list(y_true)
        
        # Ensure predictions are lists
        preds_lists = {k: list(v) for k, v in predictions_dict.items()}
        
        for i in range(len(timestamps_list)):
            point = {
                "timestamp": timestamps_list[i],
                "actual": float(y_true_list[i])
            }
            
            for model_name, preds in preds_lists.items():
                point[model_name] = float(preds[i])
                
            data_points.append(point)

        # 3. Construct Final Object
        final_json = {
            "target": target_variable,     # "Price" or "Load"
            "technique": technique,        # e.g. "MIMO (Direct)"
            "data": data_points,
            "metrics": metrics_list
        }

        # 4. Save to File
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
        
        with open(self.export_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2)
            
        print(f"✅ Dashboard updated successfully! Results saved to: {self.export_path}")
        print("➡️  Open http://localhost:3000 to view results.")

# --- EXAMPLE USAGE (Αντιγραφή στο main script σου) ---
if __name__ == "__main__":
    # Mock data for demonstration
    dates = pd.date_range(start="2023-01-01", periods=24, freq="H")
    y_true = np.random.normal(150, 20, 24)
    
    preds = {
        "LGBM": y_true + np.random.normal(0, 5, 24),
        "XGBoost": y_true + np.random.normal(0, 6, 24),
        "Naive Baseline": y_true + np.random.normal(0, 15, 24)
    }
    
    exporter = DashboardExporter(export_path='./dashboard/public/results.json')
    
    exporter.save_results(
        target_variable="Price",
        technique="MIMO (Direct)",
        timestamps=dates,
        y_true=y_true,
        predictions_dict=preds
    )
