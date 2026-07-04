import json, sys
sys.stdout.reconfigure(encoding='utf-8')
base = 'C:/Users/aggel/OneDrive/Υπολογιστής/ALPHA/ECE/ΔΙΠΛΩΜΑΤΙΚΗ/epf_greece_starter'
f = f'{base}/dashboard_data_hourly_price_openloop_h24_monthly_q4.json'
data = json.loads(open(f, encoding='utf-8').read())
metrics = data.get('metrics', [])
print('=== PRICE OL Q4 (Oct-Dec 2025) H24 ===')
for m in sorted(metrics, key=lambda x: x.get('MAE', 999)):
    print(f"{m.get('Model','?'):50s} MAE={m.get('MAE',0):.3f}  RMSE={m.get('RMSE',0):.3f}  sMAPE={m.get('sMAPE',0):.3f}%")
