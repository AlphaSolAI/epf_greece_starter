import json, sys
sys.stdout.reconfigure(encoding='utf-8')
base = 'C:/Users/aggel/OneDrive/Υπολογιστής/ALPHA/ECE/ΔΙΠΛΩΜΑΤΙΚΗ/epf_greece_starter'
f = f'{base}/dashboard_data_hourly_price_openloop_h24_monthly_q4.json'
data = json.loads(open(f, encoding='utf-8').read())
print('Top-level keys:', list(data.keys()))
print('Series keys:', list(data.get('series', {}).keys()))
print('Total metrics entries:', len(data.get('metrics', [])))
