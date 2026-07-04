import openpyxl
wb = openpyxl.load_workbook("thesis_output/eval_results_summary.xlsx", read_only=True, data_only=True)
for sh in ["Dec25_Price", "Dec25_Load"]:
    ws = wb[sh]
    print(f"\n==== {sh} ====")
    for row in ws.iter_rows(values_only=True):
        # print rows that look like recursive/OL or header
        cells = [c for c in row if c is not None]
        if not cells: continue
        s = " | ".join(str(c) for c in row if c is not None)
        if any(k in s for k in ["Model","model","Daily","OL","Rec","SS","Dense","Ensemble","MAE","SVR","RMSE","LGBM","XGB","MLP","RF"]):
            print("  ", s)
