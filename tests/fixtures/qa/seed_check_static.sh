#!/usr/bin/env bash
# FIXTURE (acceptance #1): αναπαράσταση του Block D λάθους — ΜΗΝ το τρέξεις.
# Δηλωμένος σκοπός: seed-confirm του weekly headline candidate (Q1=17.035/Summer=13.812)
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --features "default,dense" --seed 7 --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --out_json runs/qa_accept/seed7.json
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --features "default,dense" --seed 123 --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --out_json runs/qa_accept/seed123.json
