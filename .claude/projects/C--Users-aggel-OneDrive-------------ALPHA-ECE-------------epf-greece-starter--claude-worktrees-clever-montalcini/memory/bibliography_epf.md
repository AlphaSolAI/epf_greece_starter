# Scientific Bibliography — EPF & Multi-Step Forecasting

## 1. MIMO / Multi-Step Strategies

### Primary References

**Ben Taieb, S., Sorjamaa, A., & Bontempi, G. (2010)**
"Multiple-Output Modelling for Multi-Step-Ahead Time Series Forecasting"
- First formal MIMO framework for simultaneous multi-horizon prediction
- https://www.semanticscholar.org/paper/Multiple-Output-Modelling-for-Multi-Step-Ahead-Time-Taieb-Sorjamaa/4c36dc84c4c1e7a40cd1e7347cddc4a733e7aeff

**Ben Taieb, S. & Bontempi, G. (2011)**
"Long-term prediction of time series by combining direct and MIMO strategies"
- DIRMO strategy outperforms pure MIMO/direct
- https://www.semanticscholar.org/paper/Long-term-prediction-of-time-series-by-combining-Taieb-Bontempi/dbe548dc62d75bfc7762c19c510c2acf9a8af926

**Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012)**
"A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition"
Expert Systems with Applications, vol. 39, no. 7
- Empirical comparison on 111 time series; DIRMO best
- DOI: https://www.sciencedirect.com/science/article/abs/pii/S0957417412000528

**Ben Taieb, S. (2014)** — PhD Thesis
"Machine Learning Strategies for Time Series Forecasting"
- Comprehensive analysis of recursive, direct, MIMO strategies
- https://souhaib-bentaieb.com/papers/2014_phd.pdf

**Taieb, S. B. & Hyndman, R. J. (2012)**
"Recursive and direct multi-step forecasting: the best of both worlds" (DirRec strategy)
- https://robjhyndman.com/publications/rectify/

## 2. Direct Multi-Step Forecasting

- H different models, each predicting one specific future step h
- Skforecast documentation: https://skforecast.org/0.9.0/user_guides/direct-multi-step-forecasting
- Avoids error propagation but increases variance with limited data

## 3. Open-Loop (Recursive) Forecasting & Error Accumulation

- Error cascades in recursive prediction (exposure bias)
- Zhang, Y. et al. (2024): "Epistemic Error Decomposition for Multi-step Time Series Forecasting"
  arXiv:2511.11461 — bias-variance decomposition in recursive vs direct
- Wang et al. (2025): "Closing the Loop" (F-LLM feedback controller)
  arXiv:2602.12756

## 4. Scheduled Sampling

**Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015)**
"Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
NeurIPS 2015
- Curriculum learning: gradually replace teacher forcing with model predictions
- ε (epsilon) decreases from 1 → 0 over training
- arXiv:1506.03099
- NeurIPS: https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf

**Ranzato, M., Chopra, S., Auli, M., & Zaremba, W. (2016)**
"Sequence Level Training with Recurrent Neural Networks" — MRT approach
ICLR 2016 — arXiv:1511.06732

## 5. EPF Surveys & Benchmarks

**Weron, R. (2014)**
"Electricity price forecasting: A review of the state-of-the-art with a look into the future"
International Journal of Forecasting, vol. 30, no. 4, pp. 1030-1081
- Comprehensive 15-year review; postulates need for objective comparative EPF studies
- https://www.sciencedirect.com/science/article/pii/S0169207014001083

**Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021)**
"Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms,
best practices and an open-access benchmark"
Applied Energy, vol. 293, article 116983
- Compares statistical + DL methods; open-source EPFToolbox
- arXiv:2008.08004 | DOI: ScienceDirect
- GitHub: https://github.com/jeslago/epftoolbox

**Nowotarski, J. & Weron, R. (2016)**
"On the importance of the long-term seasonal component in day-ahead electricity price forecasting"
Energy Economics, vol. 57, pp. 228-235 — SCAR model
- https://www.sciencedirect.com/science/article/pii/S014098831630127X

## 6. LightGBM / XGBoost for Energy Forecasting

- Performance: XGB -44% MAE vs baselines; LGBM -42% MAE
- Greek load forecasting: MDPI Energies 2024: https://www.mdpi.com/1996-1073/18/19/5060
- Hybrid stacking methods: TechRxiv 2024
- Feature importance: hour-of-day, lag-1, lag-24 most critical
  Nature Scientific Reports: https://www.nature.com/articles/s41598-022-22024-3

## 7. Deep Learning Load Forecasting Survey

"Short-Term Electricity-Load Forecasting by Deep Learning: A Comprehensive Survey" (2024)
arXiv:2408.16202 — CNN-LSTM, Transformer-based approaches
- DWT-LSTM: MAPE 0.59-4.2% for hour-ahead to year-ahead
- CNN-BiLSTM for day-ahead forecasting

## 8. Feature Leakage in EPF

- Same-day price information prohibited (only lagged/future prices)
- Day-ahead load forecast (load_fc): legitimate feature (published before DAM)
- Gas/CO2 futures (D): borderline but standard in EPF literature
- https://www.mdpi.com/2076-3407/16/1/200

## 9. Key Insights from Own Results

| Finding | Value |
|---------|-------|
| Day-by-day 24h vs weekly 168h (OL) | -16.7% MAE improvement |
| Scheduled Sampling helps XGB daily | 15.312→15.036 MAE |
| SS does NOT help LGBM daily | 14.494→14.986 (worse) |
| Two-stage load→price MIMO | Under evaluation |
| Best OL price (1-month) | LGBM-Daily-Optuna 14.494 €/MWh |
| Best OL load (1-week) | LGBM-OL-Optuna-SS 62 MW |

## Key Thesis Citations

1. Weron 2014 (EPF survey)
2. Lago et al. 2021 (EPF benchmark)
3. Ben Taieb et al. 2012 (multi-step strategies)
4. Bengio et al. 2015 NeurIPS (Scheduled Sampling)
5. Ben Taieb & Hyndman 2012 (DirRec)
