# Power Outage Forecasting and Backup Generator Pre-positioning

CMU **95828 Machine Learning for Problem Solving (S26)** — Final Project.

**Team:** Evie Wei, Jane Wu, Mike Zheng

This project addresses two integrated tasks for the state of Michigan:
1. **Short-term outage prediction** — hourly county-level forecasts for 24h and 48h horizons.
2. **Backup generator pre-positioning** — recommending five Michigan counties for placing five 1,000-household generators based on the forecasts.

---

## File Structure

```
code/
├── README.md                       # this file
├── remove_correlated_features.py   # feature preprocessing (Spearman |r|>0.9 drop)
├── EDA+ARIMA+GRU.ipynb             # EDA, ARIMA baseline, GRU model, submission CSV
├── lstm_gridsearch.py              # LSTM hyperparameter grid search
├── transformer.ipynb               # per-county Transformer model
├── OR.ipynb                        # generator placement (K-Means + Greedy)
└── power_stations.png              # generator placement visualization
```

Deliverables (in parent directory):
- `gru_24.csv`, `gru_48.csv` — 24h / 48h prediction submissions (GRU, our final model)
- `stations_recommendation.txt` — recommended five FIPS codes for generator placement
- `ML_Final_Report.pdf` — final write-up

---

## Pipeline

### 1. Data Preprocessing — `remove_correlated_features.py`
Loads `county_sheets_with_log_out.xlsx` (per-county sheets with hourly outage + 109 weather features) and drops one of every Spearman-correlated pair with |r| > 0.9. Produces `cleaned_output.xlsx` used by all downstream models.

```bash
python remove_correlated_features.py
```

### 2. EDA + Baseline + GRU — `EDA+ARIMA+GRU.ipynb`
- Loads `train.nc` (xarray dataset).
- Exploratory analysis: outage distribution, hourly/weekly patterns, county-level totals, weather–outage Spearman correlations.
- ARIMA baseline (per county).
- GRU model (per county, 24h window → 48h forecast) — **selected as the final model**.
- Exports 24h / 48h prediction CSVs in the required submission format.

### 3. LSTM (alternate model) — `lstm_gridsearch.py`
Standalone TensorFlow/Keras script that grid-searches LSTM hyperparameters per county and writes metrics to `LSTM_metrics.xlsx`. Not used in the final submission but reported for comparison.

### 4. Transformer (alternate model) — `transformer.ipynb`
Per-county temporal Transformer (W=24, d_model=64, nhead=4, 2 encoder layers). Trains 83 separate models, evaluates MSE/MAE/R² in log space, and produces `transformer_pred_24h.csv` / `transformer_pred_48h.csv`. Used for comparison only.

### 5. Generator Placement — `OR.ipynb`
Two algorithms compared, both with **5 stations** and **per-station capacity 1,000 households**:
- **Capacitated Weighted K-Means** — geographic clustering with capacity constraints.
- **Greedy Capacitated Assignment** — seeds stations at the five highest-demand counties and greedily assigns the rest.

Cell 4 outputs **two separate figures** (`power_stations_kmeans.png`, `power_stations_greedy.png`).

**Final recommendation (Greedy):** `[26125, 26099, 26081, 26163, 26161]`
(Oakland, Macomb, Kent, Wayne, Washtenaw — see `decision_submission.txt`.)

---

## Model Comparison

| Model       | MSE     | MAE  | R²    |
|-------------|---------|------|-------|
| ARIMA       | 156004  | 43.06| 0.22  |
| **GRU**     | **0.67**| **0.43** | **0.30** |
| LSTM        | 0.70    | 0.45 | 0.34  |
| Transformer | 1.61    | 0.68 | -0.61 |

GRU has the best overall ranking and is used to generate the final submission CSVs.

---

## Dependencies

```
python >= 3.10
numpy, pandas, scipy, scikit-learn
xarray, netCDF4
matplotlib, seaborn
torch                  # transformer.ipynb
tensorflow             # lstm_gridsearch.py, EDA+ARIMA+GRU.ipynb (GRU)
statsmodels            # ARIMA baseline
openpyxl               # reading .xlsx
```

---

## How to Reproduce

1. Place `train.nc`, `test_24h_demo.nc`, `test_48h_demo.nc`, and `county_sheets_with_log_out.xlsx` in the working directory.
2. Run `remove_correlated_features.py` to create `cleaned_output.xlsx`.
3. Run `EDA+ARIMA+GRU.ipynb` to generate `gru_24.csv` and `gru_48.csv`.
4. (Optional) Run `transformer.ipynb` and `lstm_gridsearch.py` to reproduce alternate models.
5. From the GRU predictions, average per-county outages are exported to `input_OR.xlsx`. Run `OR.ipynb` to compute generator placement and save `decision_submission.txt`.

---

## Submission Files

| File | Format | Rows |
|------|--------|------|
| `gru_24.csv` | timestamp, location, pred | 1,992 (83 × 24) |
| `gru_48.csv` | timestamp, location, pred | 3,985 (83 × 48) |
| `decision_submission.txt` | `[fips, fips, fips, fips, fips]` | 1 line, 5 FIPS |
