---
title: House Price Predictor
emoji: 🏡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
short_description: Predict residential house prices with Linear Regression
---

# 🏡 House Price Predictor

Predict the **sale price of a residential house** using a Linear Regression model
trained on 1,500 Ames-style housing records.

## What it does

Enter 20 property features — size, quality, age, rooms, garage, and neighbourhood —
and the model returns an estimated sale price with a 90% confidence range.

## Model

| Property | Value |
|---|---|
| Algorithm | Linear Regression (OLS) |
| Training rows | 1,200 |
| Test rows | 300 |
| Features | 42 (after encoding) |
| Test R² | 0.9634 |
| Test MAE | $12,646 |
| Test MAPE | 5.79% |

## Features used

**Numeric**  
`OverallQual` · `OverallCond` · `GrLivArea` · `TotalBsmtSF` · `GarageCars` ·
`GarageArea` · `BedroomAbvGr` · `FullBath` · `HalfBath` · `TotRmsAbvGrd` ·
`Fireplaces` · `LotArea` · `YearBuilt` · `YearRemodAdd` · `YrSold`

**Engineered** (derived at prediction time)  
`HouseAge` · `YearsSinceRemodel` · `TotalSF` · `Bath_Total` · `QualArea`

**Categorical** (one-hot encoded)  
`Neighborhood` · `HouseStyle` · `GarageType` · `CentralAir` · `SaleCondition`

**Ordinal encoded**  
`BsmtQual`  (None=0 · Po=1 · Fa=2 · TA=3 · Gd=4 · Ex=5)

## Preprocessing pipeline

Training and serving preprocessing are identical:

1. Feature engineering (5 derived columns)
2. Ordinal encode `BsmtQual`
3. One-hot encode 5 nominal categoricals (`drop_first=True`)
4. Column alignment — missing OHE columns set to 0, exact order enforced
5. `StandardScaler` applied to continuous features (fitted on training data only)

## Project structure

```
├── app.py                  Main Gradio application
├── requirements.txt        Pinned dependencies
├── README.md               This file (also the HF Space card)
└── models/
    ├── best_model.pkl      Trained LinearRegression (1.6 KB)
    ├── scaler.pkl          Fitted StandardScaler
    ├── feature_names.pkl   Exact column order (42 features)
    ├── scale_cols.pkl      Columns requiring scaling
    └── model_metadata.json Audit trail (algorithm, date, metrics)
```

## Run locally

```bash
git clone https://huggingface.co/spaces/<your-username>/house-price-predictor
cd house-price-predictor
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Retrain with your own data

1. Replace `models/raw_data.csv` and run the training pipeline:

```bash
python src/data_loader.py
python src/data_preprocessing.py
python src/train_model.py
python src/save_model.py
```

2. The new `models/*.pkl` files will be picked up automatically on next launch.

## Confidence range

The displayed 90% range uses the model's test-set MAPE (5.8%) scaled by the
90% normal z-score (1.65) as a pragmatic approximation.  
For production use, replace with bootstrapped confidence intervals or
conformal prediction.

## License

MIT — free to use, modify, and distribute with attribution.
