# 🏡 House Price Predictor

> Predict the sale price of a residential house using a Linear Regression model trained on 1,500 Ames-style housing records.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/s4tyam2k04/house_price_predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.34.2-orange?style=for-the-badge)](https://gradio.app/)

---

## 🚀 Try It Live

**[→ Open the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/s4tyam2k04/house_price_predictor)**

Enter 20 property features and get an estimated sale price with a **90% confidence range** — instantly, in your browser.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Features Used](#features-used)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Retrain with Your Own Data](#retrain-with-your-own-data)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Confidence Range](#confidence-range)
- [License](#license)

---

## Overview

This project builds an end-to-end house price prediction system using **Ordinary Least Squares Linear Regression**. The model is trained on an Ames-style housing dataset and served via a clean Gradio web interface.

**Input:** 20 property features (size, quality, age, rooms, garage, neighbourhood)  
**Output:** Estimated sale price + 90% confidence interval

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression (OLS) |
| Training rows | 1,200 |
| Test rows | 300 |
| Features (after encoding) | 42 |
| Test R² | **0.9634** |
| Test MAE | **$12,646** |
| Test MAPE | **5.79%** |

The model explains ~96% of variance in sale prices on held-out data, with a median absolute error under $13K.

---

## Features Used

**Numeric (raw)**
| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality (1–10) |
| `OverallCond` | Overall condition rating |
| `GrLivArea` | Above-grade living area (sq ft) |
| `TotalBsmtSF` | Total basement area (sq ft) |
| `GarageCars` | Garage capacity (cars) |
| `GarageArea` | Garage area (sq ft) |
| `BedroomAbvGr` | Bedrooms above ground |
| `FullBath` | Full bathrooms above ground |
| `HalfBath` | Half bathrooms above ground |
| `TotRmsAbvGrd` | Total rooms above ground |
| `Fireplaces` | Number of fireplaces |
| `LotArea` | Lot size (sq ft) |
| `YearBuilt` | Year construction was completed |
| `YearRemodAdd` | Year of last remodel |
| `YrSold` | Year sold |

**Engineered (derived at prediction time)**
| Feature | Formula |
|---|---|
| `HouseAge` | `YrSold − YearBuilt` |
| `YearsSinceRemodel` | `YrSold − YearRemodAdd` |
| `TotalSF` | `GrLivArea + TotalBsmtSF` |
| `Bath_Total` | `FullBath + 0.5 × HalfBath` |
| `QualArea` | `OverallQual × GrLivArea` |

**Categorical (one-hot encoded, `drop_first=True`)**  
`Neighborhood` · `HouseStyle` · `GarageType` · `CentralAir` · `SaleCondition`

**Ordinal encoded**  
`BsmtQual` → None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5

---

## Project Structure

```
house-price-predictor/
├── app.py                   # Main Gradio application
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── models/
│   ├── best_model.pkl       # Trained LinearRegression (~1.6 KB)
│   ├── scaler.pkl           # Fitted StandardScaler
│   ├── feature_names.pkl    # Exact 42-column order
│   ├── scale_cols.pkl       # Columns requiring scaling
│   └── model_metadata.json  # Audit trail (algorithm, date, metrics)
└── src/
    ├── data_loader.py
    ├── data_preprocessing.py
    ├── train_model.py
    └── save_model.py
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
# Open http://localhost:7860
```

---

## Retrain with Your Own Data

1. Replace `models/raw_data.csv` with your dataset.
2. Run the training pipeline in order:

```bash
python src/data_loader.py
python src/data_preprocessing.py
python src/train_model.py
python src/save_model.py
```

3. The updated `models/*.pkl` files are picked up automatically on the next launch — no changes to `app.py` required.

---

## Preprocessing Pipeline

Training and serving use an **identical** pipeline to prevent data leakage:

1. **Feature engineering** — derive 5 computed columns
2. **Ordinal encode** `BsmtQual`
3. **One-hot encode** 5 nominal categoricals (`drop_first=True`)
4. **Column alignment** — missing OHE columns set to 0; column order enforced to match training
5. **StandardScaler** — fitted on training data only, applied at inference time

---

## Confidence Range

The displayed 90% confidence range uses the model's test-set MAPE (5.8%) scaled by the 90% normal z-score (1.645) as a pragmatic approximation:

```
Lower = prediction × (1 − 1.645 × 0.058)
Upper = prediction × (1 + 1.645 × 0.058)
```

> For production use, replace this with bootstrapped confidence intervals or conformal prediction.

---

## License

MIT — free to use, modify, and distribute with attribution. See [LICENSE](LICENSE) for details.
