"""
app.py  —  House Price Prediction · Hugging Face Spaces
--------------------------------------------------------
Gradio Blocks interface that loads a pre-trained Linear Regression
model and predicts house prices from user-supplied features.

Structure
  1. Model bundle loading  (once at startup)
  2. Preprocessing         (mirrors training pipeline exactly)
  3. Prediction function   (called by Gradio)
  4. Gradio Blocks UI      (two-column form + result panel)
  5. Example rows          (pre-filled quick-start inputs)

Run locally:
    pip install -r requirements.txt
    python app.py
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ── 0. Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── 1. Load model bundle (once at startup, kept in module scope) ──────────────
#
# HF Spaces best practice: load heavy artefacts at module-import time, not
# inside the prediction function.  The Space is a long-running process; loading
# on every request would add 20-50 ms and waste RAM by reloading from disk.

def _load_bundle() -> dict:
    required = ["best_model.pkl", "scaler.pkl",
                "feature_names.pkl", "scale_cols.pkl", "model_metadata.json"]
    for f in required:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            raise FileNotFoundError(
                f"Missing model artefact: {f}\n"
                "Ensure the models/ folder is present in the Space repository."
            )
    return {
        "model"   : joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")),
        "scaler"  : joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "features": joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl")),
        "scale_cols": joblib.load(os.path.join(MODELS_DIR, "scale_cols.pkl")),
        "meta"    : json.load(open(os.path.join(MODELS_DIR, "model_metadata.json"))),
    }

BUNDLE = _load_bundle()
META   = BUNDLE["meta"]


# ── 2. Valid input definitions ────────────────────────────────────────────────

NEIGHBORHOODS = ["NridgHt", "CollgCr", "Somerst", "Gilbert", "NWAmes",
                 "OldTown", "Edwards", "Sawyer", "Mitchel", "BrkSide"]
HOUSE_STYLES  = ["1Story", "2Story", "1.5Fin", "SFoyer", "SLvl"]
BSMT_QUAL     = ["Ex", "Gd", "TA", "Fa", "Po", "None"]
GARAGE_TYPES  = ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "None"]
SALE_CONDS    = ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]

BSMT_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


# ── 3. Preprocessing (mirrors data_preprocessing.py exactly) ─────────────────
#
# CRITICAL RULE: every transform applied during training must be applied here
# in exactly the same order.  Any divergence produces silent wrong predictions.
#
# Order:
#   a) Build raw dict → 1-row DataFrame
#   b) Feature engineering   (5 derived columns)
#   c) Ordinal encode BsmtQual
#   d) One-hot encode nominals
#   e) Column alignment      (add missing OHE cols as 0, enforce order)
#   f) Scale continuous cols with the saved scaler

def preprocess(
    overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
    garage_cars, garage_area, bedroom_abvgr, full_bath, half_bath,
    tot_rms_abvgrd, fireplaces, lot_area,
    year_built, year_remod, yr_sold,
    bsmt_qual, neighborhood, house_style, garage_type,
    central_air, sale_condition,
) -> pd.DataFrame:

    raw = {
        "OverallQual"  : int(overall_qual),
        "OverallCond"  : int(overall_cond),
        "GrLivArea"    : int(gr_liv_area),
        "TotalBsmtSF"  : float(total_bsmt_sf),
        "GarageCars"   : int(garage_cars),
        "GarageArea"   : float(garage_area),
        "BedroomAbvGr" : int(bedroom_abvgr),
        "FullBath"     : int(full_bath),
        "HalfBath"     : int(half_bath),
        "TotRmsAbvGrd" : int(tot_rms_abvgrd),
        "Fireplaces"   : int(fireplaces),
        "LotArea"      : int(lot_area),
        "YearBuilt"    : int(year_built),
        "YearRemodAdd" : int(year_remod),
        "YrSold"       : int(yr_sold),
        "BsmtQual"     : bsmt_qual,
        "Neighborhood" : neighborhood,
        "HouseStyle"   : house_style,
        "GarageType"   : garage_type,
        "CentralAir"   : central_air,
        "SaleCondition": sale_condition,
    }

    df = pd.DataFrame([raw])

    # a) Feature engineering
    df["HouseAge"]          = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalSF"]           = df["GrLivArea"] + df["TotalBsmtSF"]
    df["Bath_Total"]        = df["FullBath"]  + 0.5 * df["HalfBath"]
    df["QualArea"]          = df["OverallQual"] * df["GrLivArea"]
    df.drop(columns=["YearBuilt", "YearRemodAdd", "YrSold"], inplace=True)

    # b) Ordinal encode BsmtQual
    df["BsmtQual_Enc"] = df["BsmtQual"].map(BSMT_MAP).fillna(0).astype(int)
    df.drop(columns=["BsmtQual"], inplace=True)

    # c) One-hot encode nominals
    ohe_cols = ["Neighborhood", "HouseStyle", "GarageType", "CentralAir", "SaleCondition"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    # d) Align to training schema — add any missing OHE columns as 0
    for col in BUNDLE["features"]:
        if col not in df.columns:
            df[col] = 0
    df = df[BUNDLE["features"]]   # enforce exact column order

    # e) Scale continuous features
    df[BUNDLE["scale_cols"]] = BUNDLE["scaler"].transform(df[BUNDLE["scale_cols"]])

    return df


# ── 4. Prediction function (called by Gradio on button click) ─────────────────
#
# Gradio passes each widget's current value as a positional argument in the
# order they appear in the `inputs` list.  We validate, preprocess, run the
# model, and return three strings: price, range, and model info.

def predict(
    overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
    garage_cars, garage_area, bedroom_abvgr, full_bath, half_bath,
    tot_rms_abvgrd, fireplaces, lot_area,
    year_built, year_remod, yr_sold,
    bsmt_qual, neighborhood, house_style, garage_type,
    central_air, sale_condition,
):
    # ── Input validation ────────────────────────────────────────────────────
    errors = []
    if year_remod < year_built:
        errors.append("Year Remodelled cannot be earlier than Year Built.")
    if yr_sold < year_built:
        errors.append("Year Sold cannot be earlier than Year Built.")
    if int(tot_rms_abvgrd) < int(bedroom_abvgr):
        errors.append("Total rooms above grade must be ≥ bedrooms above grade.")
    if errors:
        msg = "⚠️  Please fix the following:\n" + "\n".join(f"  • {e}" for e in errors)
        return msg, "", ""

    try:
        X = preprocess(
            overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
            garage_cars, garage_area, bedroom_abvgr, full_bath, half_bath,
            tot_rms_abvgrd, fireplaces, lot_area,
            year_built, year_remod, yr_sold,
            bsmt_qual, neighborhood, house_style, garage_type,
            central_air, sale_condition,
        )
    except Exception as e:
        return f"⚠️  Preprocessing error: {e}", "", ""

    try:
        price = float(BUNDLE["model"].predict(X)[0])
        price = max(price, 10_000)
    except Exception as e:
        return f"⚠️  Model error: {e}", "", ""

    # Confidence interval: ± 1.65 × MAPE × price  (90% interval proxy)
    mape    = META["metrics"]["test_mape_pct"] / 100
    margin  = price * mape * 1.65
    lo, hi  = max(price - margin, 10_000), price + margin

    price_str = f"${price:,.0f}"
    range_str = f"${lo:,.0f}  –  ${hi:,.0f}"
    info_str  = (
        f"Model: {META['model_name']}  |  "
        f"Test R² = {META['metrics']['test_r2']}  |  "
        f"Avg error ≈ {META['metrics']['test_mape_pct']:.1f}%"
    )
    return price_str, range_str, info_str


# ── 5. Gradio Blocks UI ───────────────────────────────────────────────────────

DESCRIPTION = """
Predict the **sale price of a residential house** using a Linear Regression model
trained on 1,500 Ames-style housing records.

Enter the property details below and click **Predict Price**.  
All fields have sensible defaults — you can change only what you know.
"""

ARTICLE = """
---
**How it works**  
The model uses 42 features including size, quality, age, garage, and neighbourhood.  
Features are preprocessed identically to training: ordinal encoding for quality ratings,  
one-hot encoding for neighbourhood and style, and StandardScaler for all numerics.

**Confidence range**  
The 90% range is estimated as ±1.65 × the model's test-set MAPE (5.8%).  
For production use, replace with bootstrapped confidence intervals.

**Source** — Built with scikit-learn, pandas, and Gradio.  
"""

# Three example houses that show the model's range
EXAMPLES = [
    # qual cond  area  bsmt  gcars garea beds  fbath hbath rooms fire  lot    ybuilt yremod ysold bsmt  nbhd       style     garage  ca  salecond
    [4,   5,    750,  500,  1,    240,  2,    1,    0,    4,    0,    6000,  1955,  1955,  2008, "Fa", "OldTown", "1Story", "Detchd", "N", "Normal"],
    [6,   5,    1500, 900,  2,    440,  3,    2,    1,    7,    1,    9500,  1985,  2001,  2008, "TA", "CollgCr", "1Story", "Attchd", "Y", "Normal"],
    [9,   8,    2800, 2000, 3,    720,  4,    3,    1,    10,   3,    18000, 2003,  2008,  2009, "Ex", "NridgHt", "2Story", "Attchd", "Y", "Normal"],
]

EXAMPLE_LABELS = ["Modest starter home", "Average family home", "Luxury property"]

with gr.Blocks(
    title="House Price Predictor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("DM Mono"),
    ),
    css="""
    /* ── Layout ── */
    .contain { max-width: 960px !important; margin: 0 auto !important; }
    .predict-btn { min-height: 52px !important; font-size: 1.05rem !important; font-weight: 600 !important; }

    /* ── Result panel ── */
    #price-box textarea      { font-size: 2.6rem !important; font-weight: 700 !important;
                               text-align: center !important; color: #1d4ed8 !important;
                               line-height: 1.2 !important; }
    #range-box textarea      { text-align: center !important; font-size: 1rem !important;
                               color: #475569 !important; }
    #info-box  textarea      { text-align: center !important; font-size: 0.8rem !important;
                               color: #94a3b8 !important; }

    /* ── Section headers ── */
    .section-header          { font-weight: 600 !important; font-size: 0.85rem !important;
                               text-transform: uppercase !important; letter-spacing: .06em !important;
                               color: #64748b !important; margin-bottom: 4px !important; }

    /* ── Subtle card panels ── */
    .gr-group                { border-radius: 12px !important; }
    """,
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown("# 🏡  House Price Predictor")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ══ LEFT COLUMN — inputs ═══════════════════════════════════════════
        with gr.Column(scale=3):

            # — Section 1: Core property details —
            gr.Markdown("### Property details", elem_classes="section-header")
            with gr.Group():
                with gr.Row():
                    gr_liv_area   = gr.Slider(400,  4000, value=1500, step=50,
                                              label="Living area (sq ft)")
                    total_bsmt_sf = gr.Slider(0,    3000, value=900,  step=50,
                                              label="Basement area (sq ft)")
                with gr.Row():
                    lot_area      = gr.Slider(3000, 25000, value=9000, step=500,
                                             label="Lot area (sq ft)")
                    overall_qual  = gr.Slider(1, 10, value=6, step=1,
                                              label="Overall quality (1–10)")
                with gr.Row():
                    overall_cond  = gr.Slider(1, 9, value=5, step=1,
                                              label="Overall condition (1–9)")
                    year_built    = gr.Slider(1900, 2010, value=1975, step=1,
                                              label="Year built")
                with gr.Row():
                    year_remod    = gr.Slider(1900, 2010, value=1990, step=1,
                                              label="Year remodelled")
                    yr_sold       = gr.Slider(2006, 2010, value=2008, step=1,
                                              label="Year sold")

            # — Section 2: Rooms & bathrooms —
            gr.Markdown("### Rooms & bathrooms", elem_classes="section-header")
            with gr.Group():
                with gr.Row():
                    bedroom_abvgr = gr.Slider(1, 6, value=3, step=1,
                                              label="Bedrooms above grade")
                    full_bath     = gr.Slider(1, 4, value=2, step=1,
                                              label="Full bathrooms")
                with gr.Row():
                    half_bath     = gr.Slider(0, 2, value=0, step=1,
                                              label="Half bathrooms")
                    tot_rms_abvgrd = gr.Slider(2, 14, value=7, step=1,
                                               label="Total rooms above grade")
                with gr.Row():
                    fireplaces    = gr.Slider(0, 4, value=1, step=1,
                                              label="Fireplaces")

            # — Section 3: Garage —
            gr.Markdown("### Garage", elem_classes="section-header")
            with gr.Group():
                with gr.Row():
                    garage_cars   = gr.Slider(0, 4, value=2, step=1,
                                              label="Garage capacity (cars)")
                    garage_area   = gr.Slider(0, 1500, value=480, step=20,
                                              label="Garage area (sq ft)")
                garage_type   = gr.Radio(GARAGE_TYPES, value="Attchd",
                                         label="Garage type")

            # — Section 4: Character & location —
            gr.Markdown("### Character & location", elem_classes="section-header")
            with gr.Group():
                with gr.Row():
                    neighborhood  = gr.Dropdown(NEIGHBORHOODS, value="CollgCr",
                                                label="Neighbourhood")
                    house_style   = gr.Dropdown(HOUSE_STYLES,  value="1Story",
                                                label="House style")
                with gr.Row():
                    bsmt_qual     = gr.Dropdown(BSMT_QUAL,     value="Gd",
                                                label="Basement quality")
                    central_air   = gr.Radio(["Y", "N"], value="Y",
                                             label="Central air conditioning")
                sale_condition = gr.Dropdown(SALE_CONDS, value="Normal",
                                             label="Sale condition")

        # ══ RIGHT COLUMN — result panel ════════════════════════════════════
        with gr.Column(scale=2):
            gr.Markdown("### Predicted price", elem_classes="section-header")

            with gr.Group():
                price_out = gr.Textbox(
                    label="Estimated sale price",
                    placeholder="—",
                    interactive=False,
                    elem_id="price-box",
                )
                range_out = gr.Textbox(
                    label="90% confidence range",
                    placeholder="—",
                    interactive=False,
                    elem_id="range-box",
                )
                info_out  = gr.Textbox(
                    label="Model info",
                    placeholder="—",
                    interactive=False,
                    elem_id="info-box",
                )

            predict_btn = gr.Button(
                "🔮  Predict Price",
                variant="primary",
                elem_classes="predict-btn",
            )

            gr.Markdown(
                "> **Tip:** Drag any slider and click Predict.  "
                "All 20 fields map directly to the model's training features.",
                elem_classes="tip-text",
            )

            # ── Examples ────────────────────────────────────────────────
            gr.Markdown("### Quick examples", elem_classes="section-header")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[
                    overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
                    garage_cars, garage_area, bedroom_abvgr, full_bath, half_bath,
                    tot_rms_abvgrd, fireplaces, lot_area,
                    year_built, year_remod, yr_sold,
                    bsmt_qual, neighborhood, house_style, garage_type,
                    central_air, sale_condition,
                ],
                outputs=[price_out, range_out, info_out],
                fn=predict,
                cache_examples=True,
                label="",
            )

    gr.Markdown(ARTICLE)

    # ── Wire up the button ────────────────────────────────────────────────
    ALL_INPUTS = [
        overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
        garage_cars, garage_area, bedroom_abvgr, full_bath, half_bath,
        tot_rms_abvgrd, fireplaces, lot_area,
        year_built, year_remod, yr_sold,
        bsmt_qual, neighborhood, house_style, garage_type,
        central_air, sale_condition,
    ]
    ALL_OUTPUTS = [price_out, range_out, info_out]

    predict_btn.click(
        fn=predict,
        inputs=ALL_INPUTS,
        outputs=ALL_OUTPUTS,
    )

    # Also predict when any slider changes (live mode — optional)
    # Uncomment the loop below for live updates:
    # for w in ALL_INPUTS:
    #     w.change(fn=predict, inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)


# ── 6. Launch ─────────────────────────────────────────────────────────────────
#
# share=False   : HF Spaces handles routing; no tunnel needed
# server_name="0.0.0.0" : listen on all interfaces inside the container
# debug=False   : production — suppress verbose stack traces in UI

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
    )
