# Demand Forecasting & Pricing Optimization - Whirlpool México

Weekly SKU-level demand forecasting and price elasticity analysis built on 5.2M retail transaction records across major appliance categories in Mexico.

---

## Overview

This project builds an end-to-end ML pipeline that:
- Forecasts weekly units sold per SKU/retailer combination
- Estimates price elasticity of demand using log-log regression with seasonal controls
- Recommends revenue-maximizing prices per SKU
- Simulates revenue uncertainty via Monte Carlo analysis

Developed as part of an industry-academic collaboration with Whirlpool México. Data is proprietary and not included in this repository.

---

## Pipeline Structure

```
1. Data loading & aggregation     → daily transactions → weekly SKU/retailer grain
2. Feature engineering            → lag demand, rolling averages, promo flags, seasonality
3. Train / test split             → temporal split (no shuffling, no leakage)
4. Model training & comparison    → Ridge · XGBoost · LightGBM
5. Evaluation                     → RMSE · MAE · MAPE · R²
6. Price elasticity               → log-log OLS regression per SKU/retailer
7. Pricing optimizer              → revenue-maximizing price sweep
8. Monte Carlo simulation         → revenue confidence intervals (P5 / mean / P95)
```

---

## Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Ridge (baseline) | 17.12 | 9.11 | 70.8% | 0.608 |
| XGBoost | **11.83** | **5.59** | **41.9%** | **0.813** |
| LightGBM | 12.17 | 5.81 | 43.6% | 0.802 |

**XGBoost** selected as best model. Evaluated on Jan–Jul 2025 holdout data.

**Top features by importance:** last week's demand (`qty_lag1`), promotional flag (`promo_flag`), price change (`price_change`), 4-week rolling average (`qty_roll4`).

**Price elasticity finding:** log-log regression with seasonal controls revealed consistently inelastic demand (e = 0.1–0.7) across tested SKUs, suggesting Whirlpool holds meaningful pricing power in the Mexican appliance market.

---

## Key Methodological Decisions

**Why temporal train/test split?**
A random split would allow the model to train on future data. Splitting by date (train on 2023–2024, test on 2025) mirrors how the model would work in production.

**Why lag features?**
Without lag demand the model sees each week in isolation — it can't learn any time pattern. `qty_lag1`, `qty_lag4`, and `qty_lag52` give it short-, medium-, and long-term context.

**Why log-log regression for elasticity?**
Tree-based models capture the correlation between price and demand well but can't isolate the causal effect when prices and demand are both driven by the same seasonal factors (price endogeneity). A log-log OLS regression controlling for season tier, promotional periods, and month separates these effects. The coefficient on `log(price)` is directly interpretable as price elasticity.

---

## Tech Stack

```
Python 3.11+
pandas · numpy · scikit-learn
xgboost · lightgbm · statsmodels
matplotlib · seaborn
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels matplotlib seaborn
```

---

## How to Run

1. Clone the repo
2. Place your data files in a `/data` folder
3. Update `BASE_PATH` at the top of `whirlpool_pipeline.py`
4. Run:

```bash
python whirlpool_pipeline.py
```

Outputs (plots + metrics) will be saved to the path set in `OUTPUT_PATH`.

---

## Limitations

- **Price endogeneity:** Whirlpool's pricing strategy (higher prices during peak seasons) creates a spurious positive correlation between price and demand in observational data. Log-log regression with seasonal controls mitigates but does not fully eliminate this effect. Instrumental variable methods or randomized pricing experiments would yield more reliable elasticity estimates.
- **Training window:** Only 2 years of historical data available (2023–2024 for training), limiting the model's ability to capture multi-year seasonality patterns.
- **Sparse SKUs:** SKU/retailer combinations with fewer than 30 weekly observations were excluded from elasticity estimation due to insufficient price variation.

---

## Note on Data

This project was developed under a non-disclosure agreement with Whirlpool México. All data has been removed from this repository. The pipeline is fully functional with any dataset that matches the expected schema — see the comments in `load_data()` for the required columns.

---

*Industry-academic project · ITESM · 2025*
*Hannia Hernandez Salvador — A00837850*
