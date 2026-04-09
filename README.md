# Stock ML + Power BI Dashboard (Analyst-Grade)

Standalone ML + BI prototype (no Django) that:
- **pulls historical stock prices**
- **engineers technical features**
- **trains LSTM / Random Forest / XGBoost regressors**
- **exports model predictions + metrics to CSV**
- **builds an interactive Power BI report**

## Quickstart

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key (recommended):

- Windows PowerShell:

```powershell
$env:ALPHAVANTAGE_API_KEY="YOUR_KEY"
```

4. Run:

```bash
python stock_prediction_ml.py
```

## Outputs

### CSVs for Power BI
All CSVs are written into `outputs/`:
- `outputs/historical_data.csv` - engineered historical features
- `outputs/predictions.csv` - wide prediction output (Actual + all model predictions)
- `outputs/predictions_long.csv` - long format (recommended for visuals + slicers)
- `outputs/model_metrics.csv` - RMSE and MAE by model
- `outputs/model_summary.csv` - KPI-friendly summary table
- `outputs/feature_importance.csv` - feature importance (RF)

### Saved models
Models/scaler are written into `models/` (not required for Power BI visuals):
- `models/rf_model.joblib`
- `models/xgb_model.joblib`
- `models/feature_scaler.joblib`
- `models/lstm_model.keras` (if TensorFlow is available)

## Power BI: Data Load (Recommended)

1. Open Power BI Desktop.
2. Select **Get Data > Text/CSV**.
3. Load:
   - `outputs/predictions_long.csv`
   - `outputs/model_summary.csv`
   - `outputs/feature_importance.csv`
   - `outputs/historical_data.csv` (optional)
4. In Power Query:
   - Ensure `Date` is type **Date** for `historical_data` and `predictions_long`.
   - Ensure numeric columns are Decimal/Whole Number.
5. Close & Apply.

## Data Model (Simple + Reliable)

Use `predictions_long` as your primary visual table.

Optional relationship:
- `historical_data[Date]` (one) -> `predictions_long[Date]` (many)

## DAX Measures (Copy-Paste)

Create these in Power BI:

```DAX
Pred Avg = AVERAGE(predictions_long[Predicted])
Actual Avg = AVERAGE(predictions_long[Actual])
Abs Error Avg = AVERAGE(predictions_long[Abs_Error])
RMSE Avg = AVERAGE(model_summary[RMSE])
MAE Avg = AVERAGE(model_summary[MAE])

Forecast Signal = IF([Pred Avg] > [Actual Avg], "Buy", "Sell")

Best Model (RMSE) =
VAR bestRmse = MIN(model_summary[RMSE])
RETURN
    MAXX(
        FILTER(model_summary, model_summary[RMSE] = bestRmse),
        model_summary[Model]
    )
```

## Dashboard Layout (Clear + Attractive)

Use a single-page canvas first, then duplicate for detail pages if needed.

### Top Row: KPI Cards
- Card 1: `Best Model (RMSE)`
- Card 2: `RMSE Avg`
- Card 3: `MAE Avg`
- Card 4: `Forecast Signal`

Style:
- Background: white
- Card corners: 8 px
- Shadow: subtle
- Label color: `#6B7280`
- Value color: `#111827`

### Middle Left: Main Comparison Line Chart
- Visual: **Line chart**
- Axis: `predictions_long[Date]`
- Legend: `predictions_long[Model]`
- Values: `Pred Avg`
- Add `Actual Avg` as second line (Analytics/secondary values depending on visual type)

Recommended colors:
- LSTM: `#2563EB` (blue)
- RF: `#10B981` (green)
- XGB: `#F59E0B` (amber)
- Actual: `#111827` (dark gray)

### Middle Right: Feature Importance
- Visual: **Clustered bar chart**
- Axis: `feature_importance[Feature]`
- Values: `feature_importance[Importance]`
- Sort descending by `Importance`
- Turn on data labels

### Bottom: Prediction Detail Table
- Visual: **Table**
- Columns: `Date`, `Model`, `Actual`, `Predicted`, `Abs_Error`
- Add conditional formatting:
  - `Abs_Error`: green (low) to red (high)

## Interactivity Setup

Add slicers:
- `predictions_long[Model]` (dropdown, single/multi-select)
- `predictions_long[Date]` (between slider)

Enable interactions:
- Slicers should filter all visuals.
- Keep cross-highlighting on for line -> table interaction.

## Theme Suggestions (Professional Look)

- Page background: `#F8FAFC`
- Font: Segoe UI
- Base text: `#111827`
- Secondary text: `#6B7280`
- Accent: `#2563EB`

Keep whitespace generous and avoid visual clutter.

## Publish and Share

1. Save PBIX file.
2. Click **Publish** and choose workspace.
3. In Power BI Service:
   - Configure scheduled refresh if needed.
   - Share dashboard link or embed.

## Optional Enhancements

- Add `Stock` column when you expand to multiple symbols.
- Add page tooltips for model explanations.
- Add a decomposition tree for error analysis by date periods.
- Use Python visual for custom residual/error distribution charts.

## Notes (Professional / Reproducible)
- Alpha Vantage free tier can throttle requests. The script automatically falls back to `yfinance` if throttled.
- `Forecast Signal` is a directional indicator (Predicted vs Actual) meant for demo/analysis, not financial advice.
