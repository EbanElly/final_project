# Tanzania Waterborne Disease Prediction System

A Streamlit app that uses machine learning and GIS to predict and visualize waterborne disease (typhoid and cholera) risk in Tanzania.

## Features

- **Predict Risk** - individual or batch (CSV) risk prediction using an ensemble of Random Forest and XGBoost models.
- **Future Trends** - Prophet forecasts of regional cholera and typhoid cases.
- **Map Visualizer** - interactive choropleth maps of cases by region.
- User accounts (SQLite + bcrypt) with prediction and forecast history.

## Setup

```bash
python -m venv env
env\Scripts\activate          # Windows  (source env/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run Home.py
```

The SQLite database `users.db` is created automatically on first run.

## Project layout

| Path | Contents |
|---|---|
| `Home.py` | Landing page, login/sign-up, user dashboard |
| `pages/` | Predict Risk, Future Trends, Map Visualizer pages |
| `train.py` | Generates the synthetic training set and trains/saves the models |
| `models/` | Trained models, label encoders, feature scaler, performance report |
| `data/` | Regional health, facilities, sanitation data and Tanzania boundary GeoJSON |
| `images/` | Images used in the UI |

## Retraining

```bash
python train.py
```

Note: the individual-level training data is **synthetic** (generated in `train.py`), so predictions are for demonstration only and are not medical advice.

## Data sources

Health facility points from OpenStreetMap via HOTOSM (ODbL); regional boundaries from geoBoundaries.
