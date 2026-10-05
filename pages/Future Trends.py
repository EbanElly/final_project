import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split

# --- Page Config ---
st.set_page_config(page_title="Forecasting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2e7d32;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .stSubheader {
        color: #6d4c41;
        font-size: 20px;
        margin-top: 15px;
    }
    .stText {
        color: #424242;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Check Authentication ---
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.markdown('<div class="stHeader">Access Denied</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Please <a href="/">log in</a> or sign up to access this page.</div>', unsafe_allow_html=True)
    st.stop()

# --- Sidebar ---
st.sidebar.markdown(f'<div class="stText">Welcome, {st.session_state["username"]}!</div>', unsafe_allow_html=True)
if st.sidebar.button("Logout"):
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    try:
        st.switch_page("Home.py")
    except AttributeError:
        st.rerun()

# --- Database Functions ---
def save_forecast(user_id, disease, region, scenario, forecast_df):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    for _, row in forecast_df.iterrows():
        c.execute("""
            INSERT INTO forecasts (user_id, disease, region, scenario, year, forecasted_cases)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            disease,
            region,
            scenario,
            str(row['ds'].year),
            row['yhat']
        ))
    conn.commit()
    conn.close()

# --- Load Data ---
@st.cache_data
def load_health_data():
    df = pd.read_csv("Data/Tanzania_Health_Data_Updated.csv")
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df['Region'] = df['Region'].str.lower().str.strip()
    return df

@st.cache_data
def load_facilities_data():
    df = pd.read_csv("data/health_facilities_distribution_2025.csv")
    df['Region'] = df['Region'].str.lower().str.strip()
    return df

@st.cache_data
def load_sanitation_data():
    df = pd.read_csv("data/sanitation_water_supply.csv")
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df['Region'] = df['Region'].str.lower().str.strip()
    return df

health_data = load_health_data()
facilities_data = load_facilities_data()
sanitation_data = load_sanitation_data()

# --- Main Content ---
st.markdown('<div class="stHeader">📈 Disease Outbreak Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="stText">Predict future disease cases with scenario analysis based on past data.</div>', unsafe_allow_html=True)

# --- UI Selection ---
disease = st.selectbox("Choose a Disease", ["Cholera", "Typhoid"], index=None)
region = st.selectbox("Choose a Region", sorted(health_data['Region'].unique()), index=None)
n_years = st.slider("How Many Years to Forecast?", min_value=1, max_value=10, value=5)

# Generate Forecast Button
if st.button("Generate Forecast"):
    if not disease or not region:
        st.warning("Please choose both a disease and a region to generate a forecast.")
    else:
        disease_column = "Cholera_Cases" if disease == "Cholera" else "Typhoid_Cases"
        
        # --- Data Preparation ---
        region_health = health_data[health_data["Region"] == region][["Year", disease_column, "Region"]].copy()
        region_health = region_health.sort_values("Year").rename(columns={disease_column: "y"})
        if region_health.empty or region_health["y"].eq(0).all():
            st.warning(f"No outbreak data available for {region.capitalize()}. Skipping forecast.")
        else:
            # Impute missing values with linear interpolation
            region_health = region_health.set_index("Year").interpolate().reset_index()

            # Merge with facilities and sanitation data (latest year)
            latest_sanitation = sanitation_data[sanitation_data["Year"] == sanitation_data["Year"].max()]
            region_facilities = facilities_data[facilities_data["Region"] == region]
            merged_data = region_health.merge(region_facilities[["Region", "Total"]], on="Region", how="left") \
                                      .merge(latest_sanitation[["Region", "Sanitation_Level"]], on="Region", how="left") \
                                      .fillna({"Total": 0, "Sanitation_Level": latest_sanitation["Sanitation_Level"].mean()})

            # Prepare Prophet input
            prophet_df = merged_data.rename(columns={"Year": "ds", "y": "y"})[["ds", "y", "Total", "Sanitation_Level"]]
            train_df, test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)

            # --- Model Training and Forecasting ---
            model = Prophet(yearly_seasonality=True, seasonality_prior_scale=10)
            model.add_regressor("Total")
            model.add_regressor("Sanitation_Level")

            try:
                model.fit(train_df)
                forecast_future = model.make_future_dataframe(periods=n_years, freq='Y')
                
                # Use latest values for regressors
                latest_total = merged_data["Total"].iloc[-1]
                latest_sanitation = merged_data["Sanitation_Level"].iloc[-1]
                forecast_future["Total"] = latest_total
                forecast_future["Sanitation_Level"] = latest_sanitation
                
                # Ensure 'ds' is preserved and regressors are aligned
                forecast_future = forecast_future[['ds', 'Total', 'Sanitation_Level']].copy()
                forecast = model.predict(forecast_future)
                forecast["yhat"] = forecast["yhat"].clip(lower=0)

                # Save Current Trend Forecast
                save_forecast(st.session_state['user_id'], disease, region, "Current Trend", forecast[["ds", "yhat"]])

                # --- Validation ---
                test_forecast = model.predict(test_df)
                mse = np.mean((test_forecast["yhat"] - test_df["y"])**2)
                st.markdown(f'<div class="stText">Model Validation MSE: {mse:.2f}</div>', unsafe_allow_html=True)

                # --- Visualization ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"], mode='lines+markers', name='Past Cases'))
                fig.add_trace(go.Scatter(x=test_df["ds"], y=test_df["y"], mode='markers', name='Test Actual'))
                fig.add_trace(go.Scatter(x=test_forecast["ds"], y=test_forecast["yhat"], mode='lines', name='Test Predicted'))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Future Forecast', line=dict(color='orange')))
                fig.update_layout(title=f"Outbreak Forecast for {disease} in {region.capitalize()} for the Next {n_years} Years",
                                  xaxis_title="Year", yaxis_title="Number of Cases")
                st.plotly_chart(fig, use_container_width=True)

                # --- Forecast Table ---
                st.markdown('<div class="stSubheader">📄 Future Cases (Current Trend)</div>', unsafe_allow_html=True)
                st.dataframe(forecast[["ds", "yhat"]].round(0).rename(columns={"ds": "Year", "yhat": "Forecasted_Cases"}))
                csv_current = forecast[["ds", "yhat"]].rename(columns={"ds": "Year", "yhat": "Forecasted_Cases"}).to_csv(index=False)
                st.download_button(f"Download Current Trend for {region.capitalize()}", csv_current, 
                                   file_name=f"{region}_{disease}_current_trend.csv")

                # --- Actionable Insights ---
                last_historical = prophet_df["y"].iloc[-1]
                last_forecasted = forecast["yhat"].iloc[-1]
                trend = last_forecasted - last_historical
                if trend > 0:
                    st.markdown(f'<div class="stText"><strong>Actionable Insight:</strong> <span style="color: red;">The trend is increasing. Recommend building more health facilities and launching educational campaigns on hygiene and disease prevention in {region.capitalize()}.</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="stText"><strong>Actionable Insight:</strong> <span style="color: green;">The trend is stable or decreasing. No immediate action required, but continue monitoring and maintaining current health measures in {region.capitalize()}.</span></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Forecasting failed: {e}. Using fallback method.")
                # Fallback: Simple average with trend adjustment
                last_3_years = prophet_df["y"].tail(3).mean()
                trend = (prophet_df["y"].iloc[-1] - prophet_df["y"].iloc[0]) / (len(prophet_df) - 1) if len(prophet_df) > 1 else 0
                forecast_fallback = [last_3_years + i * trend for i in range(1, n_years + 1)]
                forecast_df = pd.DataFrame({
                    "ds": [prophet_df["ds"].iloc[-1] + timedelta(days=365 * i) for i in range(1, n_years + 1)],
                    "yhat": forecast_fallback
                })
                forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)
                save_forecast(st.session_state['user_id'], disease, region, "Fallback Trend", forecast_df)
                st.dataframe(forecast_df[["ds", "yhat"]].round(0).rename(columns={"ds": "Year", "yhat": "Forecasted_Cases"}))

# --- Footer ---
st.sidebar.markdown('<div class="stText">Developed by [Your Name/Organization] | Data as of June 05, 2025, 11:38 PM EAT</div>', unsafe_allow_html=True)