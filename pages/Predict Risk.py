import streamlit as st
import pandas as pd
import numpy as np
from folium import Map
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disease Risk Predictor", layout="wide", initial_sidebar_state="expanded")

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
        color: #2e7d32 !important;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .legend-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-left: 20px;
        height: 100%;
    }
    .legend-gradient {
        width: 20px;
        height: 200px;
        background: linear-gradient(to bottom, blue, green, yellow, red);
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .legend-labels {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 200px;
        font-size: 14px;
        color: #424242;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.markdown('<div class="stHeader">Access Denied</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Please <a href="/">log in</a> or sign up to access this page.</div>', unsafe_allow_html=True)
    st.stop()

st.sidebar.title("Navigation")
pages = ["Disease Prediction", "Health Risk Map"]
page = st.sidebar.radio("Go to", pages)

# --- Sidebar ---
st.sidebar.markdown(f'<div class="stText">Welcome, {st.session_state["username"]}!</div>', unsafe_allow_html=True)
if st.sidebar.button("Logout"):
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    try:
        st.switch_page("Home.py")  # Redirect to the home page (adjust if your home page is named differently)
    except AttributeError:
        st.rerun()  # Fallback if switch_page is not available (older Streamlit versions)

@st.cache_data
def load_data():
    health_facilities = pd.read_csv("data/health_facilities_distribution_2025.csv")
    sanitation_data = pd.read_csv("data/sanitation_water_supply.csv")
    regional_data = pd.read_csv("data/Tanzania_Health_Data_Updated.csv")
    regional_data['Region'] = regional_data['Region'].str.lower().str.strip()
    regional_data['case_prevalence'] = regional_data[['Cholera_Cases', 'Typhoid_Cases']].mean(axis=1).fillna(0)
    return health_facilities, sanitation_data, regional_data

health_facilities, sanitation_data, regional_data = load_data()

health_facilities_agg = health_facilities.groupby("Region")["Total"].sum().reset_index()
health_facilities_agg.rename(columns={"Total": "Number_of_Facilities"}, inplace=True)
health_facilities_agg['Region'] = health_facilities_agg['Region'].str.lower().str.strip()

latest_sanitation = sanitation_data[sanitation_data["Year"] == 2024]
latest_sanitation['Region'] = latest_sanitation['Region'].str.lower().str.strip()
avg_sanitation_level = latest_sanitation["Sanitation_Level"].mean()
avg_water_access = latest_sanitation["Water_Access_Percentage"].mean()
all_regions = pd.DataFrame({"Region": health_facilities_agg["Region"].unique()})
latest_sanitation_extended = all_regions.merge(
    latest_sanitation[["Region", "Sanitation_Level", "Water_Access_Percentage"]],
    on="Region",
    how="left"
).fillna({
    "Sanitation_Level": avg_sanitation_level,
    "Water_Access_Percentage": avg_water_access
})
merged_data = latest_sanitation_extended.merge(
    health_facilities_agg[["Region", "Number_of_Facilities"]],
    on="Region",
    how="left"
).fillna({"Number_of_Facilities": 0})

region_coords = {
    "arusha": (-3.3667, 36.6833),
    "dar es salaam": (-6.7924, 39.2083),
    "dodoma": (-6.1724, 35.7395),
    "geita": (-2.8700, 32.2500),
    "iringa": (-7.7667, 35.7000),
    "kagera": (-1.3333, 31.6667),
    "mara": (-1.7500, 34.0000),
    "mbeya": (-8.9000, 33.4500),
    "morogoro": (-6.8167, 37.6667),
    "mtwara": (-10.2667, 40.1833),
    "mwanza": (-2.5167, 32.9000),
    "njombe": (-9.3167, 34.7667),
    "pwani": (-7.3333, 38.9167),
    "rukwa": (-7.7500, 31.6167),
    "ruvuma": (-10.6833, 35.6333),
    "shinyanga": (-3.6600, 33.4200),
    "simuyu": (-3.5000, 34.0000),
    "singida": (-6.1167, 34.7500),
    "songwe": (-9.0000, 32.9000),
    "tabora": (-5.0167, 32.8000),
    "tanga": (-5.0833, 39.1000)
}

if page == "Disease Prediction":
    st.markdown('<div class="stHeader">Disease Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Find out your risk of typhoid and cholera based on your info and surroundings.</div>', unsafe_allow_html=True)

    try:
        rf_model = joblib.load("models/rf_model.pkl")
        xgb_model = joblib.load("models/xgb_model.pkl")
        le_gender = joblib.load("models/le_gender.pkl")
        le_water = joblib.load("models/le_water.pkl")
        le_region = joblib.load("models/le_region.pkl")
        le_vaccination = joblib.load("models/le_vaccination.pkl")
        le_travel = joblib.load("models/le_travel.pkl")
        le_food = joblib.load("models/le_food.pkl")
    except Exception as e:
        st.error(f"Oops! Something went wrong loading the prediction tools: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="stSubheader">Your Details</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=5, max_value=80, value=30, help="How old are you in years?")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=None, help="Your gender?")
        region = st.selectbox("Region", options=merged_data["Region"].unique(), index=None, help="Which region in Tanzania are you in?")
        water_source = st.selectbox("Water Source", options=["Piped", "Well", "River", "Rainwater"], index=None, help="What is your primary water source?")
        vaccination_status = st.selectbox("Vaccination Status", options=["Yes", "No"], index=None, help="Have you been vaccinated?")
        recent_travel = st.selectbox("Recent Travel", options=["Yes", "No"], index=None, help="Have you traveled recently?")
        food_hygiene = st.selectbox("Food Hygiene", options=["Good", "Poor"], index=None, help="Is your food hygiene good or poor?")

    with col2:
        st.markdown('<div class="stSubheader">Symptoms (0 = None, 10 = Severe)</div>', unsafe_allow_html=True)
        fever_severity = st.slider("Fever Severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        diarrhea_severity = st.slider("Diarrhea Severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        abdominal_pain_severity = st.slider("Abdominal Pain Severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    # Convert region to lowercase to match training data
    region = region.lower().strip() if region else merged_data["Region"].iloc[0].lower().strip()
    region_data = regional_data[regional_data["Region"] == region].iloc[0] if region in regional_data["Region"].values else regional_data.iloc[0]
    case_prevalence = region_data["case_prevalence"]

    # Handle None values for categorical inputs
    gender = gender if gender else "Male"
    water_source = water_source if water_source else "Piped"
    vaccination_status = vaccination_status if vaccination_status else "No"
    recent_travel = recent_travel if recent_travel else "No"
    food_hygiene = food_hygiene if food_hygiene else "Good"

    # Prepare input data with additional features from train.py
    input_data_single = pd.DataFrame({
        "age": [age],
        "gender": [le_gender.transform([gender])[0]],
        "fever_severity": [fever_severity],
        "diarrhea_severity": [diarrhea_severity],
        "abdominal_pain_severity": [abdominal_pain_severity],
        "water_source": [le_water.transform([water_source])[0]],
        "region": [le_region.transform([region])[0]],
        "vaccination_status": [le_vaccination.transform([vaccination_status])[0]],
        "recent_travel": [le_travel.transform([recent_travel])[0]],
        "food_hygiene": [le_food.transform([food_hygiene])[0]],
        "case_prevalence": [case_prevalence]
    })
    input_data_single["symptom_sum"] = input_data_single["fever_severity"] + input_data_single["diarrhea_severity"] + input_data_single["abdominal_pain_severity"]
    input_data_single["age_risk"] = np.where(input_data_single["age"] > 50, 1, 0)

    st.markdown('<div class="stSubheader">Predict for Many People</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Upload a file with info about many people to predict their risks.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="The file should have: age, gender, fever_severity, diarrhea_severity, abdominal_pain_severity, water_source, region, vaccination_status, recent_travel, food_hygiene (severities 0-10)")
    if uploaded_file is not None:
        input_data_batch = pd.read_csv(uploaded_file)
        required_columns = ["age", "gender", "fever_severity", "diarrhea_severity", "abdominal_pain_severity", "water_source", "region", "vaccination_status", "recent_travel", "food_hygiene"]
        if not all(col in input_data_batch.columns for col in required_columns):
            st.error(f"Missing column in your file: {', '.join(set(required_columns) - set(input_data_batch.columns))}")
            st.stop()
        for col in ["fever_severity", "diarrhea_severity", "abdominal_pain_severity"]:
            if not ((input_data_batch[col] >= 0) & (input_data_batch[col] <= 10)).all():
                st.error(f"Values in {col} must be between 0 and 10.")
                st.stop()
        input_data_batch = input_data_batch.merge(
            regional_data[["Region", "case_prevalence"]],
            on="Region",
            how="left"
        ).fillna({"case_prevalence": 0})
        # Handle None or invalid values in batch data
        input_data_batch["region"] = input_data_batch["region"].str.lower().str.strip()
        input_data_batch["gender"] = input_data_batch["gender"].fillna("Male")
        input_data_batch["water_source"] = input_data_batch["water_source"].fillna("Piped")
        input_data_batch["vaccination_status"] = input_data_batch["vaccination_status"].fillna("No")
        input_data_batch["recent_travel"] = input_data_batch["recent_travel"].fillna("No")
        input_data_batch["food_hygiene"] = input_data_batch["food_hygiene"].fillna("Good")
        input_data_batch["region"] = input_data_batch["region"].fillna(merged_data["Region"].iloc[0].lower().strip())
        # Filter out regions not in le_region.classes_
        valid_regions = [r for r in input_data_batch["region"] if r in le_region.classes_]
        if len(valid_regions) < len(input_data_batch):
            st.warning(f"Some regions in your file are not recognized by the model. Using only valid regions: {list(le_region.classes_)}")
            input_data_batch = input_data_batch[input_data_batch["region"].isin(valid_regions)]
        input_data_batch["gender"] = le_gender.transform(input_data_batch["gender"])
        input_data_batch["water_source"] = le_water.transform(input_data_batch["water_source"])
        input_data_batch["region"] = le_region.transform(input_data_batch["region"])
        input_data_batch["vaccination_status"] = le_vaccination.transform(input_data_batch["vaccination_status"])
        input_data_batch["recent_travel"] = le_travel.transform(input_data_batch["recent_travel"])
        input_data_batch["food_hygiene"] = le_food.transform(input_data_batch["food_hygiene"])
        # Add additional features
        input_data_batch["symptom_sum"] = input_data_batch["fever_severity"] + input_data_batch["diarrhea_severity"] + input_data_batch["abdominal_pain_severity"]
        input_data_batch["age_risk"] = np.where(input_data_batch["age"] > 50, 1, 0)

        try:
            rf_probs = rf_model.predict_proba(input_data_batch)
            xgb_probs = xgb_model.predict_proba(input_data_batch)
            ensemble_probs = (rf_probs + xgb_probs) / 2
            results_df = input_data_batch.copy()
            results_df["No_Disease_Prob"] = ensemble_probs[:, 0]
            results_df["Typhoid_Prob"] = ensemble_probs[:, 1]
            results_df["Cholera_Prob"] = ensemble_probs[:, 2]
            results_df["Prediction"] = np.argmax(ensemble_probs, axis=1)
            results_df["Prediction"] = results_df["Prediction"].map({0: "No Disease", 1: "Typhoid", 2: "Cholera"})
            st.markdown('<div class="stSubheader">Results for Many People</div>', unsafe_allow_html=True)
            st.dataframe(results_df.round(2))
        except Exception as e:
            st.error(f"Something went wrong while predicting for many people: {e}")
            st.stop()

    if st.button("Predict My Risk"):
        if not all([region, water_source, gender, vaccination_status, recent_travel, food_hygiene]):
            st.warning("Please fill in all the details to predict your risk.")
        else:
            try:
                rf_probs = rf_model.predict_proba(input_data_single)
                xgb_probs = xgb_model.predict_proba(input_data_single)
                ensemble_probs = (rf_probs + xgb_probs) / 2
                prediction = np.argmax(ensemble_probs, axis=1)[0]
                no_disease_prob = ensemble_probs[0, 0]
                typhoid_prob = ensemble_probs[0, 1]
                cholera_prob = ensemble_probs[0, 2]

                st.markdown('<div class="stSubheader">Your Prediction Results</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stText"><strong>Predicted Condition:</strong> {["No Disease", "Typhoid", "Cholera"][prediction]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stText"><strong>No Disease Probability:</strong> {no_disease_prob:.2%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stText"><strong>Typhoid Probability:</strong> {typhoid_prob:.2%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stText"><strong>Cholera Probability:</strong> {cholera_prob:.2%}</div>', unsafe_allow_html=True)

                # Display recommendation in red
                recommendation = ""
                if prediction == 1:  # Typhoid
                    recommendation = "Seek immediate medical attention for Typhoid. Improve water hygiene (e.g., boil or filter water) and consult a healthcare provider for antibiotics or further tests."
                elif prediction == 2:  # Cholera
                    recommendation = "Seek immediate medical attention for Cholera. Ensure access to clean water, rehydrate with oral rehydration salts, and contact a healthcare provider urgently."
                else:  # No Disease
                    recommendation = "You are at low risk. Maintain good hygiene, ensure safe water sources, and schedule regular health check-ups."

                st.markdown(f'<div class="stText"><strong>Recommendation:</strong> <span style="color: red;">{recommendation}</span></div>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                conditions = ['No Disease', 'Typhoid', 'Cholera']
                probs = [no_disease_prob, typhoid_prob, cholera_prob]

                bar_width = 0.5
                index = np.arange(len(conditions))

                ax.bar(index, probs, bar_width, color=['green', 'red', 'blue'])
                ax.set_xlabel('Conditions')
                ax.set_ylabel('Probability')
                ax.set_title('Disease Risk Prediction')
                ax.set_xticks(index)
                ax.set_xticklabels(conditions)
                ax.set_ylim(0, 1)

                for i, v in enumerate(probs):
                    ax.text(i, v, f'{v:.2%}', ha='center', va='bottom')

                st.pyplot(fig)

                # Generate report with recommendation
                report_data = pd.DataFrame({
                    "Input": ["Age", "Gender", "Region", "Water Source", "Vaccination Status", "Recent Travel", "Food Hygiene", "Fever Severity", "Diarrhea Severity", "Abdominal Pain Severity"],
                    "Value": [age, gender, region.capitalize(), water_source, vaccination_status, recent_travel, food_hygiene, fever_severity, diarrhea_severity, abdominal_pain_severity]
                })
                report_data["Prediction"] = ["N/A"] * 10
                report_data.loc[report_data["Input"] == "Region", "Prediction"] = f"{['No Disease', 'Typhoid', 'Cholera'][prediction]}"
                report_data.loc[report_data["Input"] == "No Disease Probability", "Value"] = f"{no_disease_prob:.2%}"
                report_data.loc[report_data["Input"] == "Typhoid Probability", "Value"] = f"{typhoid_prob:.2%}"
                report_data.loc[report_data["Input"] == "Cholera Probability", "Value"] = f"{cholera_prob:.2%}"
                report_data.loc[report_data["Input"] == "Recommendation", "Value"] = recommendation

                csv = report_data.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Report",
                    data=csv,
                    file_name="disease_risk_report.csv"
                )

            except Exception as e:
                st.error(f"Something went wrong while predicting your risk: {e}")
                st.stop()

if page == "Health Risk Map":
    st.markdown('<div class="stHeader">Health Risk Prediction Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="stText">Look at a map to see where health facilities are in Tanzania.</div>', unsafe_allow_html=True)

    heatmap_data = []
    data_min = 0
    data_max = 0
    for _, row in health_facilities_agg.iterrows():
        lat, lon = region_coords[row["Region"]]
        total = row["Number_of_Facilities"]
        heatmap_data.append([lat, lon, total])
    data_min = health_facilities_agg["Number_of_Facilities"].min()
    data_max = health_facilities_agg["Number_of_Facilities"].max()
    st.markdown('<div class="stText">This map shows the number of health facilities (hospitals, clinics, dispensaries) in each region.</div>', unsafe_allow_html=True)

    if heatmap_data:
        col1, col2 = st.columns([3, 1])
        with col1:
            m = Map(location=[-6.3690, 34.8888], zoom_start=6)
            HeatMap(heatmap_data, radius=15).add_to(m)
            folium_static(m)

        with col2:
            mid_value = (data_min + data_max) / 2
            st.markdown(
                f"""
                <div class="legend-container">
                    <div class="legend-gradient"></div>
                    <div class="legend-labels">
                        <span>{data_max:.0f}</span>
                        <span>{mid_value:.0f}</span>
                        <span>{data_min:.0f}</span>
                    </div>
                </div>
                <div class="stText" style="text-align: center;">
                    Number of Health Facilities
                </div>
                """,
                unsafe_allow_html=True
            )

st.sidebar.markdown('<div class="stText">Made by [Your Name/Organization] | Data as of June 05, 2025, 11:24 PM EAT</div>', unsafe_allow_html=True)