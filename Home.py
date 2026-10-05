import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
from datetime import datetime

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="Tanzania Waterborne Disease Prediction System", layout="wide")

# --- Styling with Updated Colors ---
st.markdown(
    """
    <style>
    .stHeader {
        color: #1A3C34; /* Deep teal for headers */
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .stSubheader {
        color: #2E7D32; /* Forest green for subheaders */
        font-size: 20px;
        margin-top: 15px;
    }
    .stText {
        color: #333333; /* Dark gray for body text */
        font-size: 16px;
    }
    .stButton>button {
        background-color: #0288D1; /* Bright blue for buttons */
        color: white;
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 16px; /* Increased for better visibility */
        border: none;
        margin-left: 5px;
    }
    .stButton>button:hover {
        background-color: #0277BD; /* Darker blue on hover */
    }
    .activeButton>button {
        background-color: #0277BD; /* Matches hover state for active button */
        color: white;
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 16px;
        border: none;
        margin-left: 5px;
    }
    .footer {
        color: #78909C; /* Muted blue-gray for footer */
        font-size: 12px;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-top: 1px solid #bdc3c7;
    }
    .image-container {
        text-align: center;
        margin-bottom: 20px;
        margin-top: 20px;
    }
    .custom-title {
        color: #2E7D32; /* Forest green for the title */
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .welcome-text {
        color: #333333;
        font-size: 18px;
        margin-bottom: 30px;
        padding: 0 20px; /* Add padding for better spacing */
    }
    .about-text {
        color: #333333;
        font-size: 16px;
        padding: 0 20px; /* Add padding for better spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Custom Title ---
st.markdown('<div class="custom-title">🩺 Waterborne Disease Prediction System</div>', unsafe_allow_html=True)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction_type TEXT,
            age INTEGER,
            gender TEXT,
            fever_severity INTEGER,
            diarrhea_severity INTEGER,
            abdominal_pain_severity INTEGER,
            water_source TEXT,
            sanitation_level TEXT,
            region TEXT,
            vaccination_status TEXT,
            recent_travel TEXT,
            food_hygiene TEXT,
            case_prevalence REAL,
            prediction TEXT,
            no_disease_prob REAL,
            typhoid_prob REAL,
            cholera_prob REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            disease TEXT,
            region TEXT,  
            year TEXT,
            forecasted_cases REAL,
            lower_ci REAL,
            upper_ci REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    try:
        c.execute("ALTER TABLE forecasts ADD COLUMN scenario TEXT;")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
    
    c.execute("CREATE INDEX IF NOT EXISTS idx_user_id_predictions ON predictions(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_user_id_forecasts ON forecasts(user_id)")
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# --- Authentication Functions ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def signup(username, password, email, full_name=""):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                  (username, hashed, email))
        user_id = c.lastrowid
        c.execute("INSERT INTO user_profiles (user_id, full_name, role) VALUES (?, ?, ?)",
                  (user_id, full_name, 'user'))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
        return False
    finally:
        conn.close()

def login(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        user_id, hashed = result
        if verify_password(password, hashed):
            return user_id
        else:
            st.error("Incorrect password.")
            return None
    else:
        st.error("Username not found.")
        return None

def get_user_history(user_id):
    conn = sqlite3.connect("users.db")
    predictions_df = pd.read_sql_query("""
        SELECT prediction_type, age, gender, region, prediction, no_disease_prob, 
               typhoid_prob, cholera_prob, timestamp 
        FROM predictions 
        WHERE user_id = ?
    """, conn, params=(user_id,))
    
    forecasts_df = pd.read_sql_query("""
        SELECT disease, region, year, forecasted_cases, lower_ci, upper_ci, timestamp 
        FROM forecasts 
        WHERE user_id = ?
    """, conn, params=(user_id,))
    
    conn.close()
    return predictions_df, forecasts_df

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'form_selection' not in st.session_state:
    st.session_state.form_selection = None
if 'get_started' not in st.session_state:
    st.session_state.get_started = False

# --- Main App Logic ---
if not st.session_state.get_started:
    # Welcome Section with Mirrored Layout
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="welcome-text">Welcome to our platform! Discover how we predict and visualize waterborne disease outbreaks in Tanzania using GIS and machine learning. Click below to get started.</div>', unsafe_allow_html=True)
    with col2:
        st.image("images/mapping.jpg", caption="Map future outbreaks on the Map", width=400)
    st.button("Get Started", on_click=lambda: st.session_state.update({"get_started": True}))
else:
    if not st.session_state.logged_in:
        # Display Login and Sign Up buttons in the top-right corner
        col1, col2 = st.columns([8, 2])
        with col2:
            login_col, signup_col = st.columns([1, 1])
            with login_col:
                if st.button("Login", key="login_btn"):
                    st.session_state.form_selection = "login"
            with signup_col:
                if st.button("Sign Up", key="signup_btn"):
                    st.session_state.form_selection = "signup"

            # Highlight the active button
            if st.session_state.form_selection == "login":
                st.markdown('<style>.loginBtn>button {background-color: #0277BD;}</style>', unsafe_allow_html=True)
            elif st.session_state.form_selection == "signup":
                st.markdown('<style>.signupBtn>button {background-color: #0277BD;}</style>', unsafe_allow_html=True)

        # Display the selected form (centered)
        if st.session_state.form_selection == "login":
            col_form1, col_form2, col_form3 = st.columns([1, 2, 1])
            with col_form2:
                st.markdown('<div class="stSubheader">Login</div>', unsafe_allow_html=True)
                with st.form("login_form"):
                    login_username = st.text_input("Username", key="login_username")
                    login_password = st.text_input("Password", type="password", key="login_password")
                    login_submit = st.form_submit_button("Login")

                if login_submit:
                    user_id = login(login_username, login_password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_id = user_id
                        st.success(f"Welcome back, {login_username}!")
                        st.rerun()

        elif st.session_state.form_selection == "signup":
            col_form1, col_form2, col_form3 = st.columns([1, 2, 1])
            with col_form2:
                st.markdown('<div class="stSubheader">Sign Up</div>', unsafe_allow_html=True)
                with st.form("signup_form"):
                    signup_username = st.text_input("Username", key="signup_username")
                    signup_password = st.text_input("Password", type="password", key="signup_password")
                    signup_email = st.text_input("Email (Optional)", key="signup_email")
                    signup_full_name = st.text_input("Full Name (Optional)", key="signup_full_name")
                    signup_submit = st.form_submit_button("Sign Up")

                if signup_submit:
                    if signup_username and signup_password:
                        if signup(signup_username, signup_password, signup_email, signup_full_name):
                            st.success("Account created successfully! Please log in.")
                            st.session_state.form_selection = "login"
                            st.rerun()
                    else:
                        st.error("Username and password are required.")

        # --- About This App Section with Mirrored Layout ---
        st.markdown('<div class="stHeader">About This App</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("images/info.jpg", caption="Empowering Health Interventions with Data", width=400)
        with col2:
            st.markdown("""
            <div class="about-text">
            The Tanzania Waterborne Disease Prediction System uses GIS and machine learning to predict and visualize waterborne disease outbreaks, such as typhoid and cholera. Designed for healthcare professionals, researchers, and policymakers, it offers:
             <strong>Predict Outbreaks</strong>: Forecast disease risks.
             <strong>Visualize Trends</strong>: Interactive maps for monitoring outbreaks.
             <strong>Support Interventions</strong>: Data-driven insights for public health.

            Log in or sign up to access secure dashboards and reports!
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="stSubheader">Welcome, {st.session_state.username}!</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stSubheader">Your Dashboard</div>', unsafe_allow_html=True)
        predictions_df, forecasts_df = get_user_history(st.session_state.user_id)
        
        st.markdown('<div class="stSubheader">Prediction History</div>', unsafe_allow_html=True)
        if not predictions_df.empty:
            st.dataframe(predictions_df)
            csv = predictions_df.to_csv(index=False)
            st.download_button("Download Prediction History", csv, "prediction_history.csv", "text/csv")
        else:
            st.info("No prediction history available.")
        
        st.markdown('<div class="stSubheader">Forecast History</div>', unsafe_allow_html=True)
        if not forecasts_df.empty:
            st.dataframe(forecasts_df)
            csv = forecasts_df.to_csv(index=False)
            st.download_button("Download Forecast History", csv, "forecast_history.csv", "text/csv")
        else:
            st.info("No forecast history available.")
        
        st.markdown("---")
        st.markdown('<div class="stText">Navigate to the following pages using the sidebar:</div>', unsafe_allow_html=True)
        st.markdown('<div class="stText">- Predict Risk: Predict disease risk for individuals or batches.</div>', unsafe_allow_html=True)
        st.markdown('<div class="stText">- Future trends: Forecast future disease cases by region.</div>', unsafe_allow_html=True)
        st.markdown('<div class="stText">- Map Visualizer: View interactive maps of disease outbreaks.</div>', unsafe_allow_html=True)

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.success("Logged out successfully.")
            st.rerun()

# --- Footer ---
st.markdown(
    '<div class="footer">Developed by [Your Name/Organization] | Data as of June 06, 2025, 08:33 AM EAT</div>',
    unsafe_allow_html=True
)