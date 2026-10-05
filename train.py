import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.cluster import KMeans
from scipy.stats import randint, uniform

# Load regional data
try:
    regional_data = pd.read_csv("data/Tanzania_Health_Data_Updated.csv")
    regional_data['Region'] = regional_data['Region'].str.lower().str.strip()
    regions = sorted(regional_data['Region'].unique())
    regional_data['case_prevalence'] = regional_data[['Cholera_Cases', 'Typhoid_Cases']].mean(axis=1).fillna(0)
except FileNotFoundError as e:
    print(f"Error loading regional data: {e}. Using default regions and synthetic prevalence.")
    regions = ["dar es salaam", "arusha", "dodoma", "mwanza", "mbeya"]
    regional_data = pd.DataFrame({
        "Region": regions,
        "Cholera_Cases": np.random.uniform(5, 20, len(regions)),
        "Typhoid_Cases": np.random.uniform(10, 30, len(regions)),
        "case_prevalence": np.random.uniform(15, 25, len(regions))
    })

# Generate synthetic data with clustering
np.random.seed(42)
n_samples = 50000

data = {
    "age": np.random.randint(5, 80, n_samples),
    "gender": np.random.choice(["Male", "Female"], n_samples, p=[0.52, 0.48]),
    "fever_severity": np.zeros(n_samples),
    "diarrhea_severity": np.zeros(n_samples),
    "abdominal_pain_severity": np.zeros(n_samples),
    "water_source": np.random.choice(["Piped", "Well", "River", "Rainwater"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    "region": np.random.choice(regions, n_samples),
    "vaccination_status": np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
    "recent_travel": np.random.choice(["Yes", "No"], n_samples, p=[0.2, 0.8]),
    "food_hygiene": np.random.choice(["Good", "Poor"], n_samples, p=[0.7, 0.3]),
    "case_prevalence": np.zeros(n_samples),
    "disease": np.zeros(n_samples, dtype=int)
}

# Assign case_prevalence
for i in range(n_samples):
    region = data["region"][i]
    data["case_prevalence"][i] = regional_data[regional_data["Region"] == region]["case_prevalence"].iloc[0]

# Cluster-based symptom assignment with array clipping
X_temp = pd.DataFrame({
    "water_source": LabelEncoder().fit_transform(data["water_source"]),
    "vaccination_status": LabelEncoder().fit_transform(data["vaccination_status"]),
    "recent_travel": LabelEncoder().fit_transform(data["recent_travel"]),
    "food_hygiene": LabelEncoder().fit_transform(data["food_hygiene"]),
    "case_prevalence": data["case_prevalence"]
})
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_temp)

# Generate symptom arrays and clip them
fever_severity = np.zeros(n_samples)
diarrhea_severity = np.zeros(n_samples)
abdominal_pain_severity = np.zeros(n_samples)

for i in range(n_samples):
    cluster = clusters[i]
    if cluster == 0:  # High typhoid risk
        fever_severity[i] = np.random.normal(7, 1.5)
        abdominal_pain_severity[i] = np.random.normal(5, 1.5)
        diarrhea_severity[i] = np.random.normal(2, 1)
    elif cluster == 1:  # High cholera risk
        fever_severity[i] = np.random.normal(6, 1.5)
        diarrhea_severity[i] = np.random.normal(6, 1.5)
        abdominal_pain_severity[i] = np.random.normal(2, 1)
    else:  # Low risk (No Disease)
        fever_severity[i] = np.random.normal(2, 1)
        diarrhea_severity[i] = np.random.normal(1, 0.5)
        abdominal_pain_severity[i] = np.random.normal(1, 0.5)

# Clip the entire arrays
data["fever_severity"] = np.clip(fever_severity, 0, 10)
data["diarrhea_severity"] = np.clip(diarrhea_severity, 0, 10)
data["abdominal_pain_severity"] = np.clip(abdominal_pain_severity, 0, 10)

# Disease assignment
for i in range(n_samples):
    fever = data["fever_severity"][i]
    diarrhea = data["diarrhea_severity"][i]
    abdominal_pain = data["abdominal_pain_severity"][i]
    water_source = data["water_source"][i]
    vaccination = data["vaccination_status"][i]
    travel = data["recent_travel"][i]
    hygiene = data["food_hygiene"][i]
    prevalence = data["case_prevalence"][i]

    base_risk = 0.1 + (prevalence / 30) * 2
    if water_source in ["River", "Rainwater"]:
        base_risk += 0.4
    if vaccination == "No":
        base_risk += 0.25
    if travel == "Yes":
        base_risk += 0.2
    if hygiene == "Poor":
        base_risk += 0.2
    base_risk = min(base_risk, 0.8)

    typhoid_risk = base_risk * (abdominal_pain / 10) * 1.8
    cholera_risk = base_risk * (diarrhea / 10) * 1.8
    total_risk = max(0.2, typhoid_risk + cholera_risk)
    no_disease_prob = max(0.2, 1 - total_risk)
    remaining_prob = 1 - no_disease_prob
    typhoid_prob = (typhoid_risk / total_risk) * remaining_prob
    cholera_prob = (cholera_risk / total_risk) * remaining_prob
    if typhoid_prob + cholera_prob > remaining_prob:
        scale_factor = remaining_prob / (typhoid_prob + cholera_prob)
        typhoid_prob *= scale_factor
        cholera_prob *= scale_factor

    r = np.random.random()
    if r < no_disease_prob:
        data["disease"][i] = 0
    elif r < (no_disease_prob + typhoid_prob):
        data["disease"][i] = 1
    else:
        data["disease"][i] = 2

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
le_gender = LabelEncoder()
le_water = LabelEncoder()
le_region = LabelEncoder()
le_vaccination = LabelEncoder()
le_travel = LabelEncoder()
le_food = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["water_source"] = le_water.fit_transform(df["water_source"])
df["region"] = le_region.fit_transform(df["region"])
df["vaccination_status"] = le_vaccination.fit_transform(df["vaccination_status"])
df["recent_travel"] = le_travel.fit_transform(df["recent_travel"])
df["food_hygiene"] = le_food.fit_transform(df["food_hygiene"])

# Prepare features
X = df.drop("disease", axis=1)
y = df["disease"]

scaler = StandardScaler()
numeric_cols = ["age", "fever_severity", "diarrhea_severity", "abdominal_pain_severity", "case_prevalence"]
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X["symptom_sum"] = X["fever_severity"] + X["diarrhea_severity"] + X["abdominal_pain_severity"]
X["age_risk"] = np.where(X["age"] > 50, 1, 0)

# Ensure no NaN values
X = X.fillna(0)
y = y.fillna(0)

# Validate class distribution
unique_classes = np.unique(y)
print(f"Unique classes in y: {unique_classes}")
if len(unique_classes) < 3:
    print("Warning: Fewer than 3 classes detected. Forcing inclusion of all classes.")
    for cls in [0, 1, 2]:
        if cls not in unique_classes:
            idx = np.random.choice(len(y), 500, replace=False)
            y.iloc[idx] = cls
    X = df.drop("disease", axis=1)
    y = df["disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Tune and train models
rf_param_dist = {'n_estimators': randint(100, 300), 'max_depth': [10, 15, 20], 'min_samples_split': [2, 5]}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), rf_param_dist, n_iter=10, cv=3, random_state=42)
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
print(f"Best RF params: {rf_search.best_params_}")

xgb_param_dist = {'n_estimators': randint(100, 300), 'learning_rate': uniform(0.01, 0.1), 'max_depth': [6, 8, 10]}
xgb_search = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=3), xgb_param_dist, n_iter=10, cv=3, random_state=42)
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
print(f"Best XGB params: {xgb_search.best_params_}")

# Evaluate models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)
xgb_probs = xgb_model.predict_proba(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
rf_log_loss = log_loss(y_test, rf_probs)
xgb_log_loss = log_loss(y_test, xgb_probs)
rf_report = classification_report(y_test, rf_pred, target_names=["No Disease", "Typhoid", "Cholera"])
xgb_report = classification_report(y_test, xgb_pred, target_names=["No Disease", "Typhoid", "Cholera"])
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
xgb_conf_matrix = confusion_matrix(y_test, xgb_pred)

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.2f} (+/- {rf_cv_scores.std() * 2:.2f})")
print(f"XGBoost CV Accuracy: {xgb_cv_scores.mean():.2f} (+/- {xgb_cv_scores.std() * 2:.2f})")

# Ensemble
ensemble_probs = (rf_probs + xgb_probs) / 2
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_log_loss = log_loss(y_test, ensemble_probs)
print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")
print(f"Ensemble Log Loss: {ensemble_log_loss:.2f}")

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
print(f"Random Forest Log Loss: {rf_log_loss:.2f}")
print(f"XGBoost Log Loss: {xgb_log_loss:.2f}")
print("Random Forest Classification Report:\n", rf_report)
print("XGBoost Classification Report:\n", xgb_report)
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
print("XGBoost Confusion Matrix:\n", xgb_conf_matrix)

# Save performance metrics
with open("model_performance.txt", "w") as f:
    f.write(f"Random Forest Accuracy: {rf_accuracy:.2f}\n")
    f.write(f"XGBoost Accuracy: {xgb_accuracy:.2f}\n")
    f.write(f"Random Forest Log Loss: {rf_log_loss:.2f}\n")
    f.write(f"XGBoost Log Loss: {xgb_log_loss:.2f}\n")
    f.write("Random Forest Classification Report:\n" + rf_report + "\n")
    f.write("XGBoost Classification Report:\n" + xgb_report + "\n")
    f.write("Random Forest Confusion Matrix:\n" + str(rf_conf_matrix) + "\n")
    f.write("XGBoost Confusion Matrix:\n" + str(xgb_conf_matrix) + "\n")
    f.write(f"Ensemble Accuracy: {ensemble_accuracy:.2f}\n")
    f.write(f"Ensemble Log Loss: {ensemble_log_loss:.2f}\n")

# Save models and encoders
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(le_gender, "models/le_gender.pkl")
joblib.dump(le_water, "models/le_water.pkl")
joblib.dump(le_region, "models/le_region.pkl")
joblib.dump(le_vaccination, "models/le_vaccination.pkl")
joblib.dump(le_travel, "models/le_travel.pkl")
joblib.dump(le_food, "models/le_food.pkl")

# Save training data
df.to_csv("models/tanzania_individual_health_data.csv", index=False)

print("Models trained, evaluated, and saved successfully.")