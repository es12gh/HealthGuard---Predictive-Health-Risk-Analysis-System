import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import io

# -------------------------------
# Load datasets
# -------------------------------
heart = pd.read_csv("HeartDiseaseTrain-Test.csv")
diabetes = pd.read_csv("diabetes.csv")

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns, scaler, X

# Prepare datasets
X_diabetes, y_diabetes, diabetes_cols, scaler_diabetes, X_diabetes_raw = preprocess_data(diabetes, "Outcome")
X_heart, y_heart, heart_cols, scaler_heart, X_heart_raw = preprocess_data(heart, "target")

# Train Random Forest models
rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_diabetes.fit(X_diabetes, y_diabetes)

rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
rf_heart.fit(X_heart, y_heart)

# -------------------------------
# Feature definitions and tooltips
# -------------------------------
heart_numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
heart_categorical_features = {
    'sex': [0, 1],
    'cp': [0, 1, 2, 3],
    'fbs': [0, 1],
    'restecg': [0, 1, 2],
    'exang': [0, 1],
    'slope': [0, 1, 2],
    'thal': [0, 1, 2, 3]
}

feature_tooltips = {
    'Pregnancies': 'Number of pregnancies',
    'Glucose': 'Plasma glucose concentration (2h OGTT)',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skinfold thickness (mm)',
    'Insulin': '2-Hour serum insulin (mu U/ml)',
    'BMI': 'Body mass index (kg/m^2)',
    'DiabetesPedigreeFunction': 'Diabetes pedigree function',
    'Age': 'Age in years',
    'age': 'Age in years',
    'sex': 'Sex (0=female, 1=male)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar >120mg/dl',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Max heart rate achieved',
    'exang': 'Exercise induced angina (0=no, 1=yes)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment (0-2)',
    'ca': 'Major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (0-3)'
}

# -------------------------------
# Streamlit app setup
# -------------------------------
st.set_page_config(page_title="🩺 HealthGuard App", layout="wide")
st.title("🩺 HealthGuard - Predictive Health Risk Analysis System")
st.write("Predict **Diabetes** and/or **Heart Disease** with risk levels, suggestions, and feature insights.")

# -------------------------------
# Functions
# -------------------------------
def probability_color(prob):
    if prob < 0.4: return '🟢 Low risk', '#d4edda'
    elif prob < 0.7: return '🟡 Moderate risk', '#fff3cd'
    else: return '🔴 High risk', '#f8d7da'

def risk_suggestion(prob, disease_name):
    if prob < 0.4: return f"Low risk for {disease_name}. Maintain a healthy lifestyle and regular check-ups."
    elif prob < 0.7: return f"Moderate risk for {disease_name}. Consider a medical check-up and monitor your health closely."
    else: return f"High risk for {disease_name}. Consult a doctor promptly for further evaluation."

def predict(model, scaler, feature_names, input_data):
    available_features = [f for f in feature_names if f in input_data]
    input_df = pd.DataFrame([input_data])[available_features]
    missing_features = [f for f in feature_names if f not in available_features]
    for f in missing_features: input_df[f] = 0
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability

def plot_feature_importance(model, feature_names, disease_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8,5))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(f"{disease_name} - Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

def create_input_features(df_raw, cols):
    user_input = {}
    for feature in cols:
        if feature in df_raw.columns:
            min_val, max_val = float(df_raw[feature].min()), float(df_raw[feature].max())
            default_val = float((min_val+max_val)/2)
            tooltip = feature_tooltips.get(feature,"")
            if df_raw[feature].dtype in [int, np.int64]:
                user_input[feature] = st.sidebar.slider(feature, int(min_val), int(max_val), int(default_val), help=tooltip)
            else:
                user_input[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(default_val), step=0.01, help=tooltip)
    return user_input

def display_prediction_card(disease, pred, prob):
    risk_label, bg_color = probability_color(prob)
    suggestion = risk_suggestion(prob, disease)
    st.markdown(
        f"<div style='background-color:{bg_color};padding:15px;border-radius:5px'>"
        f"<h3>{disease} Prediction: {'Yes' if pred==1 else 'No'}</h3>"
        f"<h4>Probability: {prob:.2f} → {risk_label}</h4>"
        f"<p>{suggestion}</p>"
        f"</div>", unsafe_allow_html=True)
    return suggestion

# -------------------------------
# PDF Generation - fixed for Unicode
# -------------------------------
def generate_pdf(predictions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"HealthGuard Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for disease, info in predictions.items():
        pdf.set_font("Arial","B",14)
        pdf.cell(0,10,disease, ln=True)
        pdf.set_font("Arial","",12)
        pdf.cell(0,10,f"Prediction: {'Yes' if info['pred']==1 else 'No'}", ln=True)
        pdf.cell(0,10,f"Probability: {info['prob']:.2f}", ln=True)
        suggestion_text = info['suggestion']
        pdf.multi_cell(0,10,f"Suggestion: {suggestion_text}")
        pdf.ln(5)

    # Instead of writing to BytesIO, use 'S' to get PDF as string of bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes


# -------------------------------
# Mode Selection
# -------------------------------
mode = st.radio("Select Mode", ["Single Disease","Compare Diseases"])
predictions = {}

if mode=="Single Disease":
    disease = st.selectbox("Select Disease", ["Diabetes","Heart Disease"])
    st.sidebar.header(f"{disease} Input Features")
    if disease=="Diabetes":
        user_input = create_input_features(X_diabetes_raw, diabetes_cols)
        model, scaler, feature_names = rf_diabetes, scaler_diabetes, diabetes_cols
    else:
        user_input = {}
        for feature in heart_numeric_features:
            if feature in X_heart_raw.columns:
                min_val = int(X_heart_raw[feature].min())
                max_val = int(X_heart_raw[feature].max())
                default_val = (min_val+max_val)//2
                user_input[feature] = st.sidebar.slider(feature, min_val,max_val,default_val, help=feature_tooltips.get(feature,""))
        for feature, options in heart_categorical_features.items():
            if feature in X_heart_raw.columns:
                user_input[feature] = st.sidebar.selectbox(feature, options, help=feature_tooltips.get(feature,""))
        model, scaler, feature_names = rf_heart, scaler_heart, heart_cols

    pred, prob = predict(model, scaler, feature_names, user_input)
    suggestion = display_prediction_card(disease, pred, prob)
    plot_feature_importance(model, feature_names, disease)
    predictions[disease] = {"pred":pred, "prob":prob, "suggestion":suggestion}

else:
    st.sidebar.header("Diabetes Input Features")
    diabetes_input = create_input_features(X_diabetes_raw, diabetes_cols)
    st.sidebar.header("Heart Disease Input Features")
    heart_input = {}
    for feature in heart_numeric_features:
        if feature in X_heart_raw.columns:
            min_val = int(X_heart_raw[feature].min())
            max_val = int(X_heart_raw[feature].max())
            default_val = (min_val+max_val)//2
            heart_input[feature] = st.sidebar.slider(f"Heart: {feature}", min_val,max_val,default_val, help=feature_tooltips.get(feature,""))
    for feature, options in heart_categorical_features.items():
        if feature in X_heart_raw.columns:
            heart_input[feature] = st.sidebar.selectbox(f"Heart: {feature}", options, help=feature_tooltips.get(feature,""))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("### Diabetes Prediction")
        pred_d, prob_d = predict(rf_diabetes, scaler_diabetes, diabetes_cols, diabetes_input)
        suggestion_d = display_prediction_card("Diabetes", pred_d, prob_d)
        plot_feature_importance(rf_diabetes, diabetes_cols, "Diabetes")
        predictions["Diabetes"] = {"pred":pred_d, "prob":prob_d, "suggestion":suggestion_d}
    with col2:
        st.subheader("### Heart Disease Prediction")
        pred_h, prob_h = predict(rf_heart, scaler_heart, heart_cols, heart_input)
        suggestion_h = display_prediction_card("Heart Disease", pred_h, prob_h)
        plot_feature_importance(rf_heart, heart_cols, "Heart Disease")
        predictions["Heart Disease"] = {"pred":pred_h, "prob":prob_h, "suggestion":suggestion_h}

# -------------------------------
# Download PDF
# -------------------------------
st.subheader("Download Report")
pdf_file = generate_pdf(predictions)
st.download_button("📄 Download PDF Report", data=pdf_file, file_name="HealthGuard_Report.pdf", mime="application/pdf")

st.write("---")
st.write("Built with ❤️ using Streamlit & Random Forest")
