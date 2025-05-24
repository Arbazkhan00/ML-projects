import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure the model is in the correct directory.")
    st.stop()

# Feature names and default values
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
    'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension'
]

default_values = [
    14.0, 20.0, 90.0, 600.0, 0.1, 0.2, 0.3, 0.2, 0.2, 0.06,
    0.3, 1.0, 2.0, 50.0, 0.005, 0.02, 0.03, 0.02, 0.02, 0.007,
    16.0, 25.0, 100.0, 800.0, 0.12, 0.25, 0.35, 0.25, 0.25, 0.08
]

# Streamlit configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .header {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #34495e;
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="header">🩺 Breast Cancer Prediction App</div>', unsafe_allow_html=True)
st.markdown("""
This application uses a machine learning model to predict whether a breast cancer tumor is **Malignant** or **Benign** based on input features.

Enter the tumor characteristics in the sidebar and click **Predict** to see the results.
""")

# Sidebar for input
st.sidebar.header("Input Tumor Characteristics")
st.sidebar.markdown('<div class="subheader">Enter the values for each feature below:</div>', unsafe_allow_html=True)

# Input fields
st.sidebar.markdown("### Feature Inputs")
col1, col2 = st.sidebar.columns(2)
input_data = []

for i, (feature, default) in enumerate(zip(feature_names, default_values)):
    if i % 2 == 0:
        with col1:
            value = st.number_input(f"{feature}", value=default, step=0.01, format="%.4f", key=feature)
    else:
        with col2:
            value = st.number_input(f"{feature}", value=default, step=0.01, format="%.4f", key=feature)
    input_data.append(value)

# Prediction
if st.sidebar.button("Predict", key="predict_button"):
    with st.spinner("Making prediction..."):
        input_array = np.array(input_data).reshape(1, -1)
        try:
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)

            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Prediction Results")

            result = "Malignant" if prediction[0] == 0 else "Benign"
            st.markdown(f"**Diagnosis**: {result}")
            st.markdown(f"**Probability (Benign)**: {prediction_proba[0][1]:.2%}")
            st.markdown(f"**Probability (Malignant)**: {prediction_proba[0][0]:.2%}")

            if result == "Malignant":
                st.warning("The model predicts a malignant tumor. Please consult a medical professional for further evaluation.")
            else:
                st.success("The model predicts a benign tumor. Please consult a medical professional for confirmation.")

            st.markdown('</div>', unsafe_allow_html=True)

            # Optional: Display a probability bar chart
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Diagnosis': ['Benign', 'Malignant'],
                'Probability': [prediction_proba[0][1], prediction_proba[0][0]]
            })
            st.bar_chart(prob_df.set_index('Diagnosis'))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
