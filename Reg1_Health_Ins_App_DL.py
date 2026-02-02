import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Health Insurance Prediction (DL)", layout="centered")

MODEL_PATH = "src/dl_model.h5"
PREPROCESSOR_PATH = "src/preprocessor.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found in app directory")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"{PREPROCESSOR_PATH} not found in app directory")

    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

st.title("Health Insurance Premium Prediction (Deep Learning)")
st.write("Enter the details below to estimate the insurance charges using a Deep Learning model.")

try:
    dl_model, preprocessor = load_artifacts()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# User Inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.slider("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

if st.button("Predict"):
    try:
        # Preprocess input
        X_processed = preprocessor.transform(input_data)

        # Deep Learning prediction
        prediction = dl_model.predict(X_processed, verbose=0).flatten()[0]

        st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
