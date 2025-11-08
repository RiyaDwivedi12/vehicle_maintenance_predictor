import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("decision_tree_maintenance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚗 Vehicle Maintenance Prediction")

# Input Fields (same as your original UI)
engine_temp = st.number_input("Engine Temperature (°C)", min_value=0, max_value=200, value=90)
oil_level = st.number_input("Oil Level", min_value=0, max_value=10, value=5)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=100000, value=15000)
brake_condition = st.number_input("Brake Condition (0=Bad, 1=Good)", min_value=0, max_value=1, value=1)

# Put input into DataFrame
input_data = pd.DataFrame({
    'engine_temp': [engine_temp],
    'oil_level': [oil_level],
    'mileage': [mileage],
    'brake_condition': [brake_condition]
})

# ✅ FIX: Match model’s expected features
try:
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Create missing feature with default value
    
    input_data = input_data[expected_features]  # Reorder to match training
except:
    # If model doesn't have feature_names_in_ attribute
    pass

# Predict
if st.button("Predict Maintenance"):
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.warning("⚠️ Maintenance Required!")
        else:
            st.success("✅ No Maintenance Needed")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
