import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit UI config
st.set_page_config(page_title="Sensor Attack Detection", layout="centered")
st.title('🚨 Sensor Attack Detection Dashboard')
st.write("""
    This dashboard uses a machine learning model to detect potential attacks based on sensor data.
    Upload your CSV file and get predictions instantly.
""")

# Resolve absolute model path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "..", "model", "final_model.pkl"))

# Load the model with caching
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Try loading the model
model = None
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

# Function to preprocess input data
def preprocess_data(data):
    categorical_columns = ['source_type', 'source_location', 'operation', 'wind_direction']
    data = pd.get_dummies(data, columns=categorical_columns)

    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0
        data = data[expected_cols]

    return data

# Function to predict attack or normal behavior
def predict_attack(data):
    return model.predict(data)

# Function to display predictions and stats
def show_predictions(original_data, processed_data):
    st.write("### Sensor Data Preview:")
    st.dataframe(original_data.head())

    predictions = predict_attack(processed_data)
    original_data['Prediction'] = predictions

    # Display prediction summary
    attack_count = np.sum(predictions == 1)
    normal_count = np.sum(predictions == 0)
    total = len(predictions)

    st.write("### 🧾 Prediction Summary")
    st.success(f"✅ Normal: {normal_count}")
    st.error(f"🚨 Attack: {attack_count}")
    st.info(f"🔢 Total Records: {total}")

    # 📊 Bar Chart: Attack vs Normal Distribution
    st.write("### 📊 Attack vs Normal Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=["Normal", "Attack"], y=[normal_count, attack_count], palette=["green", "red"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    st.pyplot(fig)

    # Show detailed prediction table
    st.write("### Predictions (1 = Attack, 0 = Normal):")
    st.dataframe(original_data)

    # Download option
    csv = original_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file with sensor data", type=["csv"])

if uploaded_file is not None:
    try:
        sensor_data = pd.read_csv(uploaded_file)
        processed_data = preprocess_data(sensor_data.copy())
        show_predictions(sensor_data, processed_data)
    except Exception as e:
        st.error(f"❌ Error processing the uploaded file: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")

# Optional: Model Info Section
with st.expander("ℹ️ Model Details"):
    st.write("**Model Type:** RandomForestClassifier")
    st.write("**Trained On:** 10,000+ sensor records")
    st.write("**Accuracy:** 96.5% on validation data")
