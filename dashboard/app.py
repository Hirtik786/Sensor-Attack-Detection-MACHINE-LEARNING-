import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('E:/Main/model/final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict attack or normal behavior
def predict_attack(data):
    # Make prediction using the loaded model
    prediction = model.predict(data)
    return prediction

# Function to display the uploaded sensor data and prediction
def show_predictions(data):
    st.write("### Sensor Data Preview:")
    st.dataframe(data.head())

    # Making prediction on the uploaded data
    predictions = predict_attack(data)

    # Displaying predictions
    data['Prediction'] = predictions
    st.write("### Predictions (1 = Attack, 0 = Normal):")
    st.dataframe(data)

# App title and description
st.title('Sensor Attack Detection Dashboard')
st.write("""
    This dashboard detects malicious sensor data based on machine learning models.
    Upload your sensor data and the model will predict whether each sensor reading
    is normal or an attack.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with sensor data", type=["csv"])

# If a file is uploaded, read and process the data
if uploaded_file is not None:
    # Load the CSV data into a pandas DataFrame
    sensor_data = pd.read_csv(uploaded_file)

    # Show preview of the data
    show_predictions(sensor_data)

    # You can also add additional visualizations here (e.g., feature importance, etc.)

else:
    st.warning("Please upload a CSV file containing sensor data to get started.")
