import streamlit as st
import joblib
import numpy as np
import pandas as pd


data = pd.read_csv("wine quality.csv")

svm_model = joblib.load('svm_model.pkl')


st.title("Wine Quality Prediction")

st.write("""
This app predicts the quality of wine based on various features like acidity, sugar, and alcohol content.
Please provide the required input values below to predict the quality of wine.
""")


fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.1)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, step=0.1)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, step=0.01)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, step=1.0)  
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=200.0, step=1.0)  
density = st.number_input("Density", min_value=0.9, max_value=1.2, step=0.001)
pH = st.number_input("pH", min_value=2.0, max_value=4.0, step=0.1)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.1)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, step=0.1)


input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])


if st.button("Predict Wine Quality"):
    prediction = svm_model.predict(input_data)  
    st.write(f"Predicted Wine Quality: {prediction[0]:.2f}")