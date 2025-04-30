import pickle
import streamlit as st
import numpy as np

# Function to load Random Forest model
def load_rf_model():
    with open('random_forest_model_iris.pkl', 'rb') as f:
        return pickle.load(f)

# Function to load XGBoost model
def load_xgboost_model():
    with open('xgboost_model_iris.pkl', 'rb') as f:
        return pickle.load(f)

# Iris flower species descriptions
species_descriptions = {
    'Setosa': 'Iris Setosa is characterized by its small flowers with wide petals. It is one of the earliest blooming Iris species.',
    'Versicolor': 'Iris Versicolor is a medium-sized species with slightly larger flowers. Its petals are more narrow than Setosa.',
    'Virginica': 'Iris Virginica is a large species with elegant flowers that have broad petals, often found in wetlands.'
}

# Streamlit inputs
st.title("Iris Flower Prediction App")

# Select model for prediction
model_option = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

# Input fields for Iris flower attributes
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Button and model prediction logic
if model_option == "Random Forest":
    if st.button('Predict with Random Forest'):
        if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
            st.error("Please provide non-zero values for all inputs!")
        else:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            model = load_rf_model()  # Load Random Forest model
            prediction = model.predict(input_data)
            species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            predicted_species = species_map[prediction[0]]  # map numeric to species
            st.write(f"The predicted Iris flower species is: {predicted_species}")
            st.write(f"Description: {species_descriptions[predicted_species]}")

elif model_option == "XGBoost":
    if st.button('Predict with XGBoost'):
        if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
            st.error("Please provide non-zero values for all inputs!")
        else:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            model = load_xgboost_model()  # Load XGBoost model
            prediction = model.predict(input_data)
            species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            predicted_species = species_map[prediction[0]]  # map numeric to species
            st.write(f"The predicted Iris flower species is: {predicted_species}")
            st.write(f"Description: {species_descriptions[predicted_species]}")
