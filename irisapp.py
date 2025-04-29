import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Streamlit app layout and title
st.title("Iris Flower Prediction App")
st.write("This app uses Random Forest and XGBoost models to predict the species of Iris flowers.")

# Load the Iris dataset
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

# Load the models
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Loading data and models only when needed
X, y, feature_names, target_names = load_iris_data()

# Preprocessing - Create scaler for feature scaling
scaler = StandardScaler()

# Button to trigger Random Forest prediction
if st.button('Predict with Random Forest'):
    # Load Random Forest Model (cache it)
    rf_model = load_model('random_forest_model_iris.pkl')

    # Get user inputs
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

    # Check if all inputs are filled
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale input data using the same scaler used in training
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction using Random Forest model
        try:
            prediction = rf_model.predict(input_data_scaled)
            species = target_names[prediction][0]
            st.write(f"Prediction: {species}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Button to trigger XGBoost prediction
if st.button('Predict with XGBoost'):
    # Load XGBoost Model (cache it)
    xgb_model = load_model('xgboost_model_iris.pkl')

    # Get user inputs
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

    # Check if all inputs are filled
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale input data using the same scaler used in training
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction using XGBoost model
        try:
            prediction = xgb_model.predict(input_data_scaled)
            species = target_names[prediction][0]
            st.write(f"Prediction: {species}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
