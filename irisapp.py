import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Streamlit app layout and title
st.title("Iris Flower Prediction App")
st.write("This app uses a Random Forest model to predict the species of Iris flowers.")

# Load the Iris dataset
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

# Load the Random Forest model
@st.cache_resource
def load_rf_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Loading data only when needed
X, y, feature_names, target_names = load_iris_data()

# Preprocessing - Create scaler for feature scaling
scaler = StandardScaler()

# Initialize session state variables if not already initialized
if 'sepal_length' not in st.session_state:
    st.session_state.sepal_length = 0.0
if 'sepal_width' not in st.session_state:
    st.session_state.sepal_width = 0.0
if 'petal_length' not in st.session_state:
    st.session_state.petal_length = 0.0
if 'petal_width' not in st.session_state:
    st.session_state.petal_width = 0.0
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Function to handle Random Forest input and prediction
def handle_input_rf():
    # User input fields
    st.session_state.sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_length)
    st.session_state.sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_width)
    st.session_state.petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_length)
    st.session_state.petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_width)

    # Ensure the input fields are filled with non-zero values
    if st.session_state.sepal_length == 0.0 or st.session_state.sepal_width == 0.0 or st.session_state.petal_length == 0.0 or st.session_state.petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Load Random Forest Model (cache it)
        rf_model = load_rf_model('random_forest_model_iris.pkl')

        # Prepare the input data
        input_data = np.array([[st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width]])

        # Scale input data using the same scaler used in training
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction using Random Forest model
        prediction = rf_model.predict(input_data_scaled)
        st.session_state.prediction = target_names[prediction][0]
        st.write(f"Prediction with Random Forest: {st.session_state.prediction}")

# Display the button for Random Forest prediction
st.write("Please select a model for prediction:")

# Predict with Random Forest when button is clicked
if st.button('Predict with Random Forest'):
    handle_input_rf()

# Display prediction after button click
if st.session_state.prediction:
    st.write(f"Prediction: {st.session_state.prediction}")
