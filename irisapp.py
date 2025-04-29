import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Streamlit app layout and title
st.title("Iris Flower Prediction App")
st.write("This app uses a Random Forest model to predict the species of Iris flowers.")

# Load the Random Forest model (Ensure it's saved and available in the same directory as this app)
@st.cache_resource
def load_model():
    with open('random_forest_model_iris.pkl', 'rb') as f:
        return pickle.load(f)

# Initialize scaler
scaler = StandardScaler()

# Initialize session state variables for inputs if they don't exist
if 'sepal_length' not in st.session_state:
    st.session_state.sepal_length = 0.0
if 'sepal_width' not in st.session_state:
    st.session_state.sepal_width = 0.0
if 'petal_length' not in st.session_state:
    st.session_state.petal_length = 0.0
if 'petal_width' not in st.session_state:
    st.session_state.petal_width = 0.0

# Prediction function for Random Forest
def predict_rf():
    # Get the input values from the user
    st.session_state.sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_length)
    st.session_state.sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_width)
    st.session_state.petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_length)
    st.session_state.petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_width)

    # Ensure the input fields are filled with non-zero values
    if st.session_state.sepal_length == 0.0 or st.session_state.sepal_width == 0.0 or st.session_state.petal_length == 0.0 or st.session_state.petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Load the Random Forest model (this should be cached)
        model = load_model()

        # Prepare the input data
        input_data = np.array([[st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width]])

        # Scale the input data using StandardScaler (same scaling as used in training)
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction using Random Forest model
        prediction = model.predict(input_data_scaled)

        # Display the prediction
        st.write(f"Predicted Iris Species: {prediction[0]}")

# Button to trigger prediction
if st.button('Predict with Random Forest'):
    predict_rf()
