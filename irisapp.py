import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Streamlit app layout and title
st.title("Iris Flower Prediction App")
st.write("This app uses a Random Forest model to predict the species of Iris flowers.")

# Load the Random Forest model (Ensure it's saved and available in the same directory as this app)
@st.cache_resource
def load_rf_model():
    with open('random_forest_model_iris1.pkl', 'rb') as f:
        return pickle.load(f)

# Preprocessing - Create scaler for feature scaling
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

# Display input fields to the user
st.session_state.sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_length)
st.session_state.sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.sepal_width)
st.session_state.petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_length)
st.session_state.petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1, value=st.session_state.petal_width)

# Ensure all input fields are filled with non-zero values
if st.session_state.sepal_length == 0.0 or st.session_state.sepal_width == 0.0 or st.session_state.petal_length == 0.0 or st.session_state.petal_width == 0.0:
    st.error("Please provide non-zero values for all inputs!")
else:
    # Show the prediction button only if inputs are valid (non-zero)
    if st.button('Predict with Random Forest'):
        try:
            # Load the Random Forest model (this should be cached)
            rf_model = load_rf_model()

            # Prepare the input data
            input_data = np.array([[st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width]])

            # Check for any missing values in the input data
            if np.any(np.isnan(input_data)):
                st.error("Input data contains missing values!")
            else:
                # Scale the input data using the same scaler used in training
                input_data_scaled = scaler.fit_transform(input_data)

                # Make prediction using Random Forest model
                prediction = rf_model.predict(input_data_scaled)

                # Display the prediction
                st.write(f"Prediction with Random Forest: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
