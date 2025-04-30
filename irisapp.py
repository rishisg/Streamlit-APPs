import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the models
@st.cache_resource
def load_rf_model():
    with open('random_forest_model_iris.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_xgb_model():
    with open('xgboost_model_iris.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Iris Flower Prediction App")

# Input fields for user to provide feature values
sepal_length = st.number_input('Sepal Length (cm)', step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', step=0.1)
petal_length = st.number_input('Petal Length (cm)', step=0.1)
petal_width = st.number_input('Petal Width (cm)', step=0.1)

# When Predict with Random Forest is clicked
if st.button('Predict with Random Forest'):
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Prepare input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Load the model and make predictions
        rf_model = load_rf_model()
        prediction = rf_model.predict(input_data_scaled)
        
        # Display results
        st.write(f"The predicted Iris flower species is: {prediction[0]}")
        
        # Add species descriptions
        species_descriptions = {
            'Setosa': 'Iris Setosa is characterized by its small flowers with wide petals. It is one of the earliest blooming Iris species.',
            'Versicolor': 'Iris Versicolor is a medium-sized species with slightly larger flowers. Its petals are more narrow than Setosa.',
            'Virginica': 'Iris Virginica is a large species with elegant flowers that have broad petals, often found in wetlands.'
        }
        
        st.write(f"Description: {species_descriptions[prediction[0]]}")

# When Predict with XGBoost is clicked
if st.button('Predict with XGBoost'):
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Prepare input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Load the model and make predictions
        xgb_model = load_xgb_model()
        prediction = xgb_model.predict(input_data_scaled)
        
        # Display results
        st.write(f"The predicted Iris flower species is: {prediction[0]}")
        
        # Add species descriptions
        species_descriptions = {
            'Setosa': 'Iris Setosa is characterized by its small flowers with wide petals. It is one of the earliest blooming Iris species.',
            'Versicolor': 'Iris Versicolor is a medium-sized species with slightly larger flowers. Its petals are more narrow than Setosa.',
            'Virginica': 'Iris Virginica is a large species with elegant flowers that have broad petals, often found in wetlands.'
        }
        
        st.write(f"Description: {species_descriptions[prediction[0]]}")
