import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the Decision Tree model
@st.cache_resource
def load_dt_model():
    model_path = 'decision_tree_model_iris6.pkl'  # Ensure the correct path
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload the model file.")
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Define a dictionary to map integer labels to Iris species names
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Adding descriptions for Iris species
species_descriptions = {
    'Setosa': 'Iris Setosa is characterized by its small flowers with wide petals. It is one of the earliest blooming Iris species.',
    'Versicolor': 'Iris Versicolor is a medium-sized species with slightly larger flowers. Its petals are more narrow than Setosa.',
    'Virginica': 'Iris Virginica is a large species with elegant flowers that have broad petals, often found in wetlands.'
}

# Streamlit inputs for flower features
st.title("Iris Flower Prediction App")

# Input fields for the user (no min/max for now)
sepal_length = st.number_input('Sepal Length (cm)', step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', step=0.1)
petal_length = st.number_input('Petal Length (cm)', step=0.1)
petal_width = st.number_input('Petal Width (cm)', step=0.1)

# Predict with Decision Tree when the button is clicked
if st.button('Predict with Decision Tree'):
    # Ensure that inputs are non-zero
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Prepare the input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the input data to match the model's training process
        scaler = StandardScaler()  # We use StandardScaler for feature scaling
        input_data_scaled = scaler.fit_transform(input_data)

        # Load the model and make predictions
        model = load_dt_model()
        
        if model is not None:
            prediction = model.predict(input_data_scaled)
            predicted_species = species_map[prediction[0]]
            st.write(f"The predicted Iris flower species is: {predicted_species}")
            st.write(f"Description: {species_descriptions[predicted_species]}")
