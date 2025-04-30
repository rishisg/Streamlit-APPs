import pickle
import streamlit as st
import numpy as np

# Load the model (Random Forest in this case)
@st.cache_resource
def load_model():
    with open('random_forest_model_iris3.pkl', 'rb') as f:
        return pickle.load(f)

# Iris flower species descriptions
species_descriptions = {
    'Setosa': 'Iris Setosa is characterized by its small flowers with wide petals. It is one of the earliest blooming Iris species.',
    'Versicolor': 'Iris Versicolor is a medium-sized species with slightly larger flowers. Its petals are more narrow than Setosa.',
    'Virginica': 'Iris Virginica is a large species with elegant flowers that have broad petals, often found in wetlands.'
}

# Streamlit inputs
st.title("Iris Flower Prediction App")

# Input fields
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Model prediction on button click
if st.button('Predict with Random Forest'):
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Prepare the input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Load the model and make a prediction
        model = load_model()
        prediction = model.predict(input_data)

        # Mapping the numeric prediction to the species name
        species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species_map[prediction[0]]  # map numeric to species

        # Show the prediction and description
        st.write(f"The predicted Iris flower species is: {predicted_species}")
        st.write(f"Description: {species_descriptions[predicted_species]}")
