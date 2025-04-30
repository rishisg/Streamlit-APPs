import pickle
import numpy as np
import streamlit as st

# Load the Decision Tree model
@st.cache_resource
def load_dt_model():
    model_path = 'decision_tree_model_iris5.pkl'
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload the model file.")
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Define a dictionary to map integer labels to Iris species names
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Streamlit inputs for flower features
st.title("Iris Flower Prediction App")

# Input fields for the user
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Predict with Decision Tree when the button is clicked
if st.button('Predict with Decision Tree'):
    # Check if all inputs are non-zero
    if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
        st.error("Please provide non-zero values for all inputs!")
    else:
        # Prepare the input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Load the model and make a prediction
        model = load_dt_model()
        
        if model is not None:
            prediction = model.predict(input_data)

            # Display the result with the species name
            st.write(f"The predicted Iris flower species is: {species_map[prediction[0]]}")
