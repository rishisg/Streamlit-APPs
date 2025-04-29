# 6. Streamlit App Development
# Now, let's build a simple Streamlit app that will load the saved models and predict the Iris species based on user input.
# Streamlit App Code

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset (for example purposes)
iris_data = load_iris()
df = iris_data.data
target = iris_data.target

# Convert to DataFrame for easier handling
import pandas as pd
df = pd.DataFrame(df, columns=iris_data.feature_names)
df['target'] = target

# Feature scaling (Standardization)
scaler = StandardScaler()

# Split the data into features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train XGBoost model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Streamlit Interface
st.title("Iris Flower Prediction App")
st.write("This app uses Random Forest and XGBoost models to predict the species of Iris flowers.")

# Input fields for user data (for Iris dataset)
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Preprocess the input data (match training data feature set)
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Feature scaling (same as training)
input_data_scaled = scaler.transform(input_data)  # Make sure scaler was fitted with the same features during training

# Make prediction
if st.button('Predict with Random Forest'):
    prediction = rf_model.predict(input_data_scaled)
    species = iris_data.target_names[prediction][0]  # Map prediction to species name
    st.write(f"Prediction: {species}")

if st.button('Predict with XGBoost'):
    prediction = xgb_model.predict(input_data_scaled)
    species = iris_data.target_names[prediction][0]  # Map prediction to species name
    st.write(f"Prediction: {species}")

# Evaluation (display metrics when the models are trained)
st.write("Model Evaluation on Test Data:")

# Scale the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
st.write("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
st.write("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
st.write("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Predict and evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
st.write("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
st.write("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
st.write("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))