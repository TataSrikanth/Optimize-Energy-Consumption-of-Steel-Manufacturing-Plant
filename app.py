# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the SVMModel class to ensure it's available
class SVMModel:
    def __init__(self):
        self.model = SVC(random_state=42)  # SVM classifier
        self.scaler = StandardScaler()  # Scaler for features
        self.label_encoder = LabelEncoder()  # Label encoder for target variable

    def fit(self, X, y):
        # Fit the scaler and transform the features
        X_scaled = self.scaler.fit_transform(X)
        # Encode the target variable
        y_encoded = self.label_encoder.fit_transform(y)
        # Fit the SVM model
        self.model.fit(X_scaled, y_encoded)

    def predict(self, X):
        # Scale the features and make predictions
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Decode predictions
        return self.label_encoder.inverse_transform(predictions)

# Load the model
svm_model = joblib.load('svm_model.pkl')

# Streamlit Interface
st.title("SVM Classification Model")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the CSV into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # Ensure the uploaded data has the correct features
    features = ['ENERGY (Energy Consumption)', 'Production (MT)', 'TT_TIME (Total Cycle Time Including Breakdown)']
    if all(feature in input_data.columns for feature in features):
        # Extract features from the uploaded data
        X_new = input_data[features].dropna()  # Drop rows with missing values
        
        # Make predictions
        predictions = svm_model.predict(X_new)
        
        # Add predictions to the original DataFrame
        input_data['Predictions'] = predictions
        
        # Display the updated dataset with predictions
        st.write("Updated Data with Predictions:")
        st.dataframe(input_data)
        
        # Option to download the updated data as CSV
        csv = input_data.to_csv(index=False)
        st.download_button(label="Download Predictions CSV", data=csv, file_name="predictions_with_all_columns.csv", mime="text/csv")
    else:
        st.error("Uploaded file does not contain the required features.")
