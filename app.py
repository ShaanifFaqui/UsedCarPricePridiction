from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for encoders and model
label_encoders = {}
model = None

def load_model_and_encoders():
    """Load the trained model and label encoders"""
    global model, label_encoders
    
    try:
        # Load the trained model
        model = joblib.load('models/random_forest_model.joblib')
        
        # Load label encoders
        label_encoders = joblib.load('models/label_encoders.joblib')
        
        logger.info("Model and encoders loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([data])
        
        # Feature engineering
        df['Company'] = df['Name'].str.split().str[0]
        
        # Extract numeric values from string columns
        df['Mileage(km/kg)'] = df['Mileage'].str.split().str[0].astype(float)
        df['Engine(CC)'] = df['Engine'].str.split().str[0].astype(float)
        df['Power(bhp)'] = df['Power'].str.split().str[0].astype(float)
        
        # Handle New_Price if provided
        if 'New_Price' in df.columns and not pd.isna(df['New_Price'].iloc[0]):
            df['New_car_Price'] = df['New_Price'].str.split().str[0].astype(float)
        else:
            df['New_car_Price'] = np.nan
        
        # Encode categorical variables
        categorical_columns = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Company']
        
        for col in categorical_columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        # Create dummy variables for Location, Fuel_Type, and Transmission
        location_dummies = pd.get_dummies(df['Location'], prefix='Location')
        fuel_dummies = pd.get_dummies(df['Fuel_Type'], prefix='Fuel_Type')
        transmission_dummies = pd.get_dummies(df['Transmission'], prefix='Transmission')
        
        # Combine all features
        final_features = pd.concat([
            df[['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats', 
                'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)']],
            location_dummies,
            fuel_dummies,
            transmission_dummies
        ], axis=1)
        
        # Ensure all expected columns are present
        expected_columns = [
            'Year', 'Kilometers_Driven', 'Owner_Type', 'Seats',
            'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)',
            'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore',
            'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur',
            'Location_Kochi', 'Location_Kolkata', 'Location_Mumbai',
            'Location_Pune', 'Fuel_Type_Diesel', 'Fuel_Type_LPG',
            'Fuel_Type_Petrol', 'Transmission_Manual'
        ]
        
        for col in expected_columns:
            if col not in final_features.columns:
                final_features[col] = 0
        
        # Reorder columns to match training data
        final_features = final_features[expected_columns]
        
        return final_features
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for car price prediction"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Name', 'Location', 'Year', 'Kilometers_Driven', 
                          'Fuel_Type', 'Transmission', 'Owner_Type', 
                          'Mileage', 'Engine', 'Power', 'Seats']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Return prediction
        return jsonify({
            'predicted_price': float(prediction),
            'predicted_price_formatted': f"₹{prediction:,.2f}",
            'input_data': data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': len(label_encoders) > 0
    })

@app.route('/api/features')
def get_features():
    """Get available features and their options"""
    return jsonify({
        'locations': ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 
                     'Jaipur', 'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'],
        'fuel_types': ['CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'],
        'transmissions': ['Manual', 'Automatic'],
        'owner_types': ['First', 'Second', 'Third', 'Fourth & Above'],
        'year_range': {'min': 1998, 'max': 2019},
        'seats_range': {'min': 2, 'max': 10}
    })

if __name__ == '__main__':
    # Load model on startup
    if not os.path.exists('models'):
        os.makedirs('models')
        logger.warning("Models directory created. Please train and save your model first.")
    
    if load_model_and_encoders():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Please ensure model files exist.")
        app.run(debug=True, host='0.0.0.0', port=5000) 