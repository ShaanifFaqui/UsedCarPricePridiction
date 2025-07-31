import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model():
    """Train the model and save it for the Flask application"""
    
    print("Loading training data...")
    # Load the training data
    train_data = pd.read_csv('./train-data.csv')
    
    print("Data preprocessing...")
    # Data preprocessing (same as in notebook)
    train_data = train_data[train_data['Mileage'].notna()]
    train_data = train_data[train_data['Engine'].notna()]
    train_data = train_data[train_data['Power'].notna()]
    train_data = train_data[train_data['Seats'].notna()]
    train_data = train_data.reset_index(drop=True)
    
    # Feature engineering
    for i in range(train_data.shape[0]):
        train_data.at[i, 'Company'] = train_data['Name'][i].split()[0]
        train_data.at[i, 'Mileage(km/kg)'] = train_data['Mileage'][i].split()[0]
        train_data.at[i, 'Engine(CC)'] = train_data['Engine'][i].split()[0]
        train_data.at[i, 'Power(bhp)'] = train_data['Power'][i].split()[0]
    
    # Convert to numeric
    train_data['Mileage(km/kg)'] = train_data['Mileage(km/kg)'].astype(float)
    train_data['Engine(CC)'] = train_data['Engine(CC)'].astype(float)
    
    # Remove rows with 'null' in Power
    train_data = train_data[train_data['Power(bhp)'] != 'null']
    train_data = train_data.reset_index(drop=True)
    train_data['Power(bhp)'] = train_data['Power(bhp)'].astype(float)
    
    # Handle New_Price
    for i in range(train_data.shape[0]):
        if pd.isnull(train_data.loc[i,'New_Price']) == False:
            train_data.at[i,'New_car_Price'] = train_data['New_Price'][i].split()[0]
    
    train_data['New_car_Price'] = train_data['New_car_Price'].astype(float)
    
    # Drop original columns
    train_data.drop(["Name", "Mileage", "Engine", "Power", "New_Price"], axis=1, inplace=True)
    
    print("Encoding categorical variables...")
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Company']
    
    for col in categorical_columns:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        label_encoders[col] = le
    
    # Create dummy variables
    location_dummies = pd.get_dummies(train_data['Location'], prefix='Location')
    fuel_dummies = pd.get_dummies(train_data['Fuel_Type'], prefix='Fuel_Type')
    transmission_dummies = pd.get_dummies(train_data['Transmission'], prefix='Transmission')
    
    # Combine features
    final_train = pd.concat([
        train_data[['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats', 
                   'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)']],
        location_dummies,
        fuel_dummies,
        transmission_dummies
    ], axis=1)
    
    # Prepare features and target
    X = final_train.loc[:,['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats',
                          'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)',
                          'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore',
                          'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur',
                          'Location_Kochi', 'Location_Kolkata', 'Location_Mumbai',
                          'Location_Pune', 'Fuel_Type_Diesel', 'Fuel_Type_LPG',
                          'Fuel_Type_Petrol', 'Transmission_Manual']]
    
    y = train_data['Price']
    
    print("Training Random Forest model...")
    # Train Random Forest model
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X, y)
    
    # Evaluate model
    y_pred = rf_reg.predict(X)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model and encoders
    print("Saving model and encoders...")
    joblib.dump(rf_reg, 'models/random_forest_model.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    print("Model and encoders saved successfully!")
    print("Files saved:")
    print("- models/random_forest_model.joblib")
    print("- models/label_encoders.joblib")
    
    return rf_reg, label_encoders

if __name__ == "__main__":
    try:
        train_and_save_model()
    except FileNotFoundError:
        print("Error: train-data.csv not found!")
        print("Please ensure the training data file is in the current directory.")
    except Exception as e:
        print(f"Error during training: {str(e)}") 