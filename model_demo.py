#!/usr/bin/env python3
"""
Model Saving and Loading Demo
This script demonstrates how to save and load machine learning models using joblib.
"""

import joblib
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

def create_sample_model():
    """Create a sample model for demonstration"""
    print("🤖 Creating sample model...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.rand(100) * 100  # Target values
    
    # Create and train a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create sample encoders
    encoders = {
        'location': LabelEncoder().fit(['Mumbai', 'Delhi', 'Bangalore']),
        'fuel_type': LabelEncoder().fit(['Petrol', 'Diesel', 'CNG'])
    }
    
    return model, encoders

def save_model_with_joblib(model, encoders, model_path='models/demo_model.joblib', encoders_path='models/demo_encoders.joblib'):
    """Save model and encoders using joblib"""
    print(f"💾 Saving model to {model_path}...")
    start_time = time.time()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save encoders
    joblib.dump(encoders, encoders_path)
    
    save_time = time.time() - start_time
    print(f"✅ Model saved successfully in {save_time:.3f} seconds")
    
    # Get file sizes
    model_size = os.path.getsize(model_path) / 1024  # KB
    encoders_size = os.path.getsize(encoders_path) / 1024  # KB
    
    print(f"📁 Model file size: {model_size:.1f} KB")
    print(f"📁 Encoders file size: {encoders_size:.1f} KB")

def load_model_with_joblib(model_path='models/demo_model.joblib', encoders_path='models/demo_encoders.joblib'):
    """Load model and encoders using joblib"""
    print(f"📂 Loading model from {model_path}...")
    start_time = time.time()
    
    # Load model
    model = joblib.load(model_path)
    
    # Load encoders
    encoders = joblib.load(encoders_path)
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.3f} seconds")
    
    return model, encoders

def test_model_prediction(model, encoders):
    """Test the loaded model with sample data"""
    print("🧪 Testing model prediction...")
    
    # Create sample input
    sample_input = np.random.rand(1, 5)
    
    # Make prediction
    prediction = model.predict(sample_input)[0]
    
    print(f"📊 Sample prediction: {prediction:.2f}")
    print(f"🔧 Model type: {type(model).__name__}")
    print(f"🔧 Number of encoders: {len(encoders)}")
    
    return prediction

def compare_joblib_vs_pickle():
    """Compare joblib vs pickle for model serialization"""
    print("\n" + "="*50)
    print("🔄 Comparing joblib vs pickle...")
    print("="*50)
    
    # Create sample model
    model, encoders = create_sample_model()
    
    # Test joblib
    print("\n📦 Testing joblib:")
    joblib_start = time.time()
    joblib.dump(model, 'models/test_joblib.joblib')
    joblib_model = joblib.load('models/test_joblib.joblib')
    joblib_time = time.time() - joblib_start
    
    # Test pickle
    import pickle
    print("\n🥒 Testing pickle:")
    pickle_start = time.time()
    with open('models/test_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/test_pickle.pkl', 'rb') as f:
        pickle_model = pickle.load(f)
    pickle_time = time.time() - pickle_start
    
    # Compare file sizes
    joblib_size = os.path.getsize('models/test_joblib.joblib') / 1024
    pickle_size = os.path.getsize('models/test_pickle.pkl') / 1024
    
    print(f"\n📊 Comparison Results:")
    print(f"   Joblib - Time: {joblib_time:.3f}s, Size: {joblib_size:.1f}KB")
    print(f"   Pickle - Time: {pickle_time:.3f}s, Size: {pickle_size:.1f}KB")
    print(f"   Joblib is {'faster' if joblib_time < pickle_time else 'slower'} and {'smaller' if joblib_size < pickle_size else 'larger'}")
    
    # Clean up test files
    os.remove('models/test_joblib.joblib')
    os.remove('models/test_pickle.pkl')

def main():
    """Main demonstration function"""
    print("🚗 Model Saving and Loading Demo with joblib")
    print("=" * 50)
    
    # Step 1: Create a sample model
    model, encoders = create_sample_model()
    
    # Step 2: Save the model
    save_model_with_joblib(model, encoders)
    
    # Step 3: Load the model
    loaded_model, loaded_encoders = load_model_with_joblib()
    
    # Step 4: Test the loaded model
    test_model_prediction(loaded_model, loaded_encoders)
    
    # Step 5: Compare with pickle
    compare_joblib_vs_pickle()
    
    print("\n" + "="*50)
    print("✅ Demo completed successfully!")
    print("💡 Key benefits of using joblib:")
    print("   - Better performance for large models")
    print("   - More efficient serialization")
    print("   - Better compatibility across Python versions")
    print("   - Built-in compression support")
    print("   - Designed specifically for NumPy arrays and scikit-learn models")

if __name__ == "__main__":
    main() 