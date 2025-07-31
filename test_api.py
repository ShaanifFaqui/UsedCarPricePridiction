import requests
import json
import time

def test_api():
    """Test the car price prediction API"""
    
    base_url = "http://localhost:5000"
    
    # Test data
    test_car = {
        "Name": "Honda City",
        "Location": "Mumbai",
        "Year": 2015,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "15.2 km/kg",
        "Engine": "1197 CC",
        "Power": "88.7 bhp",
        "Seats": 5
    }
    
    print("🚗 Testing Car Price Prediction API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data['status']}")
            print(f"   Model Loaded: {health_data['model_loaded']}")
            print(f"   Encoders Loaded: {health_data['encoders_loaded']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {str(e)}")
    
    # Test 2: Get Features
    print("\n2. Testing Features Endpoint...")
    try:
        response = requests.get(f"{base_url}/api/features")
        if response.status_code == 200:
            features = response.json()
            print(f"✅ Features Retrieved Successfully")
            print(f"   Locations: {len(features['locations'])} cities")
            print(f"   Fuel Types: {len(features['fuel_types'])} types")
            print(f"   Transmissions: {len(features['transmissions'])} types")
        else:
            print(f"❌ Features Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Features Error: {str(e)}")
    
    # Test 3: Price Prediction
    print("\n3. Testing Price Prediction...")
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            headers={"Content-Type": "application/json"},
            json=test_car
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction Successful!")
            print(f"   Car: {test_car['Name']} ({test_car['Year']})")
            print(f"   Location: {test_car['Location']}")
            print(f"   Predicted Price: {result['predicted_price_formatted']}")
            print(f"   Raw Price: {result['predicted_price']} lakhs")
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
    
    # Test 4: Invalid Data
    print("\n4. Testing Invalid Data...")
    invalid_car = {
        "Name": "Test Car",
        "Location": "Mumbai",
        "Year": 2015
        # Missing required fields
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            headers={"Content-Type": "application/json"},
            json=invalid_car
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ Invalid Data Handled Correctly")
            print(f"   Error: {error_data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Invalid Data Not Handled: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid Data Test Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 API Testing Complete!")

if __name__ == "__main__":
    print("Make sure the Flask app is running on http://localhost:5000")
    print("Press Enter to start testing...")
    input()
    
    test_api() 