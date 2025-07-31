# Car Price Prediction Web Application

A complete Flask web application for predicting used car prices using machine learning. This application provides both a web interface and REST API endpoints for car price predictions.

## Features

- 🚗 **Web Interface**: Modern, responsive UI for easy car price predictions
- 🔌 **REST API**: JSON endpoints for integration with other applications
- 🤖 **Machine Learning**: Random Forest model with 91% accuracy
- 📊 **Real-time Predictions**: Instant price predictions based on car specifications
- 🎨 **Modern UI**: Beautiful, user-friendly interface with Bootstrap 5
- 🔒 **Input Validation**: Comprehensive validation for all input fields
- 📱 **Mobile Responsive**: Works perfectly on all devices

## API Endpoints

### 1. Predict Car Price
- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body:**
```json
{
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
```

**Response:**
```json
{
    "predicted_price": 8.5,
    "predicted_price_formatted": "₹8.50",
    "input_data": {...}
}
```

### 2. Health Check
- **URL**: `/api/health`
- **Method**: `GET`

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "encoders_loaded": true
}
```

### 3. Get Available Features
- **URL**: `/api/features`
- **Method**: `GET`

**Response:**
```json
{
    "locations": ["Mumbai", "Pune", "Chennai", ...],
    "fuel_types": ["CNG", "Diesel", "Petrol", "LPG", "Electric"],
    "transmissions": ["Manual", "Automatic"],
    "owner_types": ["First", "Second", "Third", "Fourth & Above"],
    "year_range": {"min": 1998, "max": 2019},
    "seats_range": {"min": 2, "max": 10}
}
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Project
```bash
# If you have the project files, navigate to the directory
cd UsedCarPricePridiction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Training Data
Ensure you have the training data file:
- `train-data.csv` (should be in the project root directory)

### Step 4: Train and Save the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the training data
- Train the Random Forest model
- Save the model and encoders to `models/` directory

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at:
- **Web Interface**: http://localhost:5000
- **API Base URL**: http://localhost:5000/api

## Project Structure

```
UsedCarPricePridiction/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README_WEBAPP.md       # This file
├── templates/
│   └── index.html        # Web interface template
├── models/               # Saved models (created after training)
│   ├── random_forest_model.joblib
│   └── label_encoders.joblib
├── img/                  # Visualization images
├── used_cars_price_detect.ipynb  # Original notebook
└── train-data.csv        # Training dataset
```

## Usage Examples

### Using the Web Interface
1. Open http://localhost:5000 in your browser
2. Fill in the car details form
3. Click "Predict Price" to get instant results

### Using the API with curl
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using the API with Python
```python
import requests
import json

url = "http://localhost:5000/api/predict"
data = {
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

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted Price: {result['predicted_price_formatted']}")
```

## Model Performance

The Random Forest model achieves:
- **R² Score**: ~0.91 (91% accuracy)
- **Mean Absolute Error**: ~1.52 lakhs
- **Root Mean Squared Error**: ~3.24 lakhs

## Supported Features

### Car Specifications
- **Car Name**: Brand and model (e.g., "Honda City")
- **Location**: 11 major Indian cities
- **Year**: 1998-2019
- **Kilometers Driven**: Any positive number
- **Fuel Type**: Petrol, Diesel, CNG, LPG, Electric
- **Transmission**: Manual, Automatic
- **Owner Type**: First, Second, Third, Fourth & Above
- **Seats**: 2-10 seats
- **Mileage**: Format "X.X km/kg"
- **Engine**: Format "XXXX CC"
- **Power**: Format "XX.X bhp"

## Error Handling

The application includes comprehensive error handling:
- Input validation for all fields
- Proper error messages for missing data
- Network error handling
- Model loading error handling

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider:
1. Using a production WSGI server (Gunicorn, uWSGI)
2. Setting up a reverse proxy (Nginx)
3. Using environment variables for configuration
4. Implementing proper logging and monitoring

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure you've run `python train_model.py` first
   - Check that `models/` directory exists with model files
   - Models are saved as `.joblib` files for better compatibility

2. **Training data not found**
   - Ensure `train-data.csv` is in the project root directory

3. **Port already in use**
   - Change the port in `app.py` or kill the process using port 5000

4. **Dependencies not installed**
   - Run `pip install -r requirements.txt`

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For support or questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on the project repository 