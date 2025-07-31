# Car Price Prediction Application

This repository contains a complete car price prediction application with both machine learning models and a deployable web application. The application utilizes advanced machine learning techniques to predict used car prices with 91% accuracy.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python quick_start.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (if you have train-data.csv)
python train_model.py

# 3. Run the web application
python app.py
```

The application will be available at: **http://localhost:5000**

## 🌟 Features

### Web Application
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🔌 **REST API**: JSON endpoints for integration
- 📱 **Mobile Friendly**: Works on all devices
- ⚡ **Real-time Predictions**: Instant price predictions
- 🔒 **Input Validation**: Comprehensive data validation

### Machine Learning
- 🤖 **Random Forest Model**: 91% accuracy (R² score)
- 🔧 **Feature Engineering**: Advanced preprocessing
- 📊 **Data Visualization**: Insights from training data
- 🎯 **Model Persistence**: Saved models for deployment

## 📁 Project Structure

```
UsedCarPricePridiction/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── quick_start.py         # Automated setup script
├── test_api.py           # API testing script
├── requirements.txt       # Python dependencies
├── README_WEBAPP.md      # Detailed web app documentation
├── templates/
│   └── index.html        # Web interface template
├── models/               # Saved models (created after training)
│   ├── random_forest_model.joblib
│   └── label_encoders.joblib
├── img/                  # Visualization images
├── used_cars_price_detect.ipynb  # Original notebook
└── train-data.csv        # Training dataset
```

## 🔧 How it Works

1. **Data Collection**: Historical car sales data with features like make, model, year, mileage, location, etc.

2. **Data Preprocessing**: Cleaning, handling missing values, and feature engineering

3. **Feature Engineering**: Extracting meaningful features from raw data:
   - Company name from car name
   - Numeric values from string columns (mileage, engine, power)
   - Categorical encoding for locations, fuel types, etc.

4. **Model Training**: Random Forest algorithm trained on processed data

5. **Model Evaluation**: Cross-validation and performance metrics (R² = 0.91)

6. **Web Deployment**: Flask application with REST API and web interface

7. **Real-time Predictions**: Users can input car details and get instant price predictions

## 📊 Model Performance

- **R² Score**: 0.91 (91% accuracy)
- **Mean Absolute Error**: 1.52 lakhs
- **Root Mean Squared Error**: 3.24 lakhs

## 🛠️ API Endpoints

- `POST /api/predict` - Get car price prediction
- `GET /api/health` - Health check
- `GET /api/features` - Get available features

## 📖 Documentation

- **[Web Application Guide](README_WEBAPP.md)** - Complete web app documentation
- **[Training Process](training_process.md)** - Model development details

## 🎯 What Has Been Achieved

1. ✅ **Feature Engineering** - Advanced data preprocessing
2. ✅ **Data Visualization** - Insights from training data
3. ✅ **Model Building** - Random Forest with 91% accuracy
4. ✅ **Web Application** - Deployable Flask app
5. ✅ **REST API** - JSON endpoints for integration
6. ✅ **User Interface** - Modern, responsive web UI
7. ✅ **Model Persistence** - Saved models using joblib for production use 
