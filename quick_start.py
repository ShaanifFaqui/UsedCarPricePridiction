#!/usr/bin/env python3
"""
Quick Start Script for Car Price Prediction Web Application
This script automates the setup and running process.
"""

import os
import sys
import subprocess
import time
import requests

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_file_exists(filename, description):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ {description} found: {filename}")
        return True
    else:
        print(f"❌ {description} not found: {filename}")
        return False

def wait_for_server(url, timeout=30):
    """Wait for the server to start"""
    print(f"\n🔄 Waiting for server to start at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/api/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is running at {url}")
                return True
        except:
            pass
        time.sleep(1)
    
    print(f"❌ Server failed to start within {timeout} seconds")
    return False

def main():
    print("🚗 Car Price Prediction Web Application - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Step 1: Check for training data
    if not check_file_exists("train-data.csv", "Training data"):
        print("\n📋 Please ensure train-data.csv is in the current directory.")
        print("   You can download it from your original dataset source.")
        return False
    
    # Step 2: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n💡 Try running: pip install -r requirements.txt manually")
        return False
    
    # Step 3: Check if model exists, if not train it
    if not check_file_exists("models/random_forest_model.joblib", "Trained model"):
        print("\n🤖 Model not found. Training new model...")
        if not run_command("python train_model.py", "Training model"):
            print("\n❌ Model training failed. Please check the error messages above.")
            return False
    else:
        print("✅ Trained model found, skipping training")
    
    # Step 4: Start the Flask application
    print("\n🚀 Starting Flask application...")
    print("   The application will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run(["python", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Quick start failed. Please check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 