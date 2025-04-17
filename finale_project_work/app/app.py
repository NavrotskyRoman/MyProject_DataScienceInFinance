from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from joblib import load
import os
import traceback
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"]}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and scaler as None
model = None
scaler = None

# Load the trained model and scaler
try:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(app_dir, 'models')
    
    model_path = os.path.join(models_dir, 'loan_default_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    
    logger.info(f"Looking for model at: {model_path}")
    logger.info(f"Looking for scaler at: {scaler_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    model = load(model_path)
    scaler = load(scaler_path)
    
    logger.info("Model and scaler loaded successfully")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Scaler type: {type(scaler)}")
    
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    logger.error(traceback.format_exc())
    raise  # Re-raise the exception to stop the application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/medium_risk')
def medium_risk():
    return render_template('medium_risk.html')

@app.route('/high_risk')
def high_risk():
    return render_template('high_risk.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            raise ValueError("Model or scaler not loaded properly")
            
        # Get form data
        data = request.form
        
        # Extract and process input features
        loan_amnt = float(data.get('loan_amnt', 0))
        int_rate = float(data.get('int_rate', 0)) / 100  # Convert percentage to decimal
        term_months = int(data.get('term_months', 36))
        annual_inc = float(data.get('annual_inc', 0))
        installment = float(data.get('installment', 0))
        
        # Calculate monthly payment if not provided
        if installment == 0:
            r = int_rate / 12  # Monthly interest rate
            installment = loan_amnt * (r * (1 + r)**term_months) / ((1 + r)**term_months - 1)
        
        # Calculate key features
        total_pymnt = installment * term_months
        loan_cost = total_pymnt - loan_amnt
        utilization_rate = total_pymnt / loan_amnt
        payment_efficiency = loan_amnt / total_pymnt
        
        # Create feature array in the correct order
        features = pd.DataFrame({
            'loan_cost': [loan_cost],
            'utilization_rate': [utilization_rate],
            'payment_efficiency': [payment_efficiency],
            'total_pymnt': [total_pymnt],
            'int_rate': [int_rate]
        })
        
        logger.info(f"Features before scaling: {features.to_dict(orient='records')[0]}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'features_used': features.to_dict(orient='records')[0]
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000) 