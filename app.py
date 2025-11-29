import pickle
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Required for website testing

# Suppress warnings that might occur when loading the model or during prediction
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_FILE = 'model_pkl' 
# Feature names must match the order expected by the trained model
FEATURE_NAMES = ['pressure', 'temparature', 'dewpoint', 'humidity', 'windspeed']
# ---------------------

app = Flask(__name__)
# Enable CORS to allow your HTML file (which might run locally) to access this API
CORS(app) 

# Global variable to hold the loaded model
model = None

def load_model():
    """Loads the pickled scikit-learn model once when the server starts."""
    global model
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print(f"--- Successfully loaded model: {MODEL_FILE} ---")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_FILE}'. Check the 'Rainfall Project' folder.")
        model = None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None
        
        # --- NEW: Route to Serve the HTML File ---
@app.route('/')
def serve_html():
    # Flask requires HTML templates to be inside a folder named 'templates'.
    # This renders the rainfall_predictor.html file located inside the templates folder.
    return render_template('Rainfall_Predictor.html')

@app.route('/api/predict', methods=['POST'])
def predict_rainfall():
    """
    API endpoint that receives JSON input, makes a prediction, and returns JSON output.
    """
    if model is None:
        return jsonify({"status": "error", "message": "ML Model failed to load on server startup."}), 500

    try:
        # 1. Get data from POST request (JSON body)
        data = request.get_json(force=True)
        
        # 2. Extract features in the correct order
        input_data = [
            data.get('pressure'),
            data.get('temparature'),
            data.get('dewpoint'),
            data.get('humidity'),
            data.get('windspeed')
        ]

        # Basic data validation
        if not all(isinstance(x, (int, float)) for x in input_data) or len(input_data) != len(FEATURE_NAMES):
            return jsonify({
                "status": "error", 
                "message": "Invalid input format. Expected 5 numeric features: pressure, temparature, dewpoint, humidity, windspeed."
            }), 400

        # 3. Create DataFrame for prediction (Required by scikit-learn/pandas)
        data_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # 4. Make Prediction (assuming your model returns 0 or 1)
        prediction_result = model.predict(data_df)[0]
        
        # 5. Get Prediction Probabilities
        probabilities = model.predict_proba(data_df)[0]
        prob_no_rain = round(probabilities[0] * 100, 2)
        prob_rain = round(probabilities[1] * 100, 2)

        # 6. Format the response
        result_text = "YES, Expect Rainfall" if prediction_result == 1 else "NO Rainfall Expected"
        
        return jsonify({
            "status": "success",
            "prediction": result_text,
            "probabilities": {
                "No Rain": f"{prob_no_rain}%",
                "Rain": f"{prob_rain}%"
            }
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"status": "error", "message": f"An unexpected server error occurred during prediction: {str(e)}"}), 500

# Load model when the application starts
load_model()

if __name__ == '__main__':
    # Running in debug mode helps during development
    # Host '0.0.0.0' makes it accessible externally, good for testing environments
    app.run(debug=True, host='127.0.0.1', port=5000)
