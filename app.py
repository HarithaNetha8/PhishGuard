import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import tensorflow as tf
from utils import extract_url_features, preprocess_image

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models
url_model = None
screenshot_model = None

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load trained models if they exist"""
    global url_model, screenshot_model
    
    try:
        # Load URL-based model
        if os.path.exists('models/random_forest_model.pkl'):
            url_model = joblib.load('models/random_forest_model.pkl')
            logging.info("URL model loaded successfully")
        else:
            logging.warning("URL model not found. Please run train_url_model.py first")
    except Exception as e:
        logging.error(f"Error loading URL model: {str(e)}")
    
    try:
        # Load screenshot-based model
        if os.path.exists('models/cnn_screenshot_model.h5'):
            screenshot_model = tf.keras.models.load_model('models/cnn_screenshot_model.h5')
            logging.info("Screenshot model loaded successfully")
        else:
            logging.warning("Screenshot model not found. Please run train_screenshot_model.py first")
    except Exception as e:
        logging.error(f"Error loading screenshot model: {str(e)}")

# Load models on startup
load_models()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Predict phishing based on URL features"""
    try:
        if not url_model:
            return jsonify({
                'error': 'URL model not loaded. Please train the model first.',
                'prediction': None
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Invalid request. Expected JSON with "features" array.',
                'prediction': None
            }), 400
        
        features = data['features']
        
        # Validate features
        if len(features) != 31:
            return jsonify({
                'error': f'Expected 31 features, got {len(features)}',
                'prediction': None
            }), 400
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = url_model.predict(features_array)[0]
        probability = url_model.predict_proba(features_array)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(max(probability)),
            'probabilities': {
                'safe': float(probability[0]),
                'phishing': float(probability[1])
            }
        })
        
    except Exception as e:
        logging.error(f"Error in URL prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'prediction': None
        }), 500

@app.route('/predict_screenshot', methods=['POST'])
def predict_screenshot():
    """Predict phishing based on website screenshot"""
    try:
        if not screenshot_model:
            return jsonify({
                'error': 'Screenshot model not loaded. Please train the model first.',
                'prediction': None
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'prediction': None
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'prediction': None
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, webp',
                'prediction': None
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            prediction_proba = screenshot_model.predict(processed_image)
            prediction = 1 if prediction_proba[0][0] > 0.5 else 0
            confidence = float(prediction_proba[0][0] if prediction == 1 else 1 - prediction_proba[0][0])
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'safe': float(1 - prediction_proba[0][0]),
                    'phishing': float(prediction_proba[0][0])
                }
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        logging.error(f"Error in screenshot prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'prediction': None
        }), 500

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features from a given URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Invalid request. Expected JSON with "url" field.',
                'features': None
            }), 400
        
        url = data['url']
        features = extract_url_features(url)
        
        return jsonify({
            'features': features,
            'feature_names': [
                'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
                'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
                'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
                'nb_comma', 'nb_semicolon', 'nb_dollar', 'nb_space', 'nb_www',
                'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
                'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain'
            ]
        })
        
    except Exception as e:
        logging.error(f"Error extracting features: {str(e)}")
        return jsonify({
            'error': f'Feature extraction error: {str(e)}',
            'features': None
        }), 500

@app.route('/train_models', methods=['POST'])
def train_models():
    """Endpoint to trigger model training"""
    try:
        model_type = request.json.get('model_type', 'both')
        
        if model_type in ['url', 'both']:
            # Import and run URL model training
            import train_url_model
            train_url_model.main()
        
        if model_type in ['screenshot', 'both']:
            # Import and run screenshot model training
            import train_screenshot_model
            train_screenshot_model.main()
        
        # Reload models
        load_models()
        
        return jsonify({
            'message': f'Successfully trained {model_type} model(s)',
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        return jsonify({
            'error': f'Training error: {str(e)}',
            'status': 'failed'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
