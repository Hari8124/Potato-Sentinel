from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import tensorflow as tf
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to INFO or ERROR in production
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the models and scaler
try:
    logger.debug("Loading ensemble model...")
    ensemble_model = tf.keras.models.load_model('ensemble_model.keras')
    logger.debug("Ensemble model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ensemble model: {e}")
    raise

try:
    logger.debug("Loading random forest model...")
    rf_model = joblib.load('rf_model.pkl')
    logger.debug("Random Forest model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading random forest model: {e}")
    raise

try:
    logger.debug("Loading scaler...")
    scaler = joblib.load('scaler.pkl')
    logger.debug("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    raise

CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

CLASS_DESCRIPTIONS = {
    'Potato___Early_blight': 'Early blight description...',
    'Potato___healthy': 'Healthy description...',
    'Potato___Late_blight': 'Late blight description...'
}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    logger.debug(f"Checking file extension for: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_stream):
    try:
        logger.debug("Preprocessing image")
        img = tf.keras.preprocessing.image.load_img(img_stream, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def predict_image(img_stream):
    try:
        logger.debug("Starting prediction on image...")
        img_array = preprocess_image(img_stream)
        cnn_features = ensemble_model.predict(img_array)
        rf_features = scaler.transform(cnn_features)
        rf_predictions = rf_model.predict(rf_features)
        predicted_class = CLASS_NAMES[rf_predictions[0]]
        logger.debug(f"Prediction result: {predicted_class}")
        return predicted_class
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def get_class_description(predicted_class):
    try:
        logger.debug(f"Fetching description for class: {predicted_class}")
        description = CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")
        return description
    except Exception as e:
        logger.error(f"Error during description retrieval: {e}")
        raise

@app.route('/')
def index_page():
    """Serve the AI page."""
    logger.debug("Rendering home page")
    return render_template('home.html')

@app.route('/home')
def home_page():
    """Serve the home page."""
    logger.debug("Rendering home page")
    return render_template('home.html')

@app.route('/about')
def about_page():
    logger.debug("Rendering about page")
    return render_template('about.html')

@app.route('/ai')
def ai_page():
    logger.debug("Rendering AI page")
    return render_template('ai.html')

@app.route('/faq')
def faq_page():
    logger.debug("Rendering FAQ page")
    return render_template('faq.html')

@app.route('/tnc')
def tnc_page():
    logger.debug("Rendering Terms & Conditions page")
    return render_template('tnc.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and run prediction."""
    logger.debug("Handling file upload")
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']
    logger.debug(f"File selected: {file.filename}")

    if file.filename == '':
        logger.error("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        logger.debug(f"Valid file extension for {file.filename}")

        # Use BytesIO to handle the file in memory
        file_stream = BytesIO(file.read())
        filename = secure_filename(file.filename)
        logger.debug(f"File read into memory: {filename}")

        try:
            # Read the image and convert to base64
            image_data = base64.b64encode(file_stream.getvalue()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"
            logger.debug("Image converted to base64")

            # Run the prediction on the in-memory file stream
            prediction = predict_image(file_stream)
            logger.debug(f"Prediction received: {prediction}")

            # Convert prediction to user-friendly name
            friendly_class_name = {
                'Potato___Early_blight': 'Early Blight',
                'Potato___healthy': 'Healthy',
                'Potato___Late_blight': 'Late Blight'
            }.get(prediction, 'Unknown')
            logger.debug(f"Friendly class name: {friendly_class_name}")

            description = get_class_description(prediction)
            logger.debug(f"Class description: {description}")

            # Pass the prediction to the result page
            return render_template('result.html', predicted_class=prediction, image_url=image_url, description=description, friendly_class_name=friendly_class_name)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "An error occurred during prediction."
        
    else:
        logger.error(f"Invalid file extension: {file.filename}")
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
