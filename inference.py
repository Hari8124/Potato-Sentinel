import tensorflow as tf
import joblib
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load the models and scaler
logger.debug("Loading models and scaler")
ensemble_model = tf.keras.models.load_model('ensemble_model.keras')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

CLASS_DESCRIPTIONS = {
    'Potato___Early_blight': 'Early blight description...',
    'Potato___healthy': 'Healthy description...',
    'Potato___Late_blight': 'Late blight description...'
}

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
        logger.debug("Starting prediction on image")
        img_array = preprocess_image(img_stream)
        cnn_features = ensemble_model.predict(img_array)
        rf_features = scaler.transform(cnn_features)
        rf_predictions = rf_model.predict(rf_features)
        predicted_class = CLASS_NAMES[rf_predictions[0]]
        logger.debug(f"Prediction result: {predicted_class}")
        return predicted_class  # Return the predicted class directly
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def get_class_description(predicted_class):
    """Get the description of the predicted class."""
    try:
        logger.debug(f"Fetching description for class: {predicted_class}")
        return CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")
    except Exception as e:
        logger.error(f"Error during description retrieval: {e}")
        raise
