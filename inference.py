import tensorflow as tf
import joblib
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure that debug messages are shown

# Load the models and scaler
logger.debug("Loading models and scaler")
try:
    ensemble_model = tf.keras.models.load_model('ensemble_model.keras')
    logger.debug("Ensemble model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ensemble model: {e}")
    raise

try:
    rf_model = joblib.load('rf_model.pkl')
    logger.debug("Random Forest model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading random forest model: {e}")
    raise

try:
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

def preprocess_image(img_stream):
    try:
        logger.debug("Starting image preprocessing...")
        
        # Load the image
        try:
            img = tf.keras.preprocessing.image.load_img(img_stream, target_size=(256, 256))
            logger.debug("Image loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
        
        # Convert image to array
        try:
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            logger.debug("Image converted to array successfully.")
        except Exception as e:
            logger.error(f"Error converting image to array: {e}")
            raise
        
        # Expand dimensions
        try:
            img_array = np.expand_dims(img_array, axis=0)
            logger.debug("Image dimensions expanded successfully.")
        except Exception as e:
            logger.error(f"Error expanding image dimensions: {e}")
            raise
        
        return img_array

    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def predict_image(img_stream):
    try:
        logger.debug("Starting prediction on image...")
        
        # Preprocess the image
        try:
            img_array = preprocess_image(img_stream)
            logger.debug("Image preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            raise
        
        # Make prediction with ensemble model
        try:
            cnn_features = ensemble_model.predict(img_array)
            logger.debug("Prediction from ensemble model successful.")
        except Exception as e:
            logger.error(f"Error during prediction with ensemble model: {e}")
            raise
        
        # Scale the prediction features
        try:
            rf_features = scaler.transform(cnn_features)
            logger.debug("Feature scaling successful.")
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}")
            raise
        
        # Predict using random forest model
        try:
            rf_predictions = rf_model.predict(rf_features)
            logger.debug("Prediction from random forest model successful.")
        except Exception as e:
            logger.error(f"Error during prediction with random forest model: {e}")
            raise
        
        # Get predicted class
        try:
            predicted_class = CLASS_NAMES[rf_predictions[0]]
            logger.debug(f"Prediction result: {predicted_class}")
        except Exception as e:
            logger.error(f"Error retrieving predicted class: {e}")
            raise
        
        return predicted_class  # Return the predicted class directly
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def get_class_description(predicted_class):
    try:
        logger.debug(f"Fetching description for class: {predicted_class}")
        description = CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")
        logger.debug(f"Description fetched: {description}")
        return description
    except Exception as e:
        logger.error(f"Error during description retrieval: {e}")
        raise
