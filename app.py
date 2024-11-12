from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from inference import predict_image, get_class_description  
from io import BytesIO
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to INFO or ERROR in production
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    logger.debug(f"Checking file extension for: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            if prediction == 'Potato___Early_blight':
                friendly_class_name = 'Early Blight'
            elif prediction == 'Potato___healthy':
                friendly_class_name = 'Healthy'
            elif prediction == 'Potato___Late_blight':
                friendly_class_name = 'Late Blight'
            else:
                friendly_class_name = 'Unknown'  # Fallback for unexpected classes
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
