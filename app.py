from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Load the models and scaler
ensemble_model = tf.keras.models.load_model('ensemble_model.keras')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

CLASS_DESCRIPTIONS = {
    'Potato___Early_blight': 'Early blight, caused by the fungal pathogen Alternaria solani, presents a significant challenge to potato cultivation. This disease typically manifests as dark, circular lesions that appear on the older leaves, gradually expanding and developing concentric rings, resembling target patterns. As the infection progresses, the affected leaves exhibit yellowing around the lesions and may become brittle, leading to premature leaf drop. This deterioration severely impairs the plant’s photosynthetic efficiency, resulting in reduced vigor and crop yield. Favorable conditions for early blight include warm temperatures and high humidity, making it essential to implement proactive management strategies, such as crop rotation, resistant varieties, and timely fungicide applications, to mitigate its impact.',
    'Potato___healthy': 'Late blight, an exceptionally destructive disease caused by the oomycete Phytophthora infestans, is notorious for its rapid and devastating effects on potato crops. The disease typically begins with water-soaked lesions on leaves, which can quickly escalate into large, dark patches with fuzzy, grayish-white fungal growth underneath. Late blight can affect all plant parts, including stems and tubers, leading to extensive losses in both quality and quantity. This pathogen thrives in cool, moist conditions, making it particularly menacing during wet weather. The swift progression of late blight can result in total crop failure within a matter of days. Effective management is paramount, involving the use of resistant potato varieties, strategic fungicide applications, and careful monitoring of environmental conditions to prevent outbreaks.',
    'Potato___Late_blight': 'Healthy potato leaves are the hallmark of robust plant growth and development. These leaves are characterized by a vibrant green color, indicating strong photosynthetic activity and overall plant vitality. The surface is smooth and free from blemishes or lesions, reflecting an absence of disease and pest infestation. Healthy leaves are well-structured, providing optimal spacing for air circulation, which reduces humidity and the risk of fungal infections. They are crucial for energy production and nutrient absorption, directly correlating with tuber yield. Maintaining leaf health involves diligent cultural practices, including appropriate irrigation, fertilization, and pest management, all of which contribute to sustaining the overall health and productivity of the potato crop.'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_stream):
    img = tf.keras.preprocessing.image.load_img(img_stream, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_stream):
    img_array = preprocess_image(img_stream)
    cnn_features = ensemble_model.predict(img_array)
    rf_features = scaler.transform(cnn_features)
    rf_predictions = rf_model.predict(rf_features)
    predicted_class = CLASS_NAMES[rf_predictions[0]]
    return predicted_class  # Return the predicted class directly

def get_class_description(predicted_class):
    """Get the description of the predicted class."""
    return CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")

@app.route('/')
def index_page():
    """Serve the AI page."""
    return render_template('home.html')  # Ensure the file path is correct

@app.route('/home')
def home_page():
    """Serve the AI page."""
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/ai')
def ai_page():
    return render_template('ai.html')

@app.route('/faq')
def faq_page():
    return render_template('faq.html')

@app.route('/tnc')
def tnc_page():
    return render_template('tnc.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and run prediction."""
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        print("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Use BytesIO to handle the file in memory
        file_stream = BytesIO(file.read())
        filename = secure_filename(file.filename)

        try:
            # Read the image and convert to base64
            image_data = base64.b64encode(file_stream.getvalue()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"
            
            # Run the prediction on the in-memory file stream
            prediction = predict_image(file_stream)
            print(f"Prediction: {prediction}")

            # Convert prediction to user-friendly name
            if prediction == 'Potato___Early_blight':
                friendly_class_name = 'Early Blight'
            elif prediction == 'Potato___healthy':
                friendly_class_name = 'Healthy'
            elif prediction == 'Potato___Late_blight':
                friendly_class_name = 'Late Blight'
            else:
                friendly_class_name = 'Unknown'  # Fallback for unexpected classes

            description = get_class_description(prediction)

            # Pass the prediction to the result page
            return render_template('result.html', predicted_class=prediction, image_url=image_url, description=description, friendly_class_name=friendly_class_name)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction."
        
    else:
        print("Invalid file extension")
        return redirect(request.url)
    
if __name__ == '__main__':
    app.run(debug=True)
