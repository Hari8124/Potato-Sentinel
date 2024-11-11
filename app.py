from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from inference import predict_image, get_class_description  
from io import BytesIO
import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            return render_template('result.html', predicted_class=prediction, image_url=image_url, description=description, friendly_class_name = friendly_class_name)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction."
        
    else:
        print("Invalid file extension")
        return redirect(request.url)
    
if __name__ == '__main__':
    app.run(debug=True)
