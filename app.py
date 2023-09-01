from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load your pre-trained CNN model
model = keras.models.load_model('model/model 2.2.h5')

# Define a function for image preprocessing
def preprocess_image(image):
    # Resize the image to match the input size of your model (e.g., 224x224 for many models)
    image = image.resize((100, 100))
    # Convert to NumPy array and normalize
    image = image.convert('L')
    image_array = np.asarray(image) / 255.0
    # Add a batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch

class_names = {
    0: 'Normal',
    1: 'Covid_positive',
    2: 'Lung_Opacity',
    3: 'Viral_Pneumonia'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    if image_file:
        try:
            # Open and preprocess the image
            image = Image.open(image_file)
            image_batch = preprocess_image(image)

            # Make a prediction using the model
            predictions = model.predict(image_batch)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            class_name = class_names[predicted_class]

            return jsonify({
                'class': class_name,
                'confidence': float(confidence)*100
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
