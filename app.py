from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained CNN model
model = tf.saved_model.load('serv')

# Define the categories
categories = ['Lung benign tissue', 'Lung adenocarcinoma', 'Lung squamous cell carcinoma',
              'Colon adenocarcinoma', 'Colon benign tissue']

# Define a function to preprocess the image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the input size of the model
    img = np.array(img) / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return img

# Define an API endpoint for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image = request.files['image']

    
    
    # Preprocess the image
    processed_image = preprocess_image(image.stream)
    
    # Make predictions using the loaded model
    predictions = model.serve(processed_image)
    predicted_category = categories[np.argmax(predictions)]
    
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)