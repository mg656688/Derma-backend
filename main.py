from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

# Define the classes for the predictions
classes = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr = image_arr / 255.0  # Normalizing the image
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    return image_arr

# Initialize Flask application
app = Flask(__name__)

@app.route('/DermaApp', methods=["POST"])
def api():
    try:
        # Get the image from post request
        if 'fileup' not in request.files:
            return jsonify({'error': "Please try again. The Image doesn't exist"}), 400

        image = request.files.get('fileup')
        image_arr = preprocess_image(image)

        # Make prediction using the loaded model
        result = model.predict(image_arr)
        ind = np.argmax(result)
        prediction = classes[ind]

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load the SavedModel
    try:
        print("Loading Model")
        model = tf.keras.models.load_model('DermaNet.keras')
        print("Model Loaded")
    except Exception as e:
        print(f"Error loading the model: {str(e)}")

    # Run Flask application
    app.run(host="0.0.0.0", debug=True)
