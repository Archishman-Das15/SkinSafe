import os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Dummy model that always returns 69.0 as the confidence score
class DummyModel:
    def predict(self, image_array):
        # Return a confidence score of 69.0 for each class
        return np.array([[69.0] * 5])  # Assuming 5 classes (adjust based on your model)

# Initialize model with the dummy model
try:
    model = DummyModel()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Home route
@app.route('/')
def home():
    return "Welcome to the TeleDerm Backend!"

# Favicon route to prevent 404 errors for missing favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 204

# URL route for processing images and generating predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded or unavailable"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Preprocess the image
    image = image.resize((224, 224))  # Adjust based on your model's expected input size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction using the dummy model
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return jsonify({"predicted_class": str(predicted_class), "predictions": predictions.tolist()})

# Route to check model health/status
@app.route('/status', methods=['GET'])
def status():
    if model:
        return jsonify({"status": "Model is loaded and ready"})
    else:
        return jsonify({"status": "Model is not loaded or unavailable"}), 500

# Route to upload model
@app.route('/upload_model', methods=['POST'])
def upload_model():
    file = request.files.get('model_file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file_path = os.path.join('models', 'new_model.h5')
        file.save(file_path)

        global model
        model = DummyModel()  # Replace with actual model loading once available
        return jsonify({"message": "Model uploaded and loaded successfully!"})
    except Exception as e:
        return jsonify({"error": f"Error loading model: {str(e)}"}), 500

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
        
    app.run(debug=True)