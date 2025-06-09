import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load the model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hairfall_cnn_model.h5')
model = load_model(MODEL_PATH)

CLASS_LABELS = ['Phase 1', 'Phase 2', 'Phase 3']

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = int(np.argmax(prediction)) + 1

        probabilities = {
            label: float(prob) * 100 for label, prob in zip(CLASS_LABELS, prediction)
        }

        return jsonify({
            'phase': predicted_index,
            'phase_probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
