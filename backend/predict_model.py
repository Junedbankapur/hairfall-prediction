import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hairfall_cnn_model.h5')
model = load_model(MODEL_PATH)

CLASS_LABELS = ['Phase 1', 'Phase 2', 'Phase 3']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((150, 150))  # Resize to model's expected shape
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 150, 150, 3)
    return img_array

def predict_image(image_bytes):
    img = preprocess_image(image_bytes)
    preds = model.predict(img)[0]  # shape (3,)

    phase_probabilities = {
        cls: float(prob) * 100 for cls, prob in zip(CLASS_LABELS, preds)
    }

    predicted_phase_index = int(np.argmax(preds)) + 1  # 1-based index

    return {
        'phase_probabilities': phase_probabilities,
        'phase': predicted_phase_index
    }
